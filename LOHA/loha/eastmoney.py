"""Helpers for Eastmoney request hardening.

Eastmoney's push2/push2his hosts often close raw ``requests`` clients before
returning an HTTP response.  LOHA installs this small patch before importing
AKShare so Eastmoney requests:

* ignore the local proxy environment;
* carry the optional anonymous nid18 cookie;
* use curl_cffi's Chrome impersonation when available;
* stop retrying for a while after repeated failures.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests

from loha import request_stats

EASTMONEY_NID18_ENV_VAR = "LOHA_EASTMONEY_NID18"
EASTMONEY_COOLDOWN_SECONDS_ENV_VAR = "LOHA_EASTMONEY_COOLDOWN_SECONDS"
EASTMONEY_FAILURE_THRESHOLD_ENV_VAR = "LOHA_EASTMONEY_FAILURE_THRESHOLD"
EASTMONEY_TRANSPORT_ENV_VAR = "LOHA_EASTMONEY_TRANSPORT"
EASTMONEY_IMPERSONATE_ENV_VAR = "LOHA_EASTMONEY_IMPERSONATE"
IGNORE_PROXY_ENV_VAR = "LOHA_IGNORE_PROXY_ENV"

DEFAULT_COOLDOWN_SECONDS = 300.0
DEFAULT_FAILURE_THRESHOLD = 2
DEFAULT_IMPERSONATE = "chrome136"
NID18_FILE = Path(__file__).resolve().parent.parent / "data" / "eastmoney_nid18.txt"
PUSH2HIS_KLINE_HOSTS = (
    "33.push2his.eastmoney.com",
    "63.push2his.eastmoney.com",
    "72.push2his.eastmoney.com",
)

_DEFAULT_EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/",
    "Connection": "close",
}

_PATCH_INSTALLED = False
_ORIGINAL_REQUEST = requests.sessions.Session.request
_STATE_LOCK = threading.Lock()
_CIRCUIT_STATE: dict[str, dict[str, float]] = {}

log = logging.getLogger("loha.eastmoney")


class EastmoneyCooldownError(requests.exceptions.ConnectionError):
    """Raised when an Eastmoney endpoint family is temporarily cooled down."""


def _configured_nid18() -> str:
    return os.getenv(EASTMONEY_NID18_ENV_VAR, "").strip()


def nid18_configured() -> bool:
    """Return whether the current process has an Eastmoney nid18 configured."""
    return bool(_configured_nid18())


def set_nid18_for_process(value: str) -> None:
    """Update nid18 for the currently running LOHA process only."""
    value = value.strip()
    if value:
        os.environ[EASTMONEY_NID18_ENV_VAR] = value
        try:
            NID18_FILE.parent.mkdir(parents=True, exist_ok=True)
            NID18_FILE.write_text(value, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to persist Eastmoney nid18: %s", exc)
    else:
        os.environ.pop(EASTMONEY_NID18_ENV_VAR, None)
        try:
            if NID18_FILE.exists():
                NID18_FILE.unlink()
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to remove persisted Eastmoney nid18: %s", exc)


def load_persisted_nid18() -> bool:
    """Load nid18 from the local data file unless the process env already has one."""
    if _configured_nid18():
        return True
    try:
        value = NID18_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read persisted Eastmoney nid18: %s", exc)
        return False
    if not value:
        return False
    os.environ[EASTMONEY_NID18_ENV_VAR] = value
    return True


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_eastmoney_host(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "eastmoney.com" or host.endswith(".eastmoney.com")


def _transport_mode() -> str:
    return os.getenv(EASTMONEY_TRANSPORT_ENV_VAR, "curl_cffi").strip().lower()


def _ignore_proxy_env() -> bool:
    return os.getenv(IGNORE_PROXY_ENV_VAR, "1").strip().lower() not in {"0", "false", "no"}


def _impersonate() -> str:
    return os.getenv(EASTMONEY_IMPERSONATE_ENV_VAR, DEFAULT_IMPERSONATE).strip() or DEFAULT_IMPERSONATE


def _circuit_key(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path
    if "push2his" in host and path.endswith("/api/qt/stock/kline/get"):
        return "push2his.kline"
    if "push2his" in host:
        return "push2his"
    if "push2" in host and path.endswith("/api/qt/stock/get"):
        return "push2.stock"
    if "push2" in host and path.endswith("/api/qt/clist/get"):
        return "push2.clist"
    if "push2" in host:
        return "push2"
    if host == "datacenter-web.eastmoney.com":
        return "datacenter-web"
    return host or "eastmoney"


def _cooldown_seconds() -> float:
    return max(0.0, _env_float(EASTMONEY_COOLDOWN_SECONDS_ENV_VAR, DEFAULT_COOLDOWN_SECONDS))


def _failure_threshold() -> int:
    return max(1, _env_int(EASTMONEY_FAILURE_THRESHOLD_ENV_VAR, DEFAULT_FAILURE_THRESHOLD))


def _check_circuit(key: str) -> None:
    now = time.monotonic()
    with _STATE_LOCK:
        state = _CIRCUIT_STATE.get(key)
        if not state:
            return
        cool_until = float(state.get("cool_until", 0.0))
        if cool_until <= now:
            return
        remaining = cool_until - now
    raise EastmoneyCooldownError(f"Eastmoney {key} is cooling down for {remaining:.0f}s")


def _record_success(key: str) -> None:
    with _STATE_LOCK:
        if key in _CIRCUIT_STATE:
            _CIRCUIT_STATE.pop(key, None)


def _record_failure(key: str, exc: BaseException) -> None:
    threshold = _failure_threshold()
    cooldown = _cooldown_seconds()
    with _STATE_LOCK:
        state = _CIRCUIT_STATE.setdefault(key, {"failures": 0.0, "cool_until": 0.0})
        failures = int(state.get("failures", 0.0)) + 1
        state["failures"] = float(failures)
        if failures < threshold or cooldown <= 0:
            return
        state["cool_until"] = time.monotonic() + cooldown
        state["failures"] = 0.0
    log.warning(
        "Eastmoney %s failed repeatedly; cooling down for %.0fs: %s",
        key,
        cooldown,
        exc,
    )


def _merge_cookie(existing: str | None, name: str, value: str) -> str:
    if existing and f"{name}=" in existing:
        return existing
    if existing:
        return f"{existing.rstrip('; ')}; {name}={value}"
    return f"{name}={value}"


def _merge_headers(headers: dict[str, str] | None) -> dict[str, str]:
    merged = dict(headers or {})
    existing = {key.lower() for key in merged}
    for key, value in _DEFAULT_EASTMONEY_HEADERS.items():
        if key.lower() not in existing:
            merged[key] = value
    return merged


def _eastmoney_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host == "push2his.eastmoney.com" and parsed.path.endswith("/api/qt/stock/kline/get"):
        urls: list[str] = []
        for fallback_host in PUSH2HIS_KLINE_HOSTS:
            netloc = fallback_host
            if parsed.port:
                netloc = f"{fallback_host}:{parsed.port}"
            urls.append(urlunparse(parsed._replace(netloc=netloc)))
        urls.append(url)
        return urls
    return [url]


def _response_ok(response: object) -> bool:
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    try:
        return int(status_code) < 400
    except (TypeError, ValueError):
        return True


def _curl_cffi_request(method: str, url: str, kwargs: dict) -> object:
    try:
        from curl_cffi import requests as curl_requests
    except Exception as exc:  # noqa: BLE001
        raise requests.exceptions.ConnectionError(f"curl_cffi unavailable: {exc}") from exc

    allowed = {
        "params",
        "data",
        "json",
        "headers",
        "cookies",
        "timeout",
        "allow_redirects",
        "verify",
    }
    curl_kwargs = {key: value for key, value in kwargs.items() if key in allowed}
    curl_kwargs["proxies"] = {}
    last_exc: BaseException | None = None
    for candidate in _eastmoney_urls(url):
        try:
            response = curl_requests.request(
                method,
                candidate,
                impersonate=_impersonate(),
                **curl_kwargs,
            )
            request_stats.record(candidate, _response_ok(response))
            return response
        except Exception as exc:  # noqa: BLE001
            request_stats.record(candidate, False)
            last_exc = exc
    raise requests.exceptions.ConnectionError(str(last_exc)) from last_exc


def install_requests_cookie_patch() -> None:
    """Install a request patch for Eastmoney calls made through AKShare."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    @wraps(_ORIGINAL_REQUEST)
    def patched_request(self, method, url, *args, **kwargs):
        if _ignore_proxy_env():
            self.trust_env = False
            kwargs["proxies"] = {}

        if not isinstance(url, str) or not _is_eastmoney_host(url):
            try:
                response = _ORIGINAL_REQUEST(self, method, url, *args, **kwargs)
                if isinstance(url, str):
                    request_stats.record(url, _response_ok(response))
                return response
            except Exception:
                if isinstance(url, str):
                    request_stats.record(url, False)
                raise

        key = _circuit_key(url)
        _check_circuit(key)

        headers = _merge_headers(kwargs.get("headers"))
        nid18 = _configured_nid18()
        if nid18:
            headers["Cookie"] = _merge_cookie(headers.get("Cookie"), "nid18", nid18)
        kwargs["headers"] = headers

        try:
            if method.upper() == "GET" and _transport_mode() != "requests":
                response = _curl_cffi_request(method, url, kwargs)
            else:
                response = _ORIGINAL_REQUEST(self, method, url, *args, **kwargs)
                request_stats.record(url, _response_ok(response))
            _record_success(key)
            return response
        except EastmoneyCooldownError:
            request_stats.record(url, False)
            raise
        except Exception as exc:  # noqa: BLE001
            if not (method.upper() == "GET" and _transport_mode() != "requests"):
                request_stats.record(url, False)
            _record_failure(key, exc)
            if isinstance(exc, requests.exceptions.RequestException):
                raise
            raise requests.exceptions.ConnectionError(str(exc)) from exc

    requests.sessions.Session.request = patched_request
    _PATCH_INSTALLED = True
