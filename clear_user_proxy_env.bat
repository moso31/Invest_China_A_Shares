@echo off
setlocal

echo Clearing user-level proxy environment variables...

set "VARS=HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference = 'Stop';" ^
    "$vars = 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy';" ^
    "foreach ($name in $vars) { [Environment]::SetEnvironmentVariable($name, $null, 'User') };" ^
    "$codeProcesses = Get-Process -Name 'Code','Code - Insiders' -ErrorAction SilentlyContinue;" ^
    "if ($codeProcesses) { Write-Host 'VS Code is running. Closing it now...'; $codeProcesses | Stop-Process -Force } else { Write-Host 'VS Code is not running.' }"

if errorlevel 1 (
        echo.
        echo Failed to clear proxy variables or close VS Code.
        pause
        exit /b 1
)

echo.
echo Cleared user-level proxy variables:
for %%V in (%VARS%) do (
    echo   %%V
)

echo.
echo Reopen VS Code and any terminals to pick up the updated environment.
pause