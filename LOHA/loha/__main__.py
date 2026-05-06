"""`python -m loha` launches the web app."""

import uvicorn

from loha.config import HOST, PORT


def main() -> None:
    uvicorn.run("loha.server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
