import os

from .source_registry import resolve_source

__all__ = ["load_dotenv_file", "resolve_source"]


def load_dotenv_file(dotenv_path: str = ".env") -> None:
    """Load key=value pairs from .env into process env if missing."""
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, encoding="utf-8") as env_file:
        for line in env_file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
