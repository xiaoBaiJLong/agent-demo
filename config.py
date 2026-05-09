# config.py
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 30
    max_retries: int = 2



def load_env_file(env_path: Path | None = None) -> None:
    env_path = env_path or Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_settings() -> Settings:
    load_env_file()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "default-model")

    if not api_key:
        raise ValueError("Missing environment variable: LLM_API_KEY")

    if not base_url:
        raise ValueError("Missing environment variable: LLM_BASE_URL")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
