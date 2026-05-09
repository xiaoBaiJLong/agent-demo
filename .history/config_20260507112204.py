# config.py
import os
from dataclasses import dataclass


@dataclass
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 30
    max_retries: int = 2



def load_settings() -> Settings:
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