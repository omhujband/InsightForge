"""Central configuration: environment variables, limits and LLM provider specs."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
load_dotenv(BASE_DIR / ".env")

APP_TITLE = "InsightForge - AI Data Analyst"
ROLES = ("admin", "manager", "analyst", "viewer")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 50)
SANDBOX_TIMEOUT_S = _int("SANDBOX_TIMEOUT_S", 20)
SANDBOX_MAX_MEMORY_MB = _int("SANDBOX_MAX_MEMORY_MB", 2048)  # 0 disables (ignored on Windows)
MAX_REPAIR_ATTEMPTS = _int("MAX_REPAIR_ATTEMPTS", 2)
SQL_TIMEOUT_S = 10
SQL_MAX_ROWS = 5000

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "analyst").strip().lower()
if DEFAULT_ROLE not in ROLES:
    DEFAULT_ROLE = "analyst"


def _keys(*names: str) -> list[str]:
    found: list[str] = []
    for name in names:
        found += [k for k in re.split(r"[,;\s]+", os.getenv(name, "")) if k]
    return list(dict.fromkeys(found))


@dataclass(frozen=True)
class ProviderSpec:
    label: str
    kind: str  # "openai_compat" | "gemini"
    model: str
    api_key: str
    base_url: str | None = None


# name -> (kind, key env names, model env, default model, base url)
_PROVIDERS = {
    "groq": ("openai_compat", ("GROQ_API_KEYS", "GROQ_API_KEY"), "GROQ_MODEL",
             "openai/gpt-oss-120b", "https://api.groq.com/openai/v1"),
    "gemini": ("gemini", ("GOOGLE_API_KEYS", "GOOGLE_API_KEY"), "GEMINI_MODEL",
               "gemini-2.5-flash", None),
    "openrouter": ("openai_compat", ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
                   "OPENROUTER_MODEL", "openai/gpt-4o-mini", "https://openrouter.ai/api/v1"),
    "openai": ("openai_compat", ("OPENAI_API_KEYS", "OPENAI_API_KEY"), "OPENAI_MODEL",
               "gpt-4o-mini", "https://api.openai.com/v1"),
    "xai": ("openai_compat", ("XAI_API_KEYS", "XAI_API_KEY"), "XAI_MODEL",
            "grok-4", "https://api.x.ai/v1"),
}


def provider_specs() -> list[ProviderSpec]:
    """All configured providers/keys, in LLM_PROVIDER_ORDER (default: groq,gemini,...)."""
    order = [p.strip().lower() for p in os.getenv(
        "LLM_PROVIDER_ORDER", "groq,gemini,openrouter,openai,xai").split(",")]
    specs: list[ProviderSpec] = []
    for name in order:
        if name not in _PROVIDERS:
            continue
        kind, key_envs, model_env, default_model, base_url = _PROVIDERS[name]
        model = os.getenv(model_env, default_model)
        for i, key in enumerate(_keys(*key_envs), start=1):
            label = name if i == 1 else f"{name} (key {i})"
            specs.append(ProviderSpec(label, kind, model, key, base_url))
    return specs


def google_api_key() -> str | None:
    keys = _keys("GOOGLE_API_KEYS", "GOOGLE_API_KEY")
    return keys[0] if keys else None


GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
