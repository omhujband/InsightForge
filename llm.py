"""LLM access: one router with provider failover, plus small text helpers."""
from __future__ import annotations

import re
import time

import config


class LLMUnavailable(RuntimeError):
    """Raised when no configured provider could answer."""


class LLMRouter:
    COOLDOWN_S = 60

    def __init__(self, specs: list[config.ProviderSpec]):
        self._clients: list[tuple[str, object, str]] = []
        self._cooldown: dict[str, float] = {}
        self.init_errors: list[str] = []
        self.last_provider: str | None = None
        for spec in specs:
            try:
                self._clients.append((spec.label, self._build(spec), spec.model))
            except Exception as exc:  # missing package, bad arguments, ...
                self.init_errors.append(f"{spec.label}: {exc}")

    @staticmethod
    def _build(spec: config.ProviderSpec):
        if spec.kind == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=spec.model, google_api_key=spec.api_key, temperature=0)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=spec.model, api_key=spec.api_key, base_url=spec.base_url,
                          temperature=0, timeout=60, max_retries=0)

    @property
    def configured(self) -> bool:
        return bool(self._clients)

    @property
    def provider_names(self) -> list[str]:
        return [f"{label} · {model}" for label, _, model in self._clients]

    def invoke(self, prompt: str) -> str:
        """Return the model's text. Tries providers in order, skipping ones cooling down."""
        if not self._clients:
            raise LLMUnavailable("No LLM is configured. Add an API key to .env (see working.md).")
        now = time.time()
        ready = [c for c in self._clients if self._cooldown.get(c[0], 0) <= now]
        errors: list[str] = []
        for label, client, model in ready or self._clients:
            try:
                response = client.invoke(prompt)
                self.last_provider = f"{label} · {model}"
                return clean_llm_text(_content_to_text(response))
            except Exception as exc:
                msg = str(exc)
                errors.append(f"{label}: {msg[:200]}")
                if re.search(r"429|rate|quota|exhausted|overload", msg, re.I):
                    self._cooldown[label] = time.time() + self.COOLDOWN_S
        raise LLMUnavailable("All configured AI models failed:\n" + "\n".join(errors))


def _content_to_text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def clean_llm_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.split("</think>")[-1].strip()


def extract_code(text: str) -> str:
    """Pull code out of a fenced block if the model added one."""
    match = re.search(r"```(?:python|sql|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return (match.group(1) if match else text).strip()


def get_embedder():
    """Gemini embeddings if a Google key exists, else None (RAG falls back to TF-IDF)."""
    key = config.google_api_key()
    if not key:
        return None
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBED_MODEL, google_api_key=key)
    except Exception:
        return None
