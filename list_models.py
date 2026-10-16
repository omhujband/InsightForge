#!/usr/bin/env python3
"""List the LLM models your API keys can use, and optionally test which ones really work.

Put this file next to your .env (the project folder) and run:

    python list_models.py                    # list models for every key found in .env
    python list_models.py --probe            # also send 1 tiny request to the top models
    python list_models.py --provider groq --probe --limit 15
    python list_models.py --provider gemini --models gemini-2.5-flash,gemini-2.5-pro --probe

What the APIs can and cannot tell you
  * OpenRouter marks free models in its list (price 0 / ":free")  -> shown as FREE.
  * Groq / Gemini / OpenAI / xAI list what your key can see but do NOT say whether the free tier
    covers a model -> shown as "?". Use --probe to find out: a model that answers is usable now;
    "NO FREE QUOTA" means the free tier gives that model zero requests.
Only the Python standard library is required (python-dotenv is used if installed). Keys are never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:  # optional: read the project's .env
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

PROVIDERS = {
    "groq": {"keys": ("GROQ_API_KEYS", "GROQ_API_KEY"), "base": "https://api.groq.com/openai/v1", "env": "GROQ_MODEL"},
    "gemini": {"keys": ("GOOGLE_API_KEYS", "GOOGLE_API_KEY"), "base": "https://generativelanguage.googleapis.com/v1beta", "env": "GEMINI_MODEL"},
    "openrouter": {"keys": ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"), "base": "https://openrouter.ai/api/v1", "env": "OPENROUTER_MODEL"},
    "openai": {"keys": ("OPENAI_API_KEYS", "OPENAI_API_KEY"), "base": "https://api.openai.com/v1", "env": "OPENAI_MODEL"},
    "xai": {"keys": ("XAI_API_KEYS", "XAI_API_KEY"), "base": "https://api.x.ai/v1", "env": "XAI_MODEL"},
}

# Models that are not chat/text generators (speech, embeddings, safety classifiers, images ...)
NON_CHAT = re.compile(r"whisper|tts|speech|guard|safeguard|embed|moderation|dall-e|image|imagen|veo|transcribe|"
                      r"audio|realtime|rerank|orpheus|playai|aqa|learnlm|robotics|computer-use", re.I)
# Heuristic: bigger / more capable families first when choosing what to probe
PREFER = re.compile(r"70b|120b|405b|gpt-oss|flash|pro|scout|maverick|mini|gpt-4|grok", re.I)


def first_key(env_names: tuple[str, ...]) -> str | None:
    for name in env_names:
        for k in re.split(r"[,;\s]+", os.getenv(name, "")):
            if k:
                return k
    return None


def http(method: str, url: str, headers: dict | None = None, body: dict | None = None, timeout: int = 30):
    """Returns (status_code, parsed_json_or_text). Never raises on HTTP errors."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": "list-models/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read().decode("utf-8", "replace"), resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode("utf-8", "replace"), exc.code
    except Exception as exc:  # DNS, timeout, SSL ...
        return 0, f"{type(exc).__name__}: {exc}"
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def _auth(provider: str, key: str) -> dict:
    return {"x-goog-api-key": key} if provider == "gemini" else {"Authorization": f"Bearer {key}"}


def _msg(payload) -> str:
    if isinstance(payload, dict):
        err = payload.get("error", payload)
        if isinstance(err, dict):
            return str(err.get("message", err))[:200]
        return str(err)[:200]
    return str(payload)[:200]


# ----------------------------------------------------------------------------- listing
def list_models(provider: str, key: str):
    """Returns (error_or_None, [ {id, free, note} ])."""
    base = PROVIDERS[provider]["base"]
    out: list[dict] = []
    if provider == "gemini":
        token = ""
        while True:
            status, data = http("GET", f"{base}/models?pageSize=200{token}", _auth(provider, key))
            if status != 200:
                return f"HTTP {status}: {_msg(data)}", []
            for m in data.get("models", []):
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                mid = m["name"].removeprefix("models/")
                out.append({"id": mid, "free": "?", "note": m.get("displayName", "")})
            if not data.get("nextPageToken"):
                break
            token = f"&pageToken={data['nextPageToken']}"
    else:
        status, data = http("GET", f"{base}/models", _auth(provider, key))
        if status != 200:
            return f"HTTP {status}: {_msg(data)}", []
        for m in data.get("data", []):
            if m.get("active") is False:  # Groq flags retired models
                continue
            mid, free, note = m["id"], "?", ""
            if provider == "openrouter":
                price = m.get("pricing", {})
                is_free = str(price.get("prompt")) == "0" and str(price.get("completion")) == "0"
                free = "FREE" if is_free else "paid"
                note = f"{m.get('context_length', '')} ctx"
            elif m.get("context_window"):
                note = f"{m['context_window']} ctx"
            out.append({"id": mid, "free": free, "note": note})
    out = [m for m in out if not NON_CHAT.search(m["id"])]
    out.sort(key=lambda m: (0 if PREFER.search(m["id"]) else 1, m["id"]))
    return None, out


# ----------------------------------------------------------------------------- probing
def probe(provider: str, key: str, model: str) -> tuple[str, str]:
    """One tiny request. Returns (verdict, detail)."""
    base = PROVIDERS[provider]["base"]
    if provider == "gemini":
        url = f"{base}/models/{model}:generateContent"
        body = {"contents": [{"parts": [{"text": "Reply with the single word OK."}]}],
                "generationConfig": {"maxOutputTokens": 16}}
    else:
        url = f"{base}/chat/completions"
        body = {"model": model, "messages": [{"role": "user", "content": "Reply with the single word OK."}], "max_tokens": 16}
    status, data = http("POST", url, _auth(provider, key), body)
    msg = _msg(data)
    if status == 200:
        return "WORKS", ""
    if status == 429:
        zero = re.search(r"limit:\s*0|free tier.*(not|no)|exceeded your current quota", msg, re.I)
        return ("NO FREE QUOTA" if zero else "RATE-LIMITED"), msg
    if status in (401, 403):
        return "KEY REJECTED", msg
    if status in (400, 404):
        return "UNAVAILABLE", msg
    return "ERROR", f"HTTP {status}: {msg}"


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List / test the models your API keys can use.")
    ap.add_argument("--provider", choices=list(PROVIDERS), help="only this provider (default: all with a key)")
    ap.add_argument("--probe", action="store_true", help="send one tiny request per model to see what really works")
    ap.add_argument("--limit", type=int, default=8, help="max models to probe per provider (default 8)")
    ap.add_argument("--models", help="comma-separated model ids to probe instead of the top ones")
    ap.add_argument("--all", action="store_true", help="print every model, not just the first 40")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between probe requests (default 1.5)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    names = [args.provider] if args.provider else list(PROVIDERS)
    suggestions: list[str] = []
    found_any = False
    for prov in names:
        key = first_key(PROVIDERS[prov]["keys"])
        if not key:
            if args.provider:
                print(f"No key found for {prov} (set {PROVIDERS[prov]['keys'][-1]} in .env).")
            continue
        found_any = True
        print(f"\n=== {prov.upper()} " + "=" * (60 - len(prov)))
        err, models = list_models(prov, key)
        if err:
            print(f"  Could not list models: {err}")
            continue
        free_n = sum(1 for m in models if m["free"] == "FREE")
        print(f"  {len(models)} text models visible to your key" + (f" ({free_n} marked FREE)" if prov == "openrouter" else ""))
        shown = models if args.all else models[:40]
        for m in shown:
            print(f"  - {m['id']:<48} {m['free']:<5} {m['note']}")
        if len(models) > len(shown):
            print(f"  ... {len(models) - len(shown)} more (use --all)")

        if args.probe:
            wanted = [x.strip() for x in args.models.split(",")] if args.models else None
            if prov == "openrouter" and not wanted:
                pool = [m["id"] for m in models if m["free"] == "FREE"]
            else:
                pool = wanted or [m["id"] for m in models]
            targets = pool[:args.limit] if not wanted else pool
            print(f"\n  Probing {len(targets)} model(s) with a 1-request test ...")
            first_ok = None
            for i, mid in enumerate(targets):
                verdict, detail = probe(prov, key, mid)
                print(f"  [{verdict:<13}] {mid}" + (f"  - {detail[:90]}" if detail else ""))
                if verdict == "WORKS" and first_ok is None:
                    first_ok = mid
                if verdict == "KEY REJECTED":
                    break
                if i < len(targets) - 1:
                    time.sleep(args.delay)
            if first_ok:
                suggestions.append(f"{PROVIDERS[prov]['env']}={first_ok}")

    if not found_any:
        print("No API keys found. Add GROQ_API_KEY / GOOGLE_API_KEY / ... to your .env next to this script.")
        return 1
    if suggestions:
        print("\nFirst working model per provider - paste into .env to use it:")
        print("  " + "\n  ".join(suggestions))
    elif not args.probe:
        print("\nTip: run again with --probe to test which of these models actually answer on your key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())