"""Run LLM-generated pandas code in an isolated child process."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

import config
from code_policy import CodePolicyError, validate_code

_RUNNER = Path(__file__).resolve().parent / "sandbox_runner.py"


@dataclass
class SandboxResult:
    ok: bool
    value: Any = None
    kind: str = ""          # dataframe | scalar | json
    error: str = ""
    runtime_s: float = 0.0
    total_rows: int | None = None


def _child_env(workdir: str) -> dict:
    """Minimal environment: API keys and other secrets are NOT passed to the child."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
        "SANDBOX_MAX_MEMORY_MB": str(config.SANDBOX_MAX_MEMORY_MB),
        "TMPDIR": workdir, "TEMP": workdir, "TMP": workdir,
    }
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def run_code(code: str, df: pd.DataFrame, timeout: int | None = None) -> SandboxResult:
    try:
        validate_code(code)
    except CodePolicyError as exc:
        return SandboxResult(False, error=f"Policy: {exc}")

    timeout = timeout or config.SANDBOX_TIMEOUT_S
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aida_") as workdir:
        (Path(workdir) / "code.py").write_text(code, encoding="utf-8")
        df.to_pickle(Path(workdir) / "input.pkl")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(_RUNNER), workdir],
                capture_output=True, text=True, timeout=timeout, cwd=workdir,
                env=_child_env(workdir),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"Timeout: code ran longer than {timeout}s. "
                                              "Make it more efficient (avoid row-wise loops).",
                                 runtime_s=time.perf_counter() - start)
        elapsed = time.perf_counter() - start
        result_file = Path(workdir) / "result.json"
        if not result_file.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
            return SandboxResult(False, error=f"Sandbox crashed: {tail[0][:300]}", runtime_s=elapsed)
        payload = json.loads(result_file.read_text(encoding="utf-8"))

    if not payload.get("ok"):
        return SandboxResult(False, error=payload.get("error", "Unknown error"), runtime_s=elapsed)
    kind = payload["kind"]
    if kind == "dataframe":
        value = pd.read_json(StringIO(payload["data"]), orient="split")
        return SandboxResult(True, value, kind, runtime_s=elapsed, total_rows=payload.get("total_rows"))
    return SandboxResult(True, payload.get("value"), kind, runtime_s=elapsed)
