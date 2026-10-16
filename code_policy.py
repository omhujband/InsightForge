"""Static policy for LLM-generated pandas code (shared by parent and sandbox child).

This is defence in depth, NOT a security boundary on its own: the code also runs in a
separate process with a timeout, a memory limit and a stripped environment (sandbox.py).
"""
from __future__ import annotations

import ast
import builtins

MAX_CODE_CHARS = 8000


class CodePolicyError(ValueError):
    pass


_BLOCKED_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
                  ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
                  ast.Yield, ast.YieldFrom)

_BLOCKED_NAMES = frozenset({
    "open", "eval", "exec", "compile", "globals", "locals", "vars", "dir", "getattr",
    "setattr", "delattr", "hasattr", "input", "breakpoint", "help", "exit", "quit",
    "memoryview", "type", "super", "object", "classmethod", "staticmethod", "property",
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib",
    "importlib", "builtins", "pickle", "ctypes", "inspect",
})

_BLOCKED_ATTRS = frozenset({
    "eval", "query", "format", "format_map", "system", "popen", "load", "loads", "save",
    "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile",
    "memmap", "fromregex", "io", "compat", "util", "lib", "core", "ctypeslib", "f2py",
    "testing", "distutils", "os", "sys", "subprocess", "style",
})

_ALLOWED_TO = frozenset({
    "to_dict", "to_list", "to_numpy", "to_frame", "to_period", "to_timestamp", "to_string",
    "to_pydatetime", "to_series", "to_datetime", "to_numeric", "to_timedelta", "to_records",
})

_SAFE_BUILTINS = (
    "abs all any bool dict divmod enumerate filter float frozenset int isinstance len list "
    "map max min pow range repr reversed round set slice sorted str sum tuple zip "
    "Exception ValueError KeyError TypeError IndexError ZeroDivisionError"
).split()


def make_builtins() -> dict:
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTINS}
    safe["print"] = lambda *a, **k: None
    return safe


def validate_code(code: str) -> None:
    if not code or not code.strip():
        raise CodePolicyError("No code was produced.")
    if len(code) > MAX_CODE_CHARS:
        raise CodePolicyError(f"Code is too long (>{MAX_CODE_CHARS} characters).")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodePolicyError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from None

    assigns_result = False
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            raise CodePolicyError(f"`{type(node).__name__}` statements are not allowed "
                                  "(no imports; pd, np and math are already available).")
        if isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
                raise CodePolicyError(f"Use of `{node.id}` is not allowed.")
            if node.id == "result" and isinstance(node.ctx, ast.Store):
                assigns_result = True
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if (a.startswith("_") or a in _BLOCKED_ATTRS or a.startswith("read_")
                    or (a.startswith("to_") and a not in _ALLOWED_TO)):
                hint = " Use boolean masks instead of df.query()/eval." if a in ("query", "eval") else ""
                raise CodePolicyError(f"Attribute `.{a}` is not allowed.{hint}")
    if not assigns_result:
        raise CodePolicyError("The final answer must be assigned to a variable named `result`.")
