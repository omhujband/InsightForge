"""Child process that executes generated code. Started only by sandbox.py.

Reads <workdir>/input.pkl and <workdir>/code.py, writes <workdir>/result.json.
Results are returned as JSON (never pickle) so the parent never unpickles untrusted data.
"""
import json
import os
import sys
import traceback

MAX_ROWS = 1000


def _serialize(obj):
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else "value"
        if name == obj.index.name:
            name = f"{name}_value"
        obj = obj.rename(name).reset_index()
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            obj.columns = [" ".join(map(str, c)).strip() for c in obj.columns]
        if not isinstance(obj.index, pd.RangeIndex):
            obj = obj.reset_index()
        obj.columns = [str(c) for c in obj.columns]
        total = len(obj)
        data = obj.head(MAX_ROWS).to_json(orient="split", date_format="iso", default_handler=str)
        return {"kind": "dataframe", "data": data, "total_rows": total}
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"kind": "scalar", "value": obj}
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple, set, dict)):
        if isinstance(obj, set):
            obj = sorted(obj, key=str)
        return {"kind": "json", "value": json.loads(json.dumps(obj, default=str))}
    return {"kind": "scalar", "value": str(obj)}


def main(workdir: str) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(workdir, "result.json")
    code = open(os.path.join(workdir, "code.py"), encoding="utf-8").read()
    lines = code.splitlines()

    def fail(exc: BaseException) -> None:
        where = ""
        for fr in reversed(traceback.extract_tb(exc.__traceback__)):
            if fr.filename == "<generated>" and 0 < fr.lineno <= len(lines):
                where = f" (line {fr.lineno}: {lines[fr.lineno - 1].strip()[:120]})"
                break
        msg = f"{type(exc).__name__}: {str(exc)[:400]}{where}"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"ok": False, "error": msg}, fh)

    try:
        import math

        import numpy as np
        import pandas as pd

        from code_policy import make_builtins, validate_code

        df = pd.read_pickle(os.path.join(workdir, "input.pkl"))
        validate_code(code)
        try:  # POSIX only; silently skipped on Windows
            import resource
            mem = int(os.environ.get("SANDBOX_MAX_MEMORY_MB", "0"))
            if mem > 0:
                resource.setrlimit(resource.RLIMIT_AS, (mem << 20, mem << 20))
        except Exception:
            pass
        env = {"__builtins__": make_builtins(), "df": df, "pd": pd, "np": np, "math": math}
        exec(compile(code, "<generated>", "exec"), env)  # one namespace: lambdas can see df
        if env.get("result") is None:
            raise ValueError("`result` is None - the code did not produce an answer.")
        payload = {"ok": True, **_serialize(env["result"])}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        fail(exc)


if __name__ == "__main__":
    main(sys.argv[1])
