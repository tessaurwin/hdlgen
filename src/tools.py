# src/tools.py
import subprocess, pathlib, re

def _run(cmd: list[str]):
    p = subprocess.run(cmd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

def verible_lint(vfile: pathlib.Path):
    code, out, err = _run(["verible-verilog-lint", str(vfile)])
    warnings = len(re.findall(r"\[[A-Za-z0-9_-]+\]", out + err))
    ok = code == 0 or "syntax error" not in (out + err).lower()
    return ok, warnings, out + err

def verilator_compile(vfile: pathlib.Path):
    code, out, err = _run(["verilator", "--lint-only", str(vfile)])
    return code == 0, out + err

def yosys_synth(vfile: pathlib.Path, top: str):
    # quick synth gate: read + hierarchy + synth + stat
    script = f"read_verilog {vfile}; hierarchy -check -top {top}; synth -top {top}; stat"
    code, out, err = _run(["yosys", "-p", script])
    ok = (code == 0) and ("Printing statistics." in out)
    return ok, out + err
