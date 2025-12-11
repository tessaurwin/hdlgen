# src/auto_tb_compare.py
import argparse, json, pathlib, re, shutil, subprocess, sys, tempfile
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sh(cmd, cwd=None):
    p = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    return p.returncode, p.stdout, p.stderr

def which(name): return shutil.which(name) is not None

def read_tasks_any(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list): return data
    except Exception:
        pass
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        rows.append(json.loads(ln))
    return rows

def parse_ports_from_decl(decl: str):
    if not decl: return [], [], False, None
    m = re.search(r"module\s+\w+\s*\((.*?)\)\s*;", decl, flags=re.S)
    if not m: return [], [], False, None
    inside = m.group(1).replace("\n"," ")
    parts = [p.strip() for p in inside.split(",") if p.strip()]
    cur_dir, cur_width = None, ""
    inputs, outputs = [], []
    for part in parts:
        if re.search(r"\binput\b", part):  cur_dir = "input"
        if re.search(r"\boutput\b", part): cur_dir = "output"
        w = re.search(r"(\[[^]]+\])", part)
        if w: cur_width = w.group(1)
        blob = re.sub(r"\b(input|output|wire|reg)\b", "", part)
        blob = re.sub(r"\[[^]]+\]", "", blob).strip()
        if not blob: continue
        for nm in re.split(r"\s+", blob):
            if not re.match(r"^[A-Za-z_]\w*$", nm): continue
            if cur_dir == "input":  inputs.append((nm, cur_width))
            if cur_dir == "output": outputs.append((nm, cur_width))
    in_names = [n for n,_ in inputs]
    has_clk = any(n.lower() in ("clk","clock") for n in in_names)
    rst_name = None
    for cand in ("reset_n","rst_n","rst","reset"):
        if cand in in_names: rst_name = cand; break
    return inputs, outputs, has_clk, rst_name

def width_decl(width: str, kind: str):
    return f"{kind} {width} " if width else f"{kind} "

def detect_module_name(src: str):
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)\b", src)
    return m.group(1) if m else None

def rename_module_text(src: str, new_name: str):
    orig = detect_module_name(src)
    if not orig: return None, src
    out = re.sub(rf"\bmodule\s+{re.escape(orig)}\b", f"module {new_name}", src, count=1)
    return orig, out

def build_tb(task_id, decl, inputs, outputs, has_clk, rst_name, a_name, b_name, iters):
    sigs, have_clk = [], False
    for n,w in inputs:
        if has_clk and n.lower() in ("clk","clock"):
            have_clk = True
            continue
        sigs.append(f"{width_decl(w,'reg')}{n};")
    for n,w in outputs:
        sigs.append(f"{width_decl(w,'wire')}{n}_a;")
    for n,w in outputs:
        sigs.append(f"{width_decl(w,'wire')}{n}_b;")
    decls = ("\n  " + "\n  ".join(sigs)) if sigs else ""

    clk_block = ""
    if has_clk:
        clkname = "clk"
        if any(n.lower()=="clock" for n,_ in inputs): clkname = "clock"
        clk_block = f"  reg {clkname}=0;\n  always #1 {clkname} = ~{clkname};\n"

    rst_init = ""
    if rst_name:
        wait = "repeat (2) @(posedge clk);" if has_clk else "#2;"
        rst_init = f"  {rst_name} = 1'b1;\n  {wait}\n  {rst_name} = 1'b0;\n"

    conns_a = []
    conns_b = []
    for n,_ in inputs:
        conns_a.append(f".{n}({n})")
        conns_b.append(f".{n}({n})")
    for n,_ in outputs:
        conns_a.append(f".{n}({n}_a)")
        conns_b.append(f".{n}({n}_b)")
    conns_a = ", ".join(conns_a)
    conns_b = ", ".join(conns_b)

    outA = "{ " + ", ".join([f"{n}_a" for n,_ in outputs]) + " }" if outputs else "0"
    outB = "{ " + ", ".join([f"{n}_b" for n,_ in outputs]) + " }" if outputs else "0"

    rand_lines = []
    for n,w in inputs:
        if has_clk and n.lower() in ("clk","clock"): continue
        if n == rst_name: continue
        rand_lines.append(f"{n} = $random;")
    rand_body = "\n      ".join(rand_lines) if rand_lines else "// no rand-able inputs"
    step = "@(posedge clk);" if has_clk else "#1;"

    tb = f"""// auto-generated compare TB for {task_id}
`timescale 1ns/1ps
module tb;
{clk_block}{decls}

  {a_name} dut_a({conns_a});
  {b_name} dut_b({conns_b});

  integer i;
  initial begin
{rst_init}    for (i=0; i<{iters}; i=i+1) begin
      {rand_body}
      {step}
      if ({outA} !== {outB}) begin
        $display("FAIL: mismatch at iter=%0d A=%0h B=%0h", i, {outA}, {outB});
        $finish(1);
      end
    end
    $display("PASS");
    $finish(0);
  end
endmodule
"""
    return tb

def try_sim(workdir: pathlib.Path):
    if which("iverilog") and which("vvp"):
        exe = workdir / "sim.out"
        code, out, err = sh(["iverilog", "-g2012", "-o", str(exe), "tb.v", "dut_a.v", "dut_b.v"], cwd=workdir)
        if code != 0: return ("COMPILE_FAIL", f"iverilog: {err or out}")
        code, out, err = sh([str(exe)], cwd=workdir)
        if "PASS" in out: return ("PASS", "iverilog")
        if "FAIL" in out: return ("MISMATCH", out.strip())
        return ("UNKNOWN", out.strip() or f"exit {code}")
    if which("verilator"):
        code, out, err = sh(["verilator", "-Wall", "--binary", "--top-module", "tb", "tb.v", "dut_a.v", "dut_b.v"], cwd=workdir)
        if code != 0: return ("COMPILE_FAIL", f"verilator: {err or out}")
        exe = workdir / "obj_dir" / "Vtb"
        code, out, err = sh([str(exe)], cwd=workdir)
        if "PASS" in out: return ("PASS", "verilator")
        if "FAIL" in out: return ("MISMATCH", out.strip() or f"exit {code}")
        return ("UNKNOWN", out.strip() or f"exit {code}")
    return ("NO_SIM", "need iverilog or verilator")

def run_one(task, dirA: pathlib.Path, dirB: pathlib.Path, workdir: pathlib.Path, iters: int):
    tid = task.get("id","?")
    decl = task.get("module_declaration") or task.get("module_decl") or ""
    if not decl:
        ap = dirA / f"{tid}.v"
        if ap.exists():
            m = re.search(r"(module\s+\w+\s*$begin:math:text$\[\^\;\]\*$end:math:text$;\s*)", ap.read_text(encoding="utf-8"), flags=re.S)
            if m: decl = m.group(1)
    inputs, outputs, has_clk, rst_name = parse_ports_from_decl(decl)

    a = dirA / f"{tid}.v"
    b = dirB / f"{tid}.v"
    if not a.exists() and not b.exists(): return ("MISSING_FILE", f"no DUTs: {a} and {b}")
    if not a.exists(): return ("MISSING_FILE", f"missing {a}")
    if not b.exists(): return ("MISSING_FILE", f"missing {b}")

    a_src = a.read_text(encoding="utf-8")
    b_src = b.read_text(encoding="utf-8")
    _, a_ren = rename_module_text(a_src, "dut_a_mod")
    _, b_ren = rename_module_text(b_src, "dut_b_mod")
    (workdir / "dut_a.v").write_text(a_ren, encoding="utf-8")
    (workdir / "dut_b.v").write_text(b_ren, encoding="utf-8")

    tb = build_tb(tid, decl, inputs, outputs, has_clk, rst_name, "dut_a_mod", "dut_b_mod", iters)
    (workdir / "tb.v").write_text(tb, encoding="utf-8")

    return try_sim(workdir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--dirA", required=True)
    ap.add_argument("--dirB", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tagA", default="A")
    ap.add_argument("--tagB", default="B")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dump_dir", default="")  # keep generated tb + duts
    args = ap.parse_args()

    tasks = read_tasks_any(pathlib.Path(args.tasks))
    out_csv = pathlib.Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = ["task,status,detail"]
    dirA = pathlib.Path(args.dirA)
    dirB = pathlib.Path(args.dirB)
    dump_root = pathlib.Path(args.dump_dir) if args.dump_dir else None
    if dump_root: dump_root.mkdir(parents=True, exist_ok=True)

    for t in tasks:
        tid = t.get("id","?")
        with tempfile.TemporaryDirectory(prefix=f"tb_{tid}_") as tmpd:
            work = pathlib.Path(tmpd)
            status, detail = run_one(t, dirA, dirB, work, args.iters)
            if dump_root:
                outdir = dump_root / tid
                if outdir.exists():
                    shutil.rmtree(outdir)
                shutil.copytree(work, outdir)
        rows.append(f"{tid},{status},{json.dumps(detail)}")

    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} with {len(rows)-1} rows.")

if __name__ == "__main__":
    main()