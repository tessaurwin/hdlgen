# src/auto_tb_single.py

import argparse, json, pathlib, re, shutil, subprocess, tempfile
from collections import Counter
from pathlib import Path

def sh(cmd, cwd=None):
    # run a shell command, capture everything, don't spam the terminal
    p = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd)
    return p.returncode, p.stdout, p.stderr

def read_tasks(path: Path):
    # load tasks from either a JSON list file or JSONL lines
    txt = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(txt)            # if whole file is a JSON array, use it
        return data if isinstance(data, list) else []
    except Exception:
        pass                              # not a JSON array, try JSONL
    rows = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"): # ignore blanks and comments
            rows.append(json.loads(ln))   # each line is a JSON object
    return rows

def grab_decl_from_src(src: str):
    # best-effort: grab "module (...) ;" from the DUT so we can parse ports
    m = re.search(r"module\s+\w+\s*\(([^;]*)\)\s*;", src, flags=re.S|re.I)
    return f"module top({m.group(1)});" if m else ""

def parse_ports(decl: str):
    # take a minimal module header and figure out inputs, outputs, clk, and reset
    if not decl: return [], [], False, None
    m = re.search(r"\(([^;]*)\)", decl, flags=re.S|re.I)
    if not m: return [], [], False, None
    inside = " ".join(m.group(1).split())              # flatten whitespace
    toks = [t.strip() for t in inside.split(",") if t.strip()]
    ins, outs = [], []
    curw = ""                                          # remember last seen width like [7:0]
    for t in toks:
        if "input" in t:
            w = re.search(r"(\[[^\]]+\])", t)
            if w: curw = w.group(1)                    # update width context if present
            name = re.sub(r".*\binput\b", "", t)       # strip up to 'input'
            name = re.sub(r"\[[^\]]+\]", "", name).strip().split()[-1]
            ins.append((name, curw))
        elif "output" in t:
            w = re.search(r"(\[[^\]]+\])", t)
            if w: curw = w.group(1)
            name = re.sub(r".*\boutput\b", "", t)      # strip up to 'output'
            name = re.sub(r"\[[^\]]+\]", "", name).strip().split()[-1]
            outs.append((name, curw))
    in_names = [n for n,_ in ins]
    has_clk = any(n.lower() in ("clk","clock") for n in in_names)   # cheap clock detect
    rst = next((r for r in ("reset_n","rst_n","rst","reset") if r in in_names), None)  # common reset names
    return ins, outs, has_clk, rst

def build_tb(task_id, decl, ins, outs, has_clk, rst, dut_name, iters):
    # generate a tiny self-contained testbench for one DUT (device under test)
    lines = []
    if has_clk:
        # if the interface has a clock, make one and toggle it
        clk = "clk" if any(n=="clk" for n,_ in ins) else "clock"
        lines += [f"  reg {clk}=0;", f"  always #1 {clk}=~{clk};"]
    # declare inputs as regs (except the clock), outputs as wires
    for n,w in ins:
        if has_clk and n.lower() in ("clk","clock"): continue
        lines.append(f"  {'reg '+w if w else 'reg'} {n};")
    for n,w in outs:
        lines.append(f"  {'wire '+w if w else 'wire'} {n};")
    decls = ("\n".join(lines)) if lines else ""
    # connect by name, simple and effective
    conns = ", ".join([f".{n}({n})" for n,_ in ins+outs])
    # pack outputs into a bus so we can X/Z check in one line
    out_bus = "{ " + ", ".join([n for n,_ in outs]) + " }" if outs else "0"
    # randomize all inputs except clk/reset each iteration
    rand = "\n      ".join([f"{n} = $random;" for n,_ in ins if not (has_clk and n.lower() in ('clk','clock')) and n!=rst]) or "// no rand-able inputs"
    # how to advance time each step
    step = "@(posedge clk);" if has_clk else "#1;"
    # quick reset pulse if we have a reset pin
    rst_init = ""
    if rst:
        wait = "repeat (2) @(posedge clk);" if has_clk else "#2;"
        rst_init = f"  {rst} = 1'b1;\n  {wait}\n  {rst} = 1'b0;\n"
    # full testbench text
    return f"""// tb for {task_id}
`timescale 1ns/1ps
module tb;
{decls}

  {dut_name} dut({conns});

  integer i;
  initial begin
{rst_init}    for (i=0; i<{iters}; i=i+1) begin
      {rand}
      {step}
      if ({out_bus} !== {out_bus}) begin $display("FAIL_XZ %0d", i); $finish(2); end
    end
    $display("PASS");
    $finish(0);
  end
endmodule
"""

def try_run(work):
    # compile with iverilog and run with vvp, return a simple status and detail
    if not (shutil.which("iverilog") and shutil.which("vvp")):
        return "NO_SIM", "need iverilog and vvp"
    exe = work/"a.out"
    c,o,e = sh(["iverilog","-g2012","-o",str(exe),"tb.v","dut.v"], cwd=work)  # build
    if c!=0: return "COMPILE_FAIL", f"iverilog: {e or o}"
    c,o,e = sh([str(exe)], cwd=work)                                           # run
    if "PASS" in o: return "PASS","iverilog"
    if "FAIL_XZ" in o: return "FAIL_XZ", o.strip()
    return "RUNTIME_FAIL", (o.strip() or f"exit {c}")

def main():
    # cli flags: where tasks live, where DUTs live, where to write csv, how many iters, optional dump dir
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dump_dir", default="")
    args = ap.parse_args()

    tasks = read_tasks(Path(args.tasks))          # load task rows
    vdir = Path(args.dir)                         # folder of generated .v files
    dump_root = Path(args.dump_dir) if args.dump_dir else None
    if dump_root: dump_root.mkdir(parents=True, exist_ok=True)

    rows = ["task,status,detail"]                 # header for the csv
    counts = Counter()                            # live tally for the summary line

    for t in tasks:
        tid = t.get("id","?")                    # test id name, used to find <id>.v
        decl = t.get("module_declaration") or t.get("module_decl") or ""  # optional header from tasks file
        srcp = vdir/f"{tid}.v"                   # DUT path
        if not srcp.exists():
            # file missing, record and bounce
            status,detail="MISSING_FILE",str(srcp)
            rows.append(f"{tid},{status},{json.dumps(detail)}"); counts[status]+=1
            print(f"{tid}: {status}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"tb_{tid}_") as tmp:
            work = Path(tmp)                     # per-task scratch directory
            src = srcp.read_text(encoding="utf-8")
            # rename the first encountered module to 'dut' so our tb can instantiate by a fixed name
            src = re.sub(r"\bmodule\s+([A-Za-z_]\w*)\b", "module dut", src, count=1)
            (work/"dut.v").write_text(src, encoding="utf-8")
            # if no decl provided, try to scrape it from the DUT we just wrote
            if not decl: decl = grab_decl_from_src(src)
            # parse interface so we can wire up regs/wires and randomize inputs
            ins, outs, has_clk, rst = parse_ports(decl)
            # build testbench text and write it
            tb = build_tb(tid, decl, ins, outs, has_clk, rst, "dut", args.iters)
            (work/"tb.v").write_text(tb, encoding="utf-8")
            # compile + run
            status, detail = try_run(work)
            # optionally keep artifacts for inspection
            if dump_root:
                outdir = dump_root/tid
                if outdir.exists(): shutil.rmtree(outdir)
                shutil.copytree(work, outdir)

        # record row, update counts, print per-task status line
        rows.append(f"{tid},{status},{json.dumps(detail)}"); counts[status]+=1
        print(f"{tid}: {status}")

    # write the CSV with all results
    Path(args.out).write_text("\n".join(rows)+"\n", encoding="utf-8")
    # make and print the one-liner summary
    total = sum(counts.values())
    line = (f"TB Summary: PASS {counts.get('PASS',0)} | MISMATCH {counts.get('MISMATCH',0)} | "
            f"COMPILE_FAIL {counts.get('COMPILE_FAIL',0)} | UNKNOWN {counts.get('UNKNOWN',0)} | "
            f"NO_SIM {counts.get('NO_SIM',0)} | TOTAL {total}")
    print(line)
    # drop the same summary next to the CSV so we can paste it into slides without re-running
    Path(args.out).with_suffix("").with_name(Path(args.out).stem+"_summary.txt").write_text(line+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()