# src/generate_with_lora.py
from pathlib import Path
import argparse, sys
from llm_generate import generate_verilog

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instruction", required=True)
    p.add_argument("--module_decl", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    code = generate_verilog(args.instruction, args.module_decl)
    if args.out:
        Path(args.out).write_text(code)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(code + "\n")

