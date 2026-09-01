#!/usr/bin/env python3
"""Run every verification script and print a summary table.

    python3 run_all.py            full run   (~15-25 min)
    python3 run_all.py --fast     quick run  (~3 min; skips the heaviest checks)
"""
import subprocess, sys, time, glob
fast = "--fast" in sys.argv
scripts = sorted(glob.glob("check*.py"))
rows = []
for s in scripts:
    cmd = [sys.executable, s] + (["--fast"] if fast else [])
    print(f"\n{'='*72}\n{s}\n{'='*72}")
    t0 = time.time()
    r = subprocess.run(cmd)
    rows.append((s, r.returncode == 0, time.time()-t0))
print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
for s, ok, dt in rows:
    print(f"  {'PASS' if ok else 'FAIL'}  {s:36s} {dt:7.1f}s")
bad = [s for s, ok, _ in rows if not ok]
print(f"\n  {len(rows)-len(bad)}/{len(rows)} scripts passed")
sys.exit(1 if bad else 0)
