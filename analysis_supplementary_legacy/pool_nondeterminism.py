# -*- coding: utf-8 -*-
"""Demonstrate that the pair/triad candidate pool is not deterministic.

The pool is built as list(set_of_feature_id_strings)[:CAP]. Python randomises
string hashing per process, so the iteration order of that set - and therefore
which metabolites survive the cap - changes between runs.
"""
import os, sys, subprocess, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r'''
import pandas as pd, json, sys
UNI = r"{uni}"
TARGETS = ["Inhibition (%)-NF", "Inhibition (%)-NV", "Inhibition (%)-NM", "Inhibition (%)-NU"]
xl = pd.ExcelFile(UNI)
def topk(case, k=10):
    for sh in xl.sheet_names:
        if case.replace("%", "pct").replace("(", "").replace(")", "")[:14] in sh.replace("(", "").replace(")", ""):
            d = xl.parse(sh)
            col = d.columns[0]
            brs = [c for c in d.columns if str(c).lower() == "brs"]
            if brs:
                d = d.sort_values(brs[0], ascending=False)
            return d[col].astype(str).head(k).tolist()
    return []
union = set()
for c in TARGETS:
    union.update(topk(c))
print(json.dumps({{"n_union": len(union), "cap5": list(union)[:5], "cap10": list(union)[:10]}}))
'''.replace("{uni}", str(ROOT / "results" / "univariate_BRS.xlsx"))

tmp = ROOT / "_pooltest_tmp.py"
tmp.write_text(CHILD, encoding="utf-8")

runs = []
for i in range(5):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONHASHSEED", None)          # default: randomised per process
    r = subprocess.run([sys.executable, tmp.name], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        print("child failed:", r.stderr[-400:]); break
    runs.append(json.loads(line[-1]))

print(f"union size: {runs[0]['n_union']} metabolites\n")
print("pool of 5 selected by list(set)[:5], across 5 independent runs:")
for i, r in enumerate(runs):
    print(f"   run {i}: {sorted(x[:22] for x in r['cap5'])}")

distinct5 = {tuple(sorted(r["cap5"])) for r in runs}
distinct10 = {tuple(sorted(r["cap10"])) for r in runs}
print(f"\ndistinct cap-5  pools over 5 runs: {len(distinct5)}")
print(f"distinct cap-10 pools over 5 runs: {len(distinct10)}")

subsets = sum(1 for r in runs if set(r["cap5"]) <= set(r["cap10"]))
print(f"runs where cap5 is a subset of cap10 (within the same run): {subsets}/5")

print("\nwith PYTHONHASHSEED fixed to 0:")
fixed = []
for i in range(3):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, tmp.name], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    fixed.append(json.loads(line[-1]))
print(f"   distinct cap-5 pools over 3 runs: {len({tuple(sorted(r['cap5'])) for r in fixed})}")

tmp.unlink(missing_ok=True)
