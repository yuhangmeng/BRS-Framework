# -*- coding: utf-8 -*-
"""Redraw Figs 2-4 and Tables 1-3 of the manuscript into figures/output (about one minute)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["fig2_combinations.py", "fig3_stability_map.py", "fig4_mixtures.py", "table1_benchmark.py",
           "table2_reproducible_metabolites.py", "table3_mixtures.py"]

for s in SCRIPTS:
    print(f"\n=== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, check=True)
