# -*- coding: utf-8 -*-
"""Paths and data loading shared by the figure and table scripts (no computation beyond the composite score)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
SOURCE = RES / "source_data"
OUT = ROOT / "figures" / "output"
STRAINS = ["NF", "NV", "NM", "NU"]
GUILD = {"NF": "AOA", "NV": "AOA", "NM": "AOB", "NU": "AOB"}
LABEL_Q = 0.75          # high inhibition: AOI% at or above the 75th percentile


def wheat():
    """Log-scale autoscaled matrix of all 44 accessions (as code/prepare_input.py) and the AOI% of the four strains."""
    lab = pd.read_excel(DATA / "preprocessing" / "filtered_metabolites.xlsx", sheet_name="Sheet1", decimal=",")
    tcols = [c for c in lab.columns if "Inhibition" in str(c)]
    mets = [c for c in lab.columns if c not in tcols]
    raw = pd.read_excel(DATA / "preprocessing" / "filtered_notfound_removed.xlsx", sheet_name=0, decimal=",")
    X = raw[mets].apply(pd.to_numeric, errors="coerce")
    X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
    Z = np.log1p(X)
    Z = (Z - Z.mean()) / Z.std()
    Z.columns = [str(c).strip() for c in mets]
    aoi = lab[[[c for c in tcols if c.endswith(s)][0] for s in STRAINS]].apply(pd.to_numeric, errors="coerce")
    aoi.columns = STRAINS
    return Z, aoi


def labels(aoi, target):
    hi = aoi >= aoi.quantile(LABEL_Q)
    return (hi.any(axis=1) if target == "overall" else hi[target]).astype(int).values


def auc(score, y):
    r = rankdata(score)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def names():
    nm = pd.read_excel(DATA / "metabolite_names.xlsx", sheet_name=0, dtype=str).set_index("ID")
    return ({str(k).strip(): str(v).strip() for k, v in nm["Compound name"].items()},
            {str(k).strip(): str(v).strip() for k, v in nm["CHEMICAL CLASS"].items()})


def pi_single():
    p = pd.read_excel(RES / "stability_selection.xlsx", sheet_name="pi_single")
    p["key"] = p["key"].astype(str).str.strip()
    return p.set_index("key")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=300, facecolor="white")
    print("Saved", OUT / f"{stem}.pdf", "and .png")


def write_table(stem, title, header, rows, notes=""):
    """Markdown (readable on GitHub) and CSV copies of a table."""
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if notes:
        md += ["", notes]
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(rows, columns=header).to_csv(OUT / f"{stem}.csv", index=False, encoding="utf-8-sig")
    print("\n".join(md[:len(rows) + 4]))
    print("Saved", OUT / f"{stem}.md", "and .csv")
