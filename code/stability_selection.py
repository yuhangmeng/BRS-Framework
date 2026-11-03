# -*- coding: utf-8 -*-
"""
Step 2: stability selection of single metabolites.

--run      BRS (brs.py) on all accessions and on 50 complementary half-samples of 22 accessions (25 random partitions,
           seed 2026); finished runs are skipped, so the step can be interrupted and restarted
--summary  selection probability pi(metabolite, strain) = fraction of half-samples in which the metabolite is in the
           strain's BRS top ten; error bound E[V] <= q^2 / ((2 pi_thr - 1) p) for pi_thr > 0.5; single-accession
           leverage = max over accessions of [pi over half-samples with the accession - pi over those without it]
           (>= 0.7 flags a selection that rests on one accession)

    python code/stability_selection.py --run --summary --workers 6     (about 15 min)
Outputs: results/stability_runs/ (one BRS run per half-sample, subsamples.csv), results/brs_full_data/,
         results/stability_selection.xlsx (pi_single, error_bounds, leverage)
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "preprocessing" / "filtered_metabolites_log.xlsx"
ACCESSIONS = ROOT / "data" / "preprocessing" / "accessions.csv"
NAME_MAP = ROOT / "data" / "metabolite_names.xlsx"
RUNS = ROOT / "results" / "stability_runs"
FULL_RUN = ROOT / "results" / "brs_full_data"
OUT_XLSX = ROOT / "results" / "stability_selection.xlsx"
STRAINS = ["NF", "NV", "NM", "NU"]
N_PAIRS, SEED, TOP_K = 25, 2026, 10
PI_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
PI_SIG = 0.4
THREADS_PER_WORKER = 2
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside", "4-METHYLCATECHOL": "4-Methylcatechol"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_halves(n, n_pairs=N_PAIRS, seed=SEED):
    """n_pairs random splits into two disjoint halves -> 2 * n_pairs index sets."""
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(n_pairs):
        perm = rng.permutation(n)
        k = n // 2
        halves += [np.sort(perm[:k]), np.sort(perm[k:2 * k])]
    return halves


def run_brs(out: Path, idx=None) -> str:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        return f"{out.name} cached"
    env = dict(os.environ, BRS_INPUT=str(INPUT), BRS_OUT_DIR=str(out), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
               LOKY_MAX_CPU_COUNT="1", OMP_NUM_THREADS=str(THREADS_PER_WORKER), MKL_NUM_THREADS=str(THREADS_PER_WORKER))
    env.pop("BRS_SAMPLE_FILE", None)
    if idx is not None:
        np.savetxt(out / "samples.txt", idx, fmt="%d")
        env["BRS_SAMPLE_FILE"] = str(out / "samples.txt")
    with open(out / "run.log", "w", encoding="utf-8") as fh:
        r = subprocess.run([sys.executable, "-u", str(ROOT / "code" / "brs.py")], env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    return f"{out.name} {'FAILED' if r.returncode else 'done'}"


def run(workers):
    if not INPUT.exists():
        raise SystemExit(f"{INPUT} not found: run code/prepare_input.py first")
    n = len(pd.read_excel(INPUT, sheet_name=0).dropna(how="all"))
    halves = make_halves(n)
    RUNS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subsample": [f"sub_{b:03d}" for b in range(len(halves))],
                  "pair": [b // 2 for b in range(len(halves))],
                  "samples": [" ".join(map(str, h)) for h in halves]}).to_csv(RUNS / "subsamples.csv", index=False)
    log(f"{n} accessions, {len(halves)} half-samples of {len(halves[0])} and one run on all accessions")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_brs, FULL_RUN)] + [ex.submit(run_brs, RUNS / f"sub_{b:03d}", h) for b, h in enumerate(halves)]
        for f in as_completed(futs):
            log(f.result())


def summary():
    subs = sorted(d for d in RUNS.glob("sub_*") if (d / "DONE").exists())
    B = len(subs)
    if B == 0:
        raise SystemExit(f"No finished half-samples in {RUNS}: run with --run first")
    acc_names = pd.read_csv(ACCESSIONS, index_col=0)["accession"].values
    n_acc = len(acc_names)
    member = np.zeros((B, n_acc), dtype=bool)             # half-sample x accession
    sel = {}                                              # (metabolite, strain) -> bool vector over half-samples
    n_scored = []
    for b, d in enumerate(subs):
        member[b, np.loadtxt(d / "samples.txt", dtype=int)] = True
        xls = pd.ExcelFile(d / "univariate_BRS.xlsx")
        for s in STRAINS:
            t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All")
            t["Feature"] = t["Feature"].astype(str).str.strip()
            t = t.sort_values(["BRS", "Feature"], ascending=[False, True])
            n_scored.append(len(t))
            for m in t["Feature"].head(TOP_K):
                sel.setdefault((m, s), np.zeros(B, dtype=bool))[b] = True
    log(f"{B} finished half-samples; median metabolites scored per strain {int(np.median(n_scored))}")

    keys = sorted({m for m, _ in sel})
    pi = pd.DataFrame({s: [sel.get((m, s), np.zeros(B, bool)).mean() for m in keys] for s in STRAINS}, index=keys)
    pi.columns = [f"pi_{s}" for s in STRAINS]
    pi["pi_max"] = pi.max(axis=1)

    lev_rows = []
    for (m, s), v in sel.items():
        if v.mean() < 0.2:
            continue
        pin = np.array([v[member[:, i]].mean() for i in range(n_acc)])
        pout = np.array([v[~member[:, i]].mean() for i in range(n_acc)])
        i = int(np.argmax(pin - pout))
        lev_rows.append({"key": m, "strain": s, "pi": v.mean(), "leverage": pin[i] - pout[i],
                         "top_accession_row": i, "top_accession": acc_names[i],
                         "pi_with_top": pin[i], "pi_without_top": pout[i]})
    LEV = pd.DataFrame(lev_rows).sort_values(["strain", "pi"], ascending=[True, False])

    nm = pd.read_excel(NAME_MAP, sheet_name=0, dtype=str).set_index("ID")
    names = {k.strip(): DISPLAY_FIX.get(str(v).strip(), str(v).strip()) for k, v in nm["Compound name"].items()}
    classes = {k.strip(): str(v).strip() for k, v in nm["CHEMICAL CLASS"].items()}
    dup = pd.Series(names).value_counts()

    def display(k):
        b = names.get(k, k)
        return f"{b} [{k.split('_')[1]}]" if dup.get(b, 0) > 1 and "_" in k else b

    pi["set"] = [display(k) for k in pi.index]
    pi["chemical_class"] = [classes.get(k, "") for k in pi.index]
    xls = pd.ExcelFile(FULL_RUN / "univariate_BRS.xlsx")
    for s in STRAINS:
        t = pd.read_excel(xls, f"Inhibition (pct)-{s}_Uni_All").sort_values("BRS", ascending=False).reset_index(drop=True)
        t["Feature"] = t["Feature"].astype(str).str.strip()
        pi[f"BRS_full_{s}"] = pi.index.map(dict(zip(t["Feature"], t["BRS"])))
        pi[f"rank_full_{s}"] = pi.index.map(dict(zip(t["Feature"], t.index + 1)))
    for s in STRAINS:
        d = LEV[LEV.strain == s].set_index("key")
        pi[f"leverage_{s}"] = pi.index.map(d["leverage"])
        pi[f"pi_without_top_{s}"] = pi.index.map(d["pi_without_top"])
    pi = pi.sort_values("pi_max", ascending=False)
    pi.insert(0, "key", pi.index)

    p = int(np.median(n_scored))
    EB = pd.DataFrame([{"pi_thr": t, "n_selected_any_strain": int((pi["pi_max"] >= t).sum()), "p_candidates": p,
                        "q_per_halfsample": TOP_K, "E[V]_bound_per_strain": TOP_K ** 2 / ((2 * t - 1) * p)}
                       for t in PI_THRESHOLDS])
    with pd.ExcelWriter(OUT_XLSX) as w:
        pd.DataFrame([("half-samples", f"{B} complementary half-samples of all accessions (seed {SEED})"),
                      ("pi", "fraction of half-samples in which the metabolite is in the strain's BRS top-10"),
                      ("leverage", "max over accessions of pi(with accession) - pi(without); >= 0.7 flags dependence "
                                   "on a single accession"),
                      ("reproducible", f"metabolites with pi >= {PI_SIG} for at least one strain")],
                     columns=["field", "description"]).to_excel(w, sheet_name="README", index=False)
        pi.round(4).to_excel(w, sheet_name="pi_single", index=False)
        EB.round(3).to_excel(w, sheet_name="error_bounds", index=False)
        LEV.round(3).to_excel(w, sheet_name="leverage", index=False)
    pd.set_option("display.width", 250)
    show = ["set", "pi_NF", "pi_NV", "pi_NM", "pi_NU", "pi_max"] + [f"leverage_{s}" for s in STRAINS]
    print(pi[pi.pi_max >= PI_SIG][show].round(2).to_string(index=False))
    log(f"{(pi.pi_max >= PI_SIG).sum()} reproducible metabolites (pi >= {PI_SIG}) -> {OUT_XLSX}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.run:
        run(a.workers)
    if a.summary:
        summary()
