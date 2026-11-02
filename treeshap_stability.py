# -*- coding: utf-8 -*-
"""
Is the non-reproducibility of combinations specific to BRS? The stability analysis of Fig. 2 repeated with TreeSHAP
interaction values (random forest) as the scorer, on the same 50 complementary half-samples of the wheat panel.

Per strain and half-sample, a random forest (500 trees, as in the synthetic benchmark) is fitted to AOI% on all
413 metabolites, and TreeSHAP interaction values phi_ij are computed:
    single  mean |SHAP_i|                         (all 413 metabolites)
    pair    mean |phi_ij|, i < j                  (all 85,078 pairs: no candidate pool)
    triad   sum of the three pairwise mean |phi|  (triads of the run's own top-20 metabolites, as in the benchmark)
Selection probability pi = fraction of half-samples in which a set is in the top 10. Bagged consensus: mean score in
two disjoint groups of 25 half-samples (even / odd complementary pairs), Spearman over the union of both groups'
top-100 sets (singles, pairs) or over triads enumerated >= 5 times in both groups, and top-10 overlap.
Tree models are invariant to monotone transformations, so the results hold for raw and log-scale intensities alike.
Comparison: BRS on the same half-samples (results_stability_v2raw for all orders, results_stability_uni for singles).

Output: results_treeshap_stability/treeshap_stability.xlsx
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Data Preprocessing" / "filtered_metabolites_log.xlsx"
SUBS = ROOT / "results_stability_uni" / "subsamples.csv"
BRS_ALL = ROOT / "results_stability_v2raw"
BRS_UNI = ROOT / "results_stability_uni" / "stability_summary_uni.xlsx"
NAME_MAP = ROOT / "name.xlsx"
OUT_DIR = ROOT / "results_treeshap_stability"
STRAINS = ["NF", "NV", "NM", "NU"]
TOP_K, TOP_POOL, TOP_CONS, MIN_ENUM = 10, 20, 100, 5
PI_GATE, LEV_FLAG = 0.4, 0.7
RF_PARAMS = dict(n_estimators=500, random_state=42, n_jobs=-1)

# ==========================================
# 1) Data
# ==========================================
inp = pd.read_excel(INPUT, sheet_name=0, decimal=",")
tcol = {s: [c for c in inp.columns if "Inhibition" in str(c) and str(c).endswith(s)][0] for s in STRAINS}
mets = [c for c in inp.columns if c not in tcol.values()]
keys = [str(c).strip() for c in mets]
X = inp[mets].apply(pd.to_numeric, errors="coerce").values
n, p = X.shape
subs = pd.read_csv(SUBS)
halves = [np.array(list(map(int, v.split()))) for v in subs["samples"]]
group = (subs["pair"].values % 2)
nm = pd.read_excel(NAME_MAP, sheet_name=0, dtype=str).set_index("ID")["Compound name"]
names = {k.strip(): v.strip() for k, v in nm.items()}
dup = pd.Series(names).value_counts()
disp = lambda k: (f"{names.get(k, k)} [{k.split('_')[1]}]" if dup.get(names.get(k, k), 0) > 1 and "_" in k
                  else names.get(k, k))
iu = np.triu_indices(p, 1)


def treeshap(rows, y):
    m = RandomForestRegressor(**RF_PARAMS).fit(X[rows], y[rows])
    phi = np.asarray(shap.TreeExplainer(m).shap_interaction_values(X[rows]), dtype=float)
    uni = np.abs(phi.sum(2)).mean(0)
    pm = np.abs(phi).mean(0)
    pool = np.argsort(uni)[::-1][:TOP_POOL]
    trip = {tuple(sorted(c)): pm[c[0], c[1]] + pm[c[0], c[2]] + pm[c[1], c[2]] for c in combinations(pool, 3)}
    return uni, pm[iu].astype(np.float32), trip


def top(scores, k=TOP_K):
    return list(np.argsort(scores)[::-1][:k])


# ==========================================
# 2) Runs
# ==========================================
summary, cons_rows, gate_rows, lev_rows = [], [], [], []
pi_tabs = {o: [] for o in ("single", "pair", "triad")}
for s in STRAINS:
    y = pd.to_numeric(inp[tcol[s]], errors="coerce").values
    print(f"--- {s} ---", flush=True)
    f_uni, f_pair, f_trip = treeshap(np.arange(n), y)
    full_top = {"single": set(top(f_uni)), "pair": set(top(f_pair)),
                "triad": set(sorted(f_trip, key=f_trip.get, reverse=True)[:TOP_K])}
    U, P, T, sel = [], [], [], {"single": [], "pair": [], "triad": []}
    for h, rows in enumerate(halves):
        u, pr, tr = treeshap(rows, y)
        U.append(u)
        P.append(pr)
        T.append(tr)
        sel["single"].append(set(top(u)))
        sel["pair"].append(set(top(pr)))
        sel["triad"].append(set(sorted(tr, key=tr.get, reverse=True)[:TOP_K]))
    U, P = np.array(U), np.array(P)
    H = len(halves)
    for order in ("single", "pair", "triad"):
        cnt = {}
        for st in sel[order]:
            for k in st:
                cnt[k] = cnt.get(k, 0) + 1
        pi = {k: v / H for k, v in cnt.items()}
        rec = {"scorer": "TreeSHAP (RF)", "order": order, "strain": s, "max_pi": max(pi.values()),
               "n_sets_pi_ge_0.4": sum(v >= PI_GATE for v in pi.values()),
               "fulltop10_mean_pi": np.mean([pi.get(k, 0) for k in full_top[order]])}
        if order == "triad":
            enum = [np.mean([k in T[h] for h in range(H)]) for k in full_top[order]]
            rec["fulltop10_mean_enum_rate"] = np.mean(enum)
        else:
            rec["fulltop10_mean_enum_rate"] = 1.0          # every single and pair is scored in every half-sample
        summary.append(rec)

        def label(k):
            if order == "single":
                return disp(keys[k])
            if order == "pair":
                return " + ".join(disp(keys[i]) for i in (iu[0][k], iu[1][k]))
            return " + ".join(disp(keys[i]) for i in k)
        for k, v in sorted(pi.items(), key=lambda kv: -kv[1])[:50]:
            pi_tabs[order].append({"strain": s, "set": label(k), "pi": v, "in_full_top10": k in full_top[order]})

        # bagged consensus in two disjoint groups of half-samples
        cons = []
        for g in (0, 1):
            hs = np.where(group == g)[0]
            if order == "single":
                c = pd.Series(U[hs].mean(0))
            elif order == "pair":
                c = pd.Series(P[hs].mean(0))
            else:
                agg = {}
                for h in hs:
                    for k, v in T[h].items():
                        agg.setdefault(k, []).append(v)
                c = pd.Series({k: np.mean(v) for k, v in agg.items() if len(v) >= MIN_ENUM}, dtype=float)
            cons.append(c.sort_values(ascending=False))
        if order == "triad":
            common = cons[0].index.intersection(cons[1].index)
        else:
            common = cons[0].index[:TOP_CONS].union(cons[1].index[:TOP_CONS])
        rho = spearmanr(cons[0][common], cons[1][common]).correlation if len(common) > 3 else np.nan
        cons_rows.append({"scorer": "TreeSHAP (RF)", "order": order, "strain": s, "sets_compared": len(common),
                          "spearman_consensus_A_vs_B": rho,
                          "top10_overlap_A_vs_B": len(set(cons[0].index[:TOP_K]) & set(cons[1].index[:TOP_K]))})

    # single-metabolite gate and single-accession leverage
    cnt = np.zeros(p)
    for st in sel["single"]:
        for k in st:
            cnt[k] += 1
    pis = cnt / H
    for k in np.where(pis >= 0.3)[0]:
        inn = np.array([[i in halves[h] for h in range(H)] for i in range(n)])          # (n, H)
        chosen = np.array([k in sel["single"][h] for h in range(H)])
        lev = max(chosen[inn[i]].mean() - chosen[~inn[i]].mean() for i in range(n))
        lev_rows.append({"strain": s, "metabolite": disp(keys[k]), "key": keys[k], "pi_treeshap": pis[k],
                         "leverage": lev})
    for k in np.where(pis >= PI_GATE)[0]:
        gate_rows.append({"strain": s, "metabolite": disp(keys[k]), "key": keys[k], "pi_treeshap": pis[k]})

# ==========================================
# 3) BRS on the same half-samples, for comparison
# ==========================================
pe = pd.read_excel(BRS_ALL / "stability_diagnostics.xlsx", sheet_name="pool_effect")
bc_ = pd.read_excel(BRS_ALL / "stability_diagnostics.xlsx", sheet_name="bagged_consensus")
brs_rows = [{"scorer": "BRS (raw input)", "order": r.order, "strain": r.strain, "max_pi": r.max_pi,
             "fulltop10_mean_pi": r.fulltop10_mean_pi, "fulltop10_mean_enum_rate": r.fulltop10_mean_enum_rate}
            for r in pe.itertuples()]
for o, sh in (("single", "pi_single"), ("pair", "pi_pair"), ("triad", "pi_triad")):
    t = pd.read_excel(BRS_ALL / "stability_summary.xlsx", sheet_name=sh)
    for r in brs_rows:
        if r["order"] == o:
            r["n_sets_pi_ge_0.4"] = int((t[f"pi_{r['strain']}"] >= PI_GATE).sum())
uni_log = pd.read_excel(BRS_UNI, sheet_name="pi_single")
for s in STRAINS:
    brs_rows.append({"scorer": "BRS (log input)", "order": "single", "strain": s, "max_pi": uni_log[f"pi_{s}"].max(),
                     "n_sets_pi_ge_0.4": int((uni_log[f"pi_{s}"] >= PI_GATE).sum())})
S = pd.DataFrame(summary + brs_rows)
Cn = pd.concat([pd.DataFrame(cons_rows),
                bc_.rename(columns={"spearman_consensus_groupA_vs_B": "spearman_consensus_A_vs_B",
                                    "top10_overlap_consensus_A_vs_B": "top10_overlap_A_vs_B",
                                    "sets_in_both_groups": "sets_compared"})
                .drop(columns=["top10_overlap_consensus_vs_fulldata"]).assign(scorer="BRS (raw input)")],
               ignore_index=True)

G = pd.DataFrame(gate_rows)
brs_gate = {s: set(uni_log.loc[uni_log[f"pi_{s}"] >= PI_GATE, "key"].astype(str).str.strip()) for s in STRAINS}
ov = [{"strain": s, "TreeSHAP_gate": int((G.strain == s).sum()) if len(G) else 0, "BRS_gate": len(brs_gate[s]),
       "shared": len(set(G.loc[G.strain == s, "key"]) & brs_gate[s]) if len(G) else 0,
       "TreeSHAP_members": "; ".join(G.loc[G.strain == s, "metabolite"]) if len(G) else ""} for s in STRAINS]

OUT_DIR.mkdir(exist_ok=True)
readme = pd.DataFrame([(l.strip(),) for l in __doc__.strip().splitlines() if l.strip()], columns=["notes"])
with pd.ExcelWriter(OUT_DIR / "treeshap_stability.xlsx") as w:
    readme.to_excel(w, sheet_name="README", index=False)
    S.round(3).to_excel(w, sheet_name="summary", index=False)
    Cn.round(3).to_excel(w, sheet_name="bagged_consensus", index=False)
    for o, rows in pi_tabs.items():
        pd.DataFrame(rows).round(3).to_excel(w, sheet_name=f"pi_{o}", index=False)
    pd.DataFrame(ov).to_excel(w, sheet_name="gate_overlap", index=False)
    pd.DataFrame(lev_rows).round(3).to_excel(w, sheet_name="leverage", index=False)
print("Saved", OUT_DIR / "treeshap_stability.xlsx")
pd.set_option("display.width", 220)
print(S.sort_values(["order", "scorer", "strain"]).round(2).to_string(index=False))
print(Cn.sort_values(["order", "scorer", "strain"]).round(2).to_string(index=False))
print(pd.DataFrame(ov).to_string(index=False))
print(pd.DataFrame(lev_rows).round(2).to_string(index=False))
