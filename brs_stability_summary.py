# -*- coding: utf-8 -*-
"""
Stability selection on top of BRS (reads results_stability/sub_*/ written by brs_stability_subsampling.py).

1) Selection probability  pi(set, strain) = fraction of half-samples in which the set is in that
   strain's BRS top-K (K = 10), for single metabolites, pairs and triads.
2) Error control: for a stable list {sets with pi >= pi_thr}, the expected number of false selections
   is bounded by E[V] <= q^2 / ((2 pi_thr - 1) p) (Meinshausen & Buehlmann 2010; holds for
   complementary-pairs subsampling, Shah & Samworth 2013), with q = K selections per half-sample
   and p = number of candidate sets enumerated per half-sample (conservative choice).
3) Multi-strain classification of each stable set (strain-specific, AOA-shared, AOB-shared,
   cross-guild), with a per-strain permutation test of the set's composite score against AOI%
   (Spearman, 10,000 label permutations, Benjamini-Hochberg over all tests).
4) Metabolite modules: co-selection graph (edge = max pi of any pair/triad containing both
   metabolites), thresholded and partitioned into communities; each module is summarized by its
   composite score (AUC for high- vs low-BNI with bootstrap CI and permutation p, Spearman with each
   strain) and by how often its internal edges recur across half-samples.

    python brs_stability_summary.py
"""
import os
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ==========================================
# 1) Configuration
# ==========================================
STAB_DIR = Path("results_stability")
FULL_RUN = Path(os.environ.get("BRS_OUT_DIR", "results_v2_deterministic"))   # full-data BRS run
RAW_PATH = Path("Data Preprocessing/filtered_notfound_removed.xlsx")
LABEL_PATH = Path("Data Preprocessing/filtered_metabolites.xlsx")
NAME_MAP_PATH = Path("name.xlsx")
OUT_XLSX = STAB_DIR / "stability_summary.xlsx"

STRAINS = ["NF", "NV", "NM", "NU"]
GUILD = {"NF": "AOA", "NV": "AOA", "NM": "AOB", "NU": "AOB"}
ORDERS = {"single": ("univariate_BRS.xlsx", "Uni", "Feature"),
          "pair": ("pairwise_BRS.xlsx", "Pairs", "Pair"),
          "triad": ("triple_BRS.xlsx", "Triplets", "Triplet")}
TOP_K = 10
PI_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
PI_MAIN = 0.7                 # threshold used for the reported lists and the multi-strain classification
PI_EDGE = 0.5                 # module graph: keep co-selection edges with pi >= PI_EDGE
MAIN_Q = 0.75
N_PERM, N_BOOT, SEED = 10000, 2000, 42
DISPLAY_FIX = {"L-TRYPTOPHAN": "L-Tryptophan", "Uric Acid": "Uric acid", "Ala-ala": "Ala-Ala",
               "Isoshaftoside": "Isoschaftoside", "4-METHYLCATECHOL": "4-Methylcatechol"}
rng = np.random.default_rng(SEED)


def canon(s) -> str:
    return " | ".join(sorted(t.strip() for t in str(s).split("|")))


def log(msg):
    print(msg, flush=True)


# ==========================================
# 2) Selection probabilities from the half-samples
# ==========================================
def finished(d: Path) -> bool:
    """DONE marker, or a complete triad workbook (BRS creates its workbooks empty at the start of a stage)."""
    if (d / "DONE").exists():
        return True
    f = d / "triple_BRS.xlsx"
    if not f.exists() or f.stat().st_size == 0 or os.path.getmtime(f) > time.time() - 60:
        return False
    try:
        return "heatmap_values_wide" in pd.ExcelFile(f).sheet_names
    except Exception:
        return False


subs = sorted(d for d in STAB_DIR.glob("sub_*") if finished(d))
B = len(subs)
if B == 0:
    raise SystemExit("No finished half-samples in results_stability/")
log(f"--- {B} finished half-samples ---")

counts = {o: {} for o in ORDERS}          # order -> key -> {strain: count}
n_enum = {o: [] for o in ORDERS}          # candidates enumerated per half-sample and strain
per_sub_top = []                          # (sub, order, strain, set of keys)
for d in subs:
    for order, (fname, kind, col) in ORDERS.items():
        xls = pd.ExcelFile(d / fname)
        for s in STRAINS:
            t = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
            t["key"] = t[col].map(canon)
            t = t.sort_values(["BRS", "key"], ascending=[False, True])
            n_enum[order].append(len(t))
            top = list(t["key"].head(TOP_K))
            per_sub_top.append((d.name, order, s, set(top)))
            for k in top:
                counts[order].setdefault(k, dict.fromkeys(STRAINS, 0))[s] += 1

pi = {o: pd.DataFrame(counts[o]).T.reindex(columns=STRAINS).fillna(0) / B for o in ORDERS}
p_cand = {o: int(np.median(n_enum[o])) for o in ORDERS}

# full-data BRS for reference
full = {}
for order, (fname, kind, col) in ORDERS.items():
    xls = pd.ExcelFile(FULL_RUN / fname)
    rows = {}
    for s in STRAINS:
        t = pd.read_excel(xls, f"Inhibition (pct)-{s}_{kind}_All")
        t["key"] = t[col].map(canon)
        t = t.sort_values("BRS", ascending=False).reset_index(drop=True)
        for i, (k, v) in enumerate(zip(t["key"], t["BRS"])):
            rows.setdefault(k, {})[f"BRS_full_{s}"] = v
            rows[k][f"rank_full_{s}"] = i + 1
    full[order] = pd.DataFrame(rows).T

# ==========================================
# 3) Names
# ==========================================
id2name = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")["Compound name"].str.strip()
id2class = pd.read_excel(NAME_MAP_PATH, sheet_name=0, dtype=str).set_index("ID")["CHEMICAL CLASS"].str.strip()
all_ids = sorted({i for o in ORDERS for k in pi[o].index for i in k.split(" | ")})
base = {i: DISPLAY_FIX.get(id2name.get(i, i), id2name.get(i, i)) for i in all_ids}
dup = pd.Series(base).value_counts()


def name(i: str) -> str:
    tag = i.split("_")[1] if "_" in i else i[:8]
    return f"{base[i]} [{tag}]" if dup[base[i]] > 1 else base[i]


def set_name(k: str) -> str:
    return " + ".join(name(i) for i in k.split(" | "))


# ==========================================
# 4) Composite scores for association tests (same transform as Table 3)
# ==========================================
raw = pd.read_excel(RAW_PATH, sheet_name=0, decimal=",")
df_lab = pd.read_excel(LABEL_PATH, sheet_name=0)
bni_cols = {s: [c for c in df_lab.columns if "Inhibition" in str(c) and s in str(c)][0] for s in STRAINS}
aoi = df_lab[[bni_cols[s] for s in STRAINS]].set_axis(STRAINS, axis=1)
met_cols = [c for c in df_lab.columns if c not in bni_cols.values()]
X = raw[met_cols].apply(pd.to_numeric, errors="coerce")
X = X.apply(lambda s: s.where(s != 0, s[s != 0].min() / 2))
Z = np.log1p(X)
Z = (Z - Z.mean()) / Z.std()
Z.columns = [str(c).strip() for c in Z.columns]
y_bin = (aoi >= aoi.quantile(MAIN_Q)).any(axis=1).astype(int).values
aoi_rank = {s: rankdata(aoi[s].values) for s in STRAINS}
perm_idx = np.array([rng.permutation(len(y_bin)) for _ in range(N_PERM)])


def composite(ids):
    return Z[list(ids)].mean(axis=1).values


def spearman_perm(score):
    """Spearman rho with each strain's AOI% and one-sided permutation p (positive association)."""
    rs = rankdata(score)
    rs = (rs - rs.mean()) / rs.std()
    out = {}
    for s in STRAINS:
        ra = (aoi_rank[s] - aoi_rank[s].mean()) / aoi_rank[s].std()
        rho = float((rs * ra).mean())
        null = (rs[perm_idx] * ra).mean(axis=1)
        out[s] = (rho, (1 + (null >= rho).sum()) / (N_PERM + 1))
    return out


def auc_ci_p(score):
    r = rankdata(score)
    n1, n0 = y_bin.sum(), len(y_bin) - y_bin.sum()
    auc = (r[y_bin == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pos, neg = np.where(y_bin == 1)[0], np.where(y_bin == 0)[0]
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        rb, yb = rankdata(score[idx]), y_bin[idx]
        boot.append((rb[yb == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    yp = y_bin[perm_idx]
    null = ((yp * r).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc, np.percentile(boot, 2.5), np.percentile(boot, 97.5), (1 + (null >= auc).sum()) / (N_PERM + 1)


# ==========================================
# 5) Stable lists, error bounds, multi-strain classification
# ==========================================
def classify(strains):
    if not strains:
        return "not stable"
    g = {GUILD[s] for s in strains}
    if g == {"AOA", "AOB"}:
        return "cross-guild (AOA + AOB)"
    if len(strains) > 1:
        return f"{g.pop()}-shared"
    return f"{strains[0]}-specific"


bounds, tables = [], {}
for order in ORDERS:
    p_ = pi[order]
    for s in STRAINS:
        for thr in PI_THRESHOLDS:
            n_stable = int((p_[s] >= thr).sum())
            ev = TOP_K ** 2 / ((2 * thr - 1) * p_cand[order])
            bounds.append({"order": order, "strain": s, "pi_thr": thr, "n_stable": n_stable,
                           "p_candidates": p_cand[order], "q_per_halfsample": TOP_K,
                           "E[V]_bound": round(ev, 2)})
    df = p_.add_prefix("pi_").copy()
    df["pi_max"] = p_.max(axis=1)
    df["stable_strains"] = [[s for s in STRAINS if p_.loc[k, s] >= PI_MAIN] for k in p_.index]
    df["class"] = df["stable_strains"].map(classify)
    df["stable_strains"] = df["stable_strains"].map(lambda v: ", ".join(v))
    df["set"] = [set_name(k) for k in df.index]
    df = df.join(full[order], how="left")
    tables[order] = df.sort_values("pi_max", ascending=False)
bounds = pd.DataFrame(bounds)

# association tests for every set that is stable in at least one strain
assoc_rows = []
for order, df in tables.items():
    for k in df.index[df["pi_max"] >= PI_MAIN]:
        ids = k.split(" | ")
        sp = spearman_perm(composite(ids))
        auc, lo, hi, p_auc = auc_ci_p(composite(ids))
        row = {"order": order, "key": k, "set": df.loc[k, "set"], "class": df.loc[k, "class"],
               "AUC": auc, "AUC_lo": lo, "AUC_hi": hi, "p_AUC": p_auc}
        for s in STRAINS:
            row[f"rho_{s}"], row[f"p_{s}"] = sp[s]
        assoc_rows.append(row)
assoc = pd.DataFrame(assoc_rows)
if len(assoc):
    pcols = [f"p_{s}" for s in STRAINS]
    flat = assoc[pcols].values.ravel()
    o = np.argsort(flat)
    q = np.empty_like(flat)
    q[o] = np.minimum.accumulate((flat[o] * len(flat) / (np.arange(len(flat)) + 1))[::-1])[::-1]
    q = np.minimum(q, 1).reshape(assoc[pcols].shape)
    for j, s in enumerate(STRAINS):
        assoc[f"q_{s}"] = q[:, j]
    assoc["assoc_strains_FDR10"] = [", ".join(s for s in STRAINS if r[f"q_{s}"] < 0.10) for _, r in assoc.iterrows()]

# ==========================================
# 6) Metabolite modules from the co-selection graph
# ==========================================
edge_w = {}
for order in ("pair", "triad"):
    for k, row in pi[order].iterrows():
        w = row.max()
        for a, b in combinations(k.split(" | "), 2):
            e = tuple(sorted((a, b)))
            edge_w[e] = max(edge_w.get(e, 0), w)
G = nx.Graph()
G.add_weighted_edges_from((a, b, w) for (a, b), w in edge_w.items() if w >= PI_EDGE)
modules = []
if G.number_of_edges():
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    for m in sorted(comms, key=len, reverse=True):
        if len(m) < 3:
            continue
        ids = sorted(m)
        sub_e = [e for e in G.subgraph(ids).edges()]
        # recurrence: per half-sample, fraction of internal edges co-selected in any top-K pair/triad
        rec = []
        for d in subs:
            keys = set().union(*[t for (sn, o, s, t) in per_sub_top if sn == d.name and o != "single"])
            co = {tuple(sorted(e)) for k in keys for e in combinations(k.split(" | "), 2)}
            rec.append(np.mean([tuple(sorted(e)) in co for e in sub_e]) if sub_e else np.nan)
        score = composite(ids)
        auc, lo, hi, p_auc = auc_ci_p(score)
        sp = spearman_perm(score)
        modules.append({
            "module": f"M{len(modules) + 1}", "size": len(ids),
            "members": "; ".join(name(i) for i in ids),
            "classes": "; ".join(sorted({str(id2class.get(i, 'Unknown')) for i in ids})),
            "internal_edges": len(sub_e),
            "mean_edge_pi": float(np.mean([G.edges[e]["weight"] for e in sub_e])),
            "edge_recurrence_mean": float(np.nanmean(rec)),
            "AUC": auc, "AUC_lo": lo, "AUC_hi": hi, "p_AUC": p_auc,
            **{f"rho_{s}": sp[s][0] for s in STRAINS}, **{f"p_{s}": sp[s][1] for s in STRAINS},
            "member_ids": " | ".join(ids),
        })
modules = pd.DataFrame(modules)

# ==========================================
# 7) Export
# ==========================================
readme = pd.DataFrame([
    ("half-samples", f"{B} (complementary pairs of 22 + 22 accessions); full BRS rerun on each"),
    ("selection probability", f"fraction of half-samples in which the set is in the strain's BRS top-{TOP_K}"),
    ("error bound", "E[V] <= q^2 / ((2*pi_thr - 1) * p); q = top-K per half-sample, p = candidates enumerated per half-sample"),
    ("classification", f"strains with pi >= {PI_MAIN}; cross-guild = at least one AOA and one AOB strain"),
    ("association", f"Spearman of the composite score with each strain's AOI%, one-sided label permutation ({N_PERM}), BH over all tests"),
    ("modules", f"co-selection graph, edges with pi >= {PI_EDGE}, Louvain communities of >= 3 metabolites"),
    ("caveat", "selection probabilities and association tests use the same 44 accessions; they describe stability, not external validity"),
], columns=["field", "description"])
with pd.ExcelWriter(OUT_XLSX) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    bounds.to_excel(w, sheet_name="error_bounds", index=False)
    for order, df in tables.items():
        df.reset_index(names="key").to_excel(w, sheet_name=f"pi_{order}", index=False)
    assoc.to_excel(w, sheet_name="stable_sets_association", index=False)
    modules.to_excel(w, sheet_name="modules", index=False)
log(f"Saved: {OUT_XLSX}")
for order in ORDERS:
    n = {s: int((pi[order][s] >= PI_MAIN).sum()) for s in STRAINS}
    log(f"{order}: stable (pi >= {PI_MAIN}) per strain {n}; candidates per half-sample {p_cand[order]}")
if len(modules):
    log(modules[["module", "size", "mean_edge_pi", "edge_recurrence_mean", "AUC", "members"]].to_string(index=False))
