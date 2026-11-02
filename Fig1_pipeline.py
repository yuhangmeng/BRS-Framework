# -*- coding: utf-8 -*-
"""
Fig. 1: overview of the pipeline, drawn as a left-to-right schematic with small illustrative plots
(replaces the Visio box diagram of make_fig1_visio.py).

    Input -> 1 Ranking (BRS) -> 2 Stability selection -> 3 Mixture assembly -> 4 Combination test
    outputs A (reproducible metabolites), B (mixtures) and C (interaction hypotheses) below steps 2-4;
    direct enumeration of combinations (not used) below step 1.

The mini-plots are schematic (illustrative values), not data. Colours: Nature Publishing Group palette
(navy for the pipeline, red for its outputs) on neutral greys; font Arial.
Every primitive is drawn through Canvas, which also records it, so Fig1_pipeline_visio.py can rebuild the same
figure as editable Visio shapes.
Output: 论文/overleaf_v6/figures/Fig1_pipeline.(pdf|png)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "论文" / "overleaf_v6" / "figures" / "Fig1_pipeline"

# Nature Publishing Group palette (navy, red) with neutral greys
NAVY, RED, NEUTRAL = "#3C5488", "#E64B35", "#BDBDBD"
INK, INK_2, AXIS = "#1A1A1A", "#595959", "#BFBFBF"
PANEL, NAVY_L, RED_L = "#F2F2F2", "#E8EBF2", "#FBE4E0"
FS = 7.0
mpl.rcParams.update({"font.family": "Arial", "font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42})

W, H = 7.0, 3.05
Y_TOP, Y_BOT = 2.98, 0.80                 # step panels
ILL_TOP, ILL_BOT = 2.62, 1.30             # illustration area inside a panel
OUT_TOP, OUT_BOT = 0.54, 0.06             # output row
X_IN, W_IN = 0.05, 0.85                   # input panel
GAP = 0.15
W_STEP = (W - 0.10 - W_IN - 4 * GAP) / 4
X_STEP = [X_IN + W_IN + GAP + i * (W_STEP + GAP) for i in range(4)]
DASH, DASH_SHORT = (0, (3, 2)), (0, (2.5, 1.5))


def mix(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return to_hex(a + (b - a) * t)


class Canvas:
    """Draws on a matplotlib axes in inch coordinates and records every primitive (for the Visio export)."""

    def __init__(self, ax):
        self.ax, self.ops = ax, []

    def _rec(self, **op):
        self.ops.append(op)

    def rbox(self, x, y, w, h, fc, ec="none", lw=0.6, r=0.06, ls="-", z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, ls=ls, zorder=z))
        self._rec(kind="rbox", x=x, y=y, w=w, h=h, fc=fc, ec=ec, lw=lw, r=r, ls=ls, z=z)

    def rect(self, x, y, w, h, fc, ec="none", lw=0.0, z=2):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z))
        self._rec(kind="rbox", x=x, y=y, w=w, h=h, fc=fc, ec=ec, lw=lw, r=0.0, ls="-", z=z)

    def circle(self, cx, cy, r, fc, ec="none", lw=0.0, z=3):
        self.ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=z))
        self._rec(kind="circle", cx=cx, cy=cy, r=r, fc=fc, ec=ec, lw=lw, z=z)

    def line(self, xs, ys, color, lw, ls="-", z=2):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
        self._rec(kind="line", xs=list(map(float, xs)), ys=list(map(float, ys)), color=color, lw=lw, ls=ls, z=z)

    def polygon(self, xs, ys, fc, z=2):
        self.ax.fill(xs, ys, color=fc, lw=0, zorder=z)
        self._rec(kind="polygon", xs=list(map(float, xs)), ys=list(map(float, ys)), fc=fc, z=z)

    def arrow(self, x0, y0, x1, y1, color=INK_2, lw=0.9, ls="-", ms=7, z=4):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color,
                                          lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=z))
        self._rec(kind="arrow", x0=x0, y0=y0, x1=x1, y1=y1, color=color, lw=lw, ls=ls, ms=ms, z=z)

    def text(self, x, y, s, size, color=INK, weight="normal", style="normal", ha="left", va="center", rotation=0,
             linespacing=1.2, z=5):
        a = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, style=style, ha=ha, va=va,
                         rotation=rotation, linespacing=linespacing, zorder=z)
        self._rec(kind="text", artist=a, s=s, color=color, weight=weight, style=style, ha=ha, rotation=rotation,
                  linespacing=linespacing, z=z)
        return a


def build():
    """Draw the figure; return (fig, ax, ops) with the text groups already fitted to their boxes."""
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = Canvas(ax)
    rng = np.random.default_rng(3)
    fit = {"title": [], "desc": [], "out_title": [], "out_sub": []}   # text groups that must fit their boxes

    def panel(x, w, num, title, desc):
        c.rbox(x, Y_BOT, w, Y_TOP - Y_BOT, PANEL)
        if num is None:
            t = c.text(x + 0.09, Y_TOP - 0.17, title, FS + 0.5, weight="bold")
        else:
            c.circle(x + 0.17, Y_TOP - 0.17, 0.085, NAVY)
            c.text(x + 0.17, Y_TOP - 0.172, str(num), FS, color="white", weight="bold", ha="center")
            t = c.text(x + 0.31, Y_TOP - 0.17, title, FS + 0.5, weight="bold")
        fit["title"].append((t, x + w - 0.06))
        d = c.text(x + 0.09, Y_BOT + 0.09, desc, FS - 0.6, color=INK_2, va="bottom", linespacing=1.3)
        fit["desc"].append((d, x + w - 0.06))

    def output_box(x, w, letter, title, sub, fc, ec, ls, title_color):
        c.rbox(x, OUT_BOT, w, OUT_TOP - OUT_BOT, fc, ec, lw=0.7, ls=ls)
        x_t = x + 0.09
        if letter:
            c.text(x + 0.08, OUT_TOP - 0.15, letter, FS + 0.5, color=RED, weight="bold")
            x_t = x + 0.23
        t = c.text(x_t, OUT_TOP - 0.15, title, FS, color=title_color, weight="bold")
        s = c.text(x_t, OUT_TOP - 0.33, sub, FS - 1.0, color=INK_2, linespacing=1.25)
        fit["out_title"].append((t, x + w - 0.05))
        fit["out_sub"].append((s, x + w - 0.05))

    # ---------------- Input ----------------
    panel(X_IN, W_IN, None, "Input", "intensities;\none phenotype\nper strain")
    cs, nr, nc = 0.075, 9, 6
    gx, gy = X_IN + 0.17, ILL_BOT + 0.10
    vals = rng.random((nr, nc))
    for i in range(nr):
        for j in range(nc):
            c.rect(gx + j * cs, gy + i * cs, cs, cs, mix(NAVY_L, NAVY, vals[i, j] * 0.85), "white", 0.4)
    px = gx + nc * cs + 0.08
    yv = np.sort(rng.random(nr))
    for i in range(nr):
        c.rect(px, gy + i * cs, cs, cs, mix(RED_L, RED, yv[i] * 0.9), "white", 0.4)
    c.text(gx + nc * cs / 2, gy + nr * cs + 0.06, "metabolites", FS - 0.8, color=INK_2, ha="center", va="bottom")
    c.text(px + cs / 2, gy + nr * cs + 0.06, "y", FS - 0.8, color=INK_2, style="italic", ha="center", va="bottom")
    c.text(gx - 0.06, gy + nr * cs / 2, "samples", FS - 0.8, color=INK_2, ha="right", rotation=90)

    # ---------------- Step 1: ranking ----------------
    x = X_STEP[0]
    panel(x, W_STEP, 1, "Ranking (BRS)", "three learners and the\ncorrelation, per strain")
    cy0, ch, cw = ILL_TOP - 0.22, 0.17, 0.56
    for k, lab in enumerate(["Elastic Net", "LightGBM", "MLP", "r²"]):
        yy = cy0 - k * (ch + 0.08)
        c.rbox(x + 0.10, yy, cw, ch, "white", AXIS, lw=0.5, r=0.04, z=2)
        c.text(x + 0.10 + cw / 2, yy + ch / 2, lab, FS - 0.8, ha="center")
    y_mid = cy0 - 1.5 * (ch + 0.08) + ch / 2
    c.arrow(x + 0.10 + cw + 0.03, y_mid, x + 0.10 + cw + 0.17, y_mid)
    bx, bw_max, bh = x + 0.10 + cw + 0.21, W_STEP - cw - 0.38, 0.075
    for k, L in enumerate([1.0, 0.86, 0.74, 0.62, 0.52, 0.42, 0.33, 0.25]):
        c.rect(bx, cy0 + ch - 0.02 - k * (bh + 0.045) - bh, bw_max * L, bh, mix(NEUTRAL, INK_2, 0.35))
    c.text(bx, cy0 + ch + 0.04, "ranked", FS - 0.8, color=INK_2, va="bottom")

    # ---------------- Step 2: stability selection ----------------
    x = X_STEP[1]
    panel(x, W_STEP, 2, "Stability selection", "π over 50 half-samples;\nkeep π ≥ 0.4 and\ncheck leverage")
    r_dot, top_y = 0.032, ILL_TOP - 0.06
    for k in range(10):
        c.circle(x + 0.22 + k * 0.105, top_y, r_dot, INK_2)
    half_y = top_y - 0.27
    for k in range(10):
        side = 0 if k < 5 else 1
        c.circle(x + 0.17 + side * 0.60 + (k % 5) * 0.085, half_y, r_dot, NAVY if side == 0 else NEUTRAL)
    c.arrow(x + 0.55, top_y - 0.06, x + 0.40, half_y + 0.07, lw=0.7, ms=5)
    c.arrow(x + 0.70, top_y - 0.06, x + 0.92, half_y + 0.07, lw=0.7, ms=5)
    c.text(x + W_STEP - 0.08, half_y - 0.115, "× 25", FS - 0.8, color=INK_2, ha="right")
    b0, bhmax, bx0, bwid = ILL_BOT + 0.02, 0.55, x + 0.22, 0.072
    c.line([bx0 - 0.04, bx0 - 0.04], [b0, b0 + bhmax], AXIS, 0.6)
    c.line([bx0 - 0.04, bx0 + 10 * 0.098], [b0, b0], AXIS, 0.6)
    for k, p in enumerate([0.74, 0.58, 0.52, 0.47, 0.42, 0.34, 0.27, 0.21, 0.15, 0.10]):
        c.rect(bx0 + k * 0.098, b0, bwid, p * bhmax, NAVY if p >= 0.4 else NEUTRAL, z=3)
    yt = b0 + 0.4 * bhmax
    c.line([bx0 - 0.04, bx0 + 10 * 0.098], [yt, yt], INK_2, 0.6, ls=DASH, z=4)
    c.text(bx0 + 10 * 0.098, yt + 0.025, "0.4", FS - 1.0, color=INK_2, ha="right", va="bottom")
    c.text(bx0 - 0.08, b0 + bhmax / 2, "π", FS, color=INK_2, ha="right")

    # ---------------- Step 3: mixture assembly ----------------
    x = X_STEP[2]
    panel(x, W_STEP, 3, "Mixture assembly", "add metabolites in order\nof π; size chosen in\nthe training folds")
    ox, oy, aw, ah = x + 0.30, ILL_BOT + 0.12, W_STEP - 0.42, ILL_TOP - ILL_BOT - 0.24
    c.line([ox, ox], [oy, oy + ah], AXIS, 0.6)
    c.line([ox, ox + aw], [oy, oy], AXIS, 0.6)
    k = np.arange(1, 8)
    auc = np.array([0.66, 0.72, 0.755, 0.775, 0.78, 0.778, 0.774])
    yy = oy + (auc - 0.60) / 0.22 * ah
    xx = ox + (k - 0.5) / 7 * aw
    band = 0.11 * ah
    c.polygon(np.r_[xx, xx[::-1]], np.r_[yy - band, (yy + band)[::-1]], NAVY_L)
    c.line(xx, yy, NAVY, 1.2, z=3)
    for i in range(7):
        c.circle(xx[i], yy[i], 0.028, NAVY, "white", 0.4, z=4)
    c.circle(xx[3], yy[3], 0.05, "none", RED, 1.0, z=5)
    c.text(xx[3], yy[3] + 0.09, "size", FS - 0.8, color=RED, ha="center", va="bottom")
    c.text(ox - 0.05, oy + ah / 2, "AUC", FS - 0.8, color=INK_2, ha="right", rotation=90)
    c.text(ox + aw / 2, oy - 0.04, "number of metabolites", FS - 0.8, color=INK_2, ha="center", va="top")

    # ---------------- Step 4: combination test ----------------
    x = X_STEP[3]
    # stable pool, or an exploratory pool for a strain without reproducible metabolites (NU; main text 2.1.4)
    panel(x, W_STEP, 4, "Combination test", "pairs and triads of the\nstable (or exploratory)\npool; permutation test")
    cx, cyc = x + W_STEP / 2, ILL_BOT + 0.70
    nodes = {"a": (cx - 0.36, cyc + 0.20), "b": (cx + 0.02, cyc + 0.42), "c": (cx + 0.38, cyc + 0.12),
             "d": (cx - 0.10, cyc - 0.16)}
    for u, v in [("a", "b"), ("b", "c")]:
        c.line(*zip(nodes[u], nodes[v]), INK_2, 1.4)
    c.line(*zip(nodes["a"], nodes["d"]), RED, 1.4, ls=DASH_SHORT)
    for (nx, ny) in nodes.values():
        c.circle(nx, ny, 0.065, NAVY, "white", 0.8)
    ly = ILL_BOT + 0.08
    c.line([x + 0.12, x + 0.30], [ly + 0.10, ly + 0.10], INK_2, 1.4)
    c.text(x + 0.35, ly + 0.10, "complementary", FS - 0.8, color=INK_2)
    c.line([x + 0.12, x + 0.30], [ly - 0.03, ly - 0.03], RED, 1.4, ls=DASH_SHORT)
    c.text(x + 0.35, ly - 0.03, "interaction", FS - 0.8, color=INK_2)

    # ---------------- arrows between panels ----------------
    ya = (ILL_TOP + ILL_BOT) / 2 + 0.15
    for x0 in [X_IN + W_IN] + [X_STEP[i] + W_STEP for i in range(3)]:
        c.arrow(x0 + 0.025, ya, x0 + GAP - 0.025, ya, lw=1.1, ms=8)

    # ---------------- outputs and the unused route ----------------
    outs = [(1, "A", "Reproducible metabolites", "per strain (Table 2)"),
            (2, "B", "Candidate mixtures", "order and size (Table 3)"),
            (3, "C", "Interaction hypotheses", "factorial tests (Table S16)")]
    for i, letter, title, sub in outs:
        xo = X_STEP[i]
        output_box(xo, W_STEP, letter, title, sub, "white", RED, "-", INK)
        c.arrow(xo + W_STEP / 2, Y_BOT - 0.02, xo + W_STEP / 2, OUT_TOP + 0.02, color=RED)
    xo = X_STEP[0]
    output_box(xo, W_STEP, None, "Not used", "direct enumeration of all\npairs and triads (Fig. 2)", "white", NEUTRAL,
               DASH, INK_2)
    c.arrow(xo + W_STEP / 2, Y_BOT - 0.02, xo + W_STEP / 2, OUT_TOP + 0.02, color=NEUTRAL, ls=DASH)

    # one common font size per text group, small enough that every member stays inside its box
    r = fig.canvas.get_renderer()
    for name, items in fit.items():
        while any(a.get_window_extent(r).x1 > ax.transData.transform((xmax, 0))[0] for a, xmax in items):
            for a, _ in items:
                a.set_fontsize(a.get_fontsize() - 0.1)
        print(f"  {name}: {items[0][0].get_fontsize():.1f} pt")
    return fig, ax, c.ops


if __name__ == "__main__":
    fig, _, _ = build()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, facecolor="white")
    print(f"Saved {OUT}.pdf/.png")
