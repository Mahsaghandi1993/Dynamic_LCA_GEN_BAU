"""Regenerate manuscript result figures 5, 6a, 6b, 7, 8 (F4), 9 (F5), 10.

House style: GEN teal #0E7C7B, REF orange #E07B39; label 'GHG emissions';
units kt CO2-eq (1 M kg = 1 kt); dual PNG(300dpi)+PDF export.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "derived"
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

GEN = "#0E7C7B"
REF = "#E07B39"
INK = "#333333"
M_STATIC = "#55575E"
M_TEXP = "#C7A465"
M_DYN = "#0E7C7B"
DIFF = "#7E6BAD"  # Monte-Carlo difference histogram (distinct from GEN/REF)
KT = 1_000_000.0  # kg -> kt

YLAB = "GHG emissions, GWP100 (kt CO$_2$-eq)"


def style(base=14):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": base + 2,
        "axes.titlesize": base + 2,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.8,
    })


def grid_y(ax, axis="y"):
    ax.grid(False)
    ax.grid(axis=axis, color="#E0E0E0", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext, dpi in ((".png", 300), (".pdf", 300)):
        fig.savefig(OUT / f"{stem}{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", stem)


def load_stage_data():
    frames = []
    for st in ["A", "B", "C"]:
        d = pd.read_csv(DATA / f"stage_{st}_method_comparison.csv")
        d = d[(d.metric_family == "climate") & (d.metric_key.isin(["climate_static", "dynamic_GWP100"]))]
        frames.append(d)
    d = pd.concat(frames)
    out = {}
    for case in ["GEN", "BAU"]:
        for st in ["A", "B", "C"]:
            sub = d[(d.case == case) & (d.stage == st)]
            out[(case, st, "static")] = float(sub[sub.method_mode == "conventional_static"].score_case_total.iloc[0]) / KT
            out[(case, st, "texp")] = float(sub[sub.method_mode == "time_explicit_static"].score_case_total.iloc[0]) / KT
            out[(case, st, "dyn")] = float(sub[sub.method_mode == "dynamic_climate"].score_case_total.iloc[0]) / KT
    for case in ["GEN", "BAU"]:
        for m in ["static", "texp", "dyn"]:
            out[(case, "T", m)] = sum(out[(case, st, m)] for st in "ABC")
    return out


# ---------------------------------------------------------------- fig05
def fig05(S):
    style(13)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.12)
    panels = [("A", "Stage A  –  construction"), ("B", "Stage B  –  operation"),
              ("C", "Stage C  –  end-of-life"), ("T", "Total  –  cradle to grave")]
    meths = [("static", "Static", M_STATIC), ("texp", "Time-explicit", M_TEXP), ("dyn", "Dynamic", M_DYN)]
    w = 0.24
    for ax, (st, title) in zip(axes.flat, panels):
        grid_y(ax)
        for ci, case in enumerate(["GEN", "BAU"]):
            for mi, (mk, _, mc) in enumerate(meths):
                v = S[(case, st, mk)]
                x = ci + (mi - 1) * w
                if st == "C" and case == "BAU" and mk == "dyn" and abs(v) < 0.005:
                    ax.text(x, 0.008, "$\\approx$0", ha="center", va="bottom", fontsize=11, color=INK)
                    continue
                ax.bar(x, v, width=w * 0.92, color=mc)
                off = 0.02 * max(abs(S[("BAU", st, "static")]), abs(S[("GEN", st, "static")]), 0.4)
                ax.text(x, v + off if v >= 0 else v - off, f"{v:.2f}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=11.2, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GEN", "REF"])
        ax.set_title(title, fontweight="bold", fontsize=14.5)
        ax.margins(y=0.18)
        if st == "C":
            ax.axhline(0, color="#666666", lw=0.9)
    handles = [Patch(color=c, label=l) for _, l, c in meths]
    fig.legend(handles=handles, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=13.5)
    fig.supylabel(YLAB, fontsize=15.5, x=0.02)
    save(fig, "fig05_climate_method_stage")


# ---------------------------------------------------------------- fig06
def waterfall(S, case, stem):
    style(13)
    lab = "GEN" if case == "GEN" else "REF"
    static = S[(case, "T", "static")]
    texp = S[(case, "T", "texp")]
    dyn = S[(case, "T", "dyn")]
    d1 = {st: S[(case, st, "texp")] - S[(case, st, "static")] for st in "ABC"}
    d2 = {st: S[(case, st, "dyn")] - S[(case, st, "texp")] for st in "ABC"}
    stage_cols = {"A": "#4C78A8", "B": "#7BA05B", "C": "#C25B4E"}

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    grid_y(ax)
    xs = {"static": 0.0, "texp": 3.0, "dyn": 6.6}
    bw = 0.8
    # totals
    ax.bar(xs["static"], static, width=bw, color=M_STATIC)
    ax.bar(xs["texp"], texp, width=bw, color=M_TEXP)
    ax.bar(xs["dyn"], dyn, width=bw, color=M_DYN)
    top = max(static, texp, dyn)
    for x, v in ((xs["static"], static), (xs["texp"], texp), (xs["dyn"], dyn)):
        ax.text(x, v + 0.03 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=15, fontweight="bold", color=INK)
    # transition 1 bridges
    run = static
    bx = 1.0
    for st in "ABC":
        dv = d1[st]
        if abs(dv) < 0.01:
            continue
        ax.bar(bx, dv, bottom=run, width=0.62, color=stage_cols[st], alpha=0.95)
        va = "bottom" if dv > 0 else "top"
        off = 0.02 * top if dv > 0 else -0.02 * top
        outside = run + dv + off if dv > 0 else run + dv + off
        txt_y = min(run, run + dv) - 0.025 * top if dv < 0 else run + dv + 0.02 * top
        if abs(dv) > 0.18 * top:
            ax.text(bx, run + dv / 2, f"Stage {st}", ha="center", va="center", rotation=90,
                    fontsize=12.5, color="white", fontweight="bold")
            ax.text(bx, txt_y, f"{dv:+.2f}", ha="center", va="top" if dv < 0 else "bottom",
                    fontsize=12.5, color=INK, fontweight="bold")
        else:
            ax.annotate(f"Stage {st} {dv:+.2f}", xy=(bx, run + dv), xytext=(bx, txt_y),
                        ha="center", va="top" if dv < 0 else "bottom", fontsize=11.5, color=INK, fontweight="bold")
        run += dv
        bx += 0.75
    ax.plot([xs["static"] + bw / 2, xs["texp"] - bw / 2], [static, static], ls="--", lw=0.9, color="#B99")
    # transition 2 bridges
    run = texp
    bx = 4.2
    for st in "ABC":
        dv = d2[st]
        if abs(dv) < 0.02:
            continue
        ax.bar(bx, dv, bottom=run, width=0.62, color=stage_cols[st], alpha=0.95)
        txt_y = run + dv + 0.02 * top if dv > 0 else min(run, run + dv) - 0.025 * top
        if abs(dv) > 0.18 * top:
            ax.text(bx, run + dv / 2, f"Stage {st}", ha="center", va="center", rotation=90,
                    fontsize=12.5, color="white", fontweight="bold")
            ax.text(bx, txt_y, f"{dv:+.2f}", ha="center", va="bottom" if dv > 0 else "top",
                    fontsize=12.5, color=INK, fontweight="bold")
        else:
            ax.text(bx, txt_y, f"Stage {st} {dv:+.2f}", ha="center", va="bottom" if dv > 0 else "top",
                    fontsize=11.5, color=INK, fontweight="bold")
        run += dv
        bx += 0.75
    ax.plot([xs["texp"] + bw / 2, xs["dyn"] - bw / 2], [texp, texp], ls="--", lw=0.9, color="#B99")

    ax.set_xticks([xs["static"], xs["texp"], xs["dyn"]])
    ax.set_xticklabels(["Static\n(total)", "Time-explicit\n(total)", "Dynamic\n(total)"], fontsize=13.5)
    ax.set_ylabel(f"Cumulative GHG emissions, GWP100 (kt CO$_2$-eq)", fontsize=15)
    ax.set_ylim(min(0, texp - 0.1 * top), top * 1.24)
    # transition annotations
    ymax = top * 1.15
    ax.annotate("", xy=(xs["texp"] - 0.5, ymax), xytext=(xs["static"] + 0.55, ymax),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text((xs["static"] + xs["texp"]) / 2, ymax + 0.02 * top, "Prospective grid\ndecarbonization",
            ha="center", va="bottom", fontsize=12.8, fontweight="bold", color=INK)
    ax.annotate("", xy=(xs["dyn"] - 0.55, ymax), xytext=(xs["texp"] + 0.55, ymax),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
    ax.text((xs["texp"] + xs["dyn"]) / 2, ymax + 0.02 * top, "Dynamic timing &\nnear-term methane forcing",
            ha="center", va="bottom", fontsize=12.8, fontweight="bold", color=INK)
    ax.text(0.99, 0.97, lab, transform=ax.transAxes, ha="right", va="top",
            fontsize=17, fontweight="bold", color=GEN if case == "GEN" else REF)
    save(fig, stem)


# ---------------------------------------------------------------- fig07
def fig07():
    style(13)
    g = pd.read_csv(DATA / "database_trajectory_comparison.csv")
    r = pd.read_csv(DATA / "database_trajectory_comparison_RF.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(14.2, 6.4))
    fig.subplots_adjust(wspace=0.24, bottom=0.24)

    # (a) GHG
    grid_y(a)
    a.plot(g.year, g.GEN_PkBudg1000_cumulative_Mkg, color=GEN, lw=2.8)
    a.plot(g.year, g.REF_PkBudg1000_cumulative_Mkg, color=REF, lw=2.8)
    a.plot(g.year, g.GEN_NPi_cumulative_Mkg, color=GEN, lw=2.2, ls="--", alpha=0.75)
    a.plot(g.year, g.REF_NPi_cumulative_Mkg, color=REF, lw=2.2, ls="--", alpha=0.75)
    a.fill_between(g.year, g.GEN_PkBudg1000_cumulative_Mkg, g.REF_PkBudg1000_cumulative_Mkg,
                   where=g.REF_PkBudg1000_cumulative_Mkg >= g.GEN_PkBudg1000_cumulative_Mkg,
                   color=GEN, alpha=0.10)
    cross = g[g.REF_PkBudg1000_cumulative_Mkg > g.GEN_PkBudg1000_cumulative_Mkg].iloc[0]
    a.scatter([cross.year], [cross.REF_PkBudg1000_cumulative_Mkg], s=55, color="#333333", zorder=6)
    a.annotate(f"carbon payback\n$\\approx$ {int(cross.year)}",
               xy=(cross.year, cross.REF_PkBudg1000_cumulative_Mkg),
               xytext=(cross.year + 3, cross.REF_PkBudg1000_cumulative_Mkg - 2.6),
               fontsize=12.5, fontweight="bold", color=INK,
               arrowprops=dict(arrowstyle="-", color="#555555", lw=1.0))
    ends = g.iloc[-1]
    a.text(ends.year + 1.5, ends.GEN_PkBudg1000_cumulative_Mkg - 0.15, f"{ends.GEN_PkBudg1000_cumulative_Mkg:.2f}", color=GEN, fontsize=12.5, fontweight="bold", va="top")
    a.text(ends.year + 1, ends.REF_PkBudg1000_cumulative_Mkg, f"{ends.REF_PkBudg1000_cumulative_Mkg:.2f}", color=REF, fontsize=12.5, fontweight="bold", va="center")
    a.text(ends.year + 1, ends.GEN_NPi_cumulative_Mkg + 0.25, f"{ends.GEN_NPi_cumulative_Mkg:.2f}", color=GEN, alpha=0.75, fontsize=12.5, fontweight="bold", va="bottom")
    a.text(ends.year + 1, ends.REF_NPi_cumulative_Mkg + 0.35, f"{ends.REF_NPi_cumulative_Mkg:.2f}", color=REF, alpha=0.8, fontsize=12.5, fontweight="bold", va="bottom")
    a.set_xlim(2025, 2137)
    a.set_ylim(0, float(g.REF_NPi_cumulative_Mkg.max()) * 1.12)
    a.set_xticks([2025, 2045, 2065, 2085, 2105, 2125])
    a.axvline(2075, color="#999999", lw=1.0, ls=":")
    a.axvspan(2075, 2137, color="#F4F4F0", zorder=0)
    a.text(2077, float(g.REF_NPi_cumulative_Mkg.max()) * 0.30,
           "end of 50-yr service period (2075):\ncumulative GHG emissions complete",
           fontsize=11, color="#777777", ha="left", va="center", linespacing=1.25)
    a.set_xlabel("Calendar year")
    a.set_ylabel("Cumulative GHG emissions (kt CO$_2$-eq)", fontsize=15)
    a.set_title("(a)  Global Warming Potential (dynamic GWP100)", loc="left", fontweight="bold", fontsize=14.5)

    # (b) RF
    grid_y(b)
    b.plot(r.year, r.GEN_PkBudg1000_cumulative_RFe7, color=GEN, lw=2.8)
    b.plot(r.year, r.REF_PkBudg1000_cumulative_RFe7, color=REF, lw=2.8)
    b.plot(r.year, r.GEN_NPi_cumulative_RFe7, color=GEN, lw=2.2, ls="--", alpha=0.75)
    b.plot(r.year, r.REF_NPi_cumulative_RFe7, color=REF, lw=2.2, ls="--", alpha=0.75)
    b.fill_between(r.year, r.GEN_PkBudg1000_cumulative_RFe7, r.REF_PkBudg1000_cumulative_RFe7,
                   where=r.REF_PkBudg1000_cumulative_RFe7 >= r.GEN_PkBudg1000_cumulative_RFe7,
                   color=GEN, alpha=0.10)
    adv = r[(r.REF_PkBudg1000_cumulative_RFe7 - r.GEN_PkBudg1000_cumulative_RFe7) > 0]
    # first sustained advantage year
    yr_adv = None
    diff = (r.REF_PkBudg1000_cumulative_RFe7 - r.GEN_PkBudg1000_cumulative_RFe7).to_numpy()
    for i in range(len(diff) - 3):
        if diff[i] > 0 and np.all(diff[i:] > 0):
            yr_adv = int(r.year.iloc[i]); break
    if yr_adv:
        b.axvline(yr_adv, color="#888888", lw=1.0, ls=":")
        b.text(yr_adv + 2, b.get_ylim()[1] * 0.0 + max(r.REF_PkBudg1000_cumulative_RFe7) * 0.86,
               f"GEN advantage\nfrom $\\approx$ {yr_adv}", fontsize=12.5, fontweight="bold", color="#555555")
    er = r.iloc[-1]
    b.text(er.year + 1, er.GEN_PkBudg1000_cumulative_RFe7 - 0.02, f"{er.GEN_PkBudg1000_cumulative_RFe7:.2f}", color=GEN, fontsize=12.5, fontweight="bold", va="top")
    b.text(er.year + 1, er.REF_PkBudg1000_cumulative_RFe7, f"{er.REF_PkBudg1000_cumulative_RFe7:.2f}", color=REF, fontsize=12.5, fontweight="bold", va="center")
    b.text(er.year + 1, er.GEN_NPi_cumulative_RFe7 + 0.06, f"{er.GEN_NPi_cumulative_RFe7:.2f}", color=GEN, alpha=0.75, fontsize=12.5, fontweight="bold", va="bottom")
    b.text(er.year + 1, er.REF_NPi_cumulative_RFe7 + 0.12, f"{er.REF_NPi_cumulative_RFe7:.2f}", color=REF, alpha=0.8, fontsize=12.5, fontweight="bold", va="bottom")
    b.set_xlim(2025, 2137)
    b.set_xticks([2025, 2045, 2065, 2085, 2105, 2125])
    b.set_xlabel("Calendar year")
    b.set_ylabel("Cumulative RF ($\\times 10^{-7}$ W$\\cdot$yr m$^{-2}$)", fontsize=15)
    b.set_title("(b)  Radiative forcing (100-yr horizon)", loc="left", fontweight="bold", fontsize=14.5)

    handles = [
        Line2D([0], [0], color=GEN, lw=2.8, label="GEN · PkBudg1000 (decarbonizing grid)"),
        Line2D([0], [0], color=GEN, lw=2.2, ls="--", alpha=0.75, label="GEN · NPi (slower-policy grid)"),
        Line2D([0], [0], color=REF, lw=2.8, label="REF · PkBudg1000 (decarbonizing grid)"),
        Line2D([0], [0], color=REF, lw=2.2, ls="--", alpha=0.8, label="REF · NPi (slower-policy grid)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.015))
    save(fig, "fig07_dynamic_cumulative_trajectory")


# ---------------------------------------------------------------- F4 tornado (fig 8)
def fig08():
    style(15)
    rows = [  # label, low, high  (base 42.4) -- from Table 5 / final sensitivity results
        ("S1  Fuel split (REF gas:oil share)", 42.7, 42.2),
        ("S2  CH$_4$ leakage rate (REF fugitive)", 40.7, 43.9),
        ("S3  Background electricity grid trajectory", 46.0, 42.2),
        ("S4  Equipment efficiency / heat-pump COP", 35.0, 48.2),
        ("S5  System service life (40 / 60 yr)", 41.2, 40.9),
        ("S6  GEN borefield residual life (75 / 100 yr)", 49.5, 53.1),
        ("S7  GEN Stage A embodied GHG ($\\pm$20%)", 35.8, 49.1),
    ]
    base = 42.4
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.9, bottom=0.13)
    grid_y(ax, "x")
    ys = np.arange(len(rows))[::-1]
    for y, (label, lo, hi) in zip(ys, rows):
        lo_v, hi_v = sorted((lo, hi))
        ax.plot([lo_v, hi_v], [y, y], color="#B9CDD3", lw=11, solid_capstyle="round", zorder=2)
        ax.plot([lo_v, hi_v], [y, y], color="#39707E", lw=3.2, solid_capstyle="round", zorder=3)
        ax.scatter([lo], [y], s=170, color="#C9720E", zorder=5, edgecolor="white", linewidth=1.2)
        ax.scatter([hi], [y], s=170, color="#0E7C7B", zorder=5, edgecolor="white", linewidth=1.2)
        left, right = (lo, hi) if lo < hi else (hi, lo)
        lcol = "#C9720E" if left == lo else "#0E7C7B"
        rcol = "#0E7C7B" if right == hi else "#C9720E"
        ax.text(left - 0.7, y, f"{left:.1f}%", ha="right", va="center", fontsize=14.5, fontweight="bold", color=lcol)
        ax.text(right + 0.7, y, f"{right:.1f}%", ha="left", va="center", fontsize=14.5, fontweight="bold", color=rcol)
    ax.axvline(base, color="#4A4A4A", lw=1.6, ls="--", zorder=1)
    ax.text(base, len(rows) - 0.28, f"Base case ({base:.1f}%)", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#4A4A4A")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=15)
    ax.set_xlim(28, 60)
    ax.set_ylim(-0.6, len(rows) - 0.3 + 0.75)
    ax.set_xlabel("GEN reduction in life-cycle GHG emissions vs REF (%)", fontsize=16.5, labelpad=9)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C9720E", markersize=13, label="Low level of parameter"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0E7C7B", markersize=13, label="High level of parameter"),
        Line2D([0], [0], color="#4A4A4A", lw=1.6, ls="--", label="Base case (42.4%)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, frameon=False,
               fontsize=13.5, bbox_to_anchor=(0.55, -0.005))
    fig.subplots_adjust(bottom=0.19)
    save(fig, "F4_sensitivity_tornado")


# ---------------------------------------------------------------- F5 MC (fig 9)
def fig09():
    style(13)
    draws = pd.read_csv(DATA / "mc_draws_three_method.csv")
    meths = [("static", "(a)  Conventional static\n(frozen 2025 grid)"),
             ("time_explicit", "(b)  Time-explicit static\n(SSP2-PkBudg1000)"),
             ("dynamic", "(c)  Dynamic climate\n(IPCC AR6)")]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.30, top=0.90)
    for col, (mk, title) in enumerate(meths):
        d = draws[draws.method == mk]
        gen = d.GEN_kgCO2e.to_numpy() / KT
        ref = d.REF_kgCO2e.to_numpy() / KT
        diff = d.REF_minus_GEN_kgCO2e.to_numpy() / KT

        ax = axes[0, col]
        grid_y(ax)
        parts = ax.violinplot([gen, ref], positions=[0, 1], widths=0.72, showextrema=False)
        for pc, c in zip(parts["bodies"], [GEN, REF]):
            pc.set_facecolor(c); pc.set_alpha(0.92); pc.set_edgecolor("none")
        for i, (arr, c) in enumerate([(gen, GEN), (ref, REF)]):
            med, p5, p95 = np.median(arr), np.percentile(arr, 5), np.percentile(arr, 95)
            ax.vlines(i, p5, p95, color="white", lw=2.4, zorder=5)
            ax.scatter([i], [med], s=42, color="white", edgecolor="#555555", zorder=6)
            ax.text(i, np.percentile(arr, 99.7) + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.001,
                    f"{med:.1f}", ha="center", va="bottom", fontsize=14.5, fontweight="bold", color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["GEN", "REF"], fontweight="bold", fontsize=13.5)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.margins(y=0.20)
        if col == 0:
            ax.set_ylabel(YLAB, fontsize=14.5)

        ax = axes[1, col]
        grid_y(ax)
        ax.hist(diff, bins=48, color=DIFF, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="#C23B22", lw=2.0)
        ax.text(0, ax.get_ylim()[1] * 0.55, "REF = GEN", rotation=90, ha="right", va="center",
                color="#C23B22", fontsize=12, fontweight="bold")
        med = np.median(diff); p5, p95 = np.percentile(diff, 5), np.percentile(diff, 95)
        ax.axvline(med, color="#333333", lw=1.4, ls="--")
        ax.text(0.97, 0.95,
                f"P(GEN < REF) = 100%\nmedian $\\Delta$ = {med:.1f} kt CO$_2$-eq\n90% CI [{p5:.1f}, {p95:.1f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=12.3,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CCCCCC", alpha=0.95))
        ax.set_xlabel("REF $-$ GEN advantage (kt CO$_2$-eq)", fontsize=14)
        if col == 0:
            ax.set_ylabel("Monte Carlo draws", fontsize=14.5)
    save(fig, "F5_paired_monte_carlo_three_method")


# ---------------------------------------------------------------- fig10
def fig10():
    style(15)
    rows = []
    for case in ["GEN", "BAU"]:
        t = pd.read_csv(DATA / f"total_method_comparison_{case}.csv")
        t = t[t.metric_family == "non_climate"]
        w = t.pivot_table(index="metric_key", columns="method_mode", values="score_case_total", aggfunc="first")
        w["case"] = case
        rows.append(w.reset_index())
    d = pd.concat(rows)
    order = ["fossil_energy_demand", "ozone_depletion", "acidification",
             "eutrophication", "particulate_matter", "photochemical_oxidant_formation"]
    labels = {"fossil_energy_demand": "Fossil energy demand", "ozone_depletion": "Ozone depletion",
              "acidification": "Acidification", "eutrophication": "Eutrophication",
              "particulate_matter": "Particulate matter formation",
              "photochemical_oxidant_formation": "Photochemical oxidant\nformation"}
    gen = d[d.case == "GEN"].set_index("metric_key")
    ref = d[d.case == "BAU"].set_index("metric_key")
    pst, pte = [], []
    for k in order:
        pst.append((gen.loc[k, "conventional_static"] / ref.loc[k, "conventional_static"] - 1) * 100)
        pte.append((gen.loc[k, "time_explicit_static"] / ref.loc[k, "time_explicit_static"] - 1) * 100)

    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.12)
    grid_y(ax, "x")
    ys = np.arange(len(order))[::-1]
    h = 0.34
    b1 = ax.barh(ys + h / 2 + 0.02, pst, height=h, color="#55575E", label="Conventional static")
    b2 = ax.barh(ys - h / 2 - 0.02, pte, height=h, color="#C9720E", label="Time-explicit static")
    for bars, vals in ((b1, pst), (b2, pte)):
        for bar, v in zip(bars, vals):
            x = bar.get_width()
            ax.text(x + (4 if x >= 0 else -4), bar.get_y() + bar.get_height() / 2,
                    f"{v:+.0f}%", ha="left" if x >= 0 else "right", va="center",
                    fontsize=14.5, fontweight="bold", color=INK)
    ax.axvline(0, color="#333333", lw=1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[k] for k in order], fontsize=15.5)
    ax.set_xlim(-75, 255)
    ax.set_xlabel("GEN relative to REF (%)   —   negative = GEN lower (better)", fontsize=16, labelpad=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncols=2, frameon=False, fontsize=15,
              title=None)
    save(fig, "fig10_nonclimate_summary")
    return dict(zip(order, zip(pst, pte)))


if __name__ == "__main__":
    S = load_stage_data()
    fig05(S)
    waterfall(S, "GEN", "fig06a_method_decomposition_gen")
    waterfall(S, "BAU", "fig06b_method_decomposition_ref")
    fig07()
    fig08()
    fig09()
    vals = fig10()
    print("fig10 percentages:", {k: (round(a), round(b)) for k, (a, b) in vals.items()})
    print("stage totals:", {k: round(v, 3) for k, v in S.items() if k[1] == 'T'})
