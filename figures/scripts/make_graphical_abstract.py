"""Graphical abstract v2 — Elsevier landscape (13x5 in, 300 dpi).

Act 1: calendar-dated life cycle (EN 15978 stages) around a professional
       urban cross-section with borefield (red/blue U-tubes).
Act 2: three temporal LCA methods.
Act 3: trajectory, 2036 carbon payback, -42% hero number.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "data" / "derived" / "database_trajectory_comparison.csv"
OUT = REPO / "figures"

GEN = "#0E7C7B"
GEN_D = "#0A5B5A"
REFC = "#E07B39"
INK = "#2B3440"
MUTED = "#6B7280"
CARD = "#FFFFFF"
EDGE = "#DFE3E8"
HDR = "#3E5C76"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "pdf.fonttype": 42})

fig = plt.figure(figsize=(13, 5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(6.5, 4.72, "When does a geothermal energy network pay back its carbon?",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
ax.text(6.5, 4.38, "Time-explicit, prospective and dynamic-climate LCA of networked geothermal (GEN) vs a gas/oil reference (REF)",
        ha="center", va="center", fontsize=10.5, color=MUTED)

def card(x, y, w, h, header):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=CARD, edgecolor=EDGE, lw=1.4, zorder=1))
    ax.add_patch(FancyBboxPatch((x + 0.08, y + h - 0.60), w - 0.16, 0.52,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=HDR, edgecolor="none", zorder=2))
    ax.text(x + w / 2, y + h - 0.34, header, ha="center", va="center",
            fontsize=9.8, fontweight="bold", color="white", zorder=3, linespacing=1.3)

y0, h0 = 0.42, 3.68

# ================================================================ ACT 1
x0, w0 = 0.25, 4.85
card(x0, y0, w0, h0, "Two systems deliver one identical service; every\nlife-cycle stage is dated on a calendar (EN 15978)")

# ---------- central urban cross-section ----------
ix, iw = x0 + 1.42, 2.02          # illustration bounds
street = 2.30
gnd_top, gnd_bot = street - 0.035, 1.30
rng = np.random.default_rng(7)

# concrete ground with speckles
ax.add_patch(Rectangle((ix, gnd_bot), iw, gnd_top - gnd_bot, facecolor="#D8D4CC",
                       edgecolor="none", zorder=2))
spx = rng.uniform(ix + 0.03, ix + iw - 0.03, 130)
spy = rng.uniform(gnd_bot + 0.03, gnd_top - 0.05, 130)
sps = rng.uniform(0.004, 0.014, 130)
for sx_, sy_, ss_ in zip(spx, spy, sps):
    ax.add_patch(Circle((sx_, sy_), ss_, facecolor="#B9B4AA", edgecolor="none",
                        alpha=0.6, zorder=3))
# street band
ax.add_patch(Rectangle((ix, gnd_top), iw, 0.05, facecolor="#4A5158", edgecolor="none", zorder=4))
ax.plot([ix + 0.06, ix + iw - 0.06], [gnd_top + 0.025, gnd_top + 0.025], color="white",
        lw=0.7, ls=(0, (2.2, 2.2)), zorder=5, alpha=0.85)

# header pipe + U-tube boreholes (alternating warm/cold legs)
hdr_y = gnd_top - 0.09
ax.plot([ix + 0.16, ix + iw - 0.16], [hdr_y, hdr_y], color="#5B6570", lw=2.6, zorder=5,
        solid_capstyle="round")
bores = np.linspace(ix + 0.24, ix + iw - 0.24, 6)
for i, bx in enumerate(bores):
    c = "#C0574F" if i % 2 == 0 else "#4C78A8"
    depth = gnd_bot + 0.06
    ax.plot([bx - 0.035, bx - 0.035, bx + 0.035, bx + 0.035],
            [hdr_y, depth + 0.05, depth + 0.05, hdr_y], color=c, lw=2.0, zorder=5,
            solid_capstyle="round")
    ax.add_patch(Circle((bx, depth + 0.05), 0.035, facecolor="none", edgecolor=c, lw=2.0, zorder=5))

# skyline (muted professional greys with window grids)
def building(bx, bw, bh, color, wc=3):
    ax.add_patch(Rectangle((bx, street + 0.015), bw, bh, facecolor=color,
                           edgecolor="#59616A", lw=0.6, zorder=5))
    rows = max(2, int(bh / 0.115))
    for r in range(rows):
        for ccol in range(wc):
            wx = bx + bw * (0.14 + 0.75 * ccol / wc) + bw * 0.06
            wy = street + 0.055 + r * (bh - 0.07) / rows
            if wy + 0.045 < street + bh:
                ax.add_patch(Rectangle((wx, wy), bw * 0.14, 0.042, facecolor="#EDF0F2",
                                       edgecolor="none", zorder=6))

building(ix + 0.06, 0.34, 0.92, "#7C868F", 3)
building(ix + 0.46, 0.30, 0.58, "#98A1A8", 3)
# gable house
hx0 = ix + 0.82
ax.add_patch(Rectangle((hx0, street + 0.015), 0.30, 0.30, facecolor="#A8B0B6",
                       edgecolor="#59616A", lw=0.6, zorder=5))
ax.add_patch(Polygon([(hx0 - 0.03, street + 0.315), (hx0 + 0.33, street + 0.315),
                      (hx0 + 0.15, street + 0.47)], facecolor="#8C959D",
                     edgecolor="#59616A", lw=0.6, zorder=5))
ax.add_patch(Rectangle((hx0 + 0.10, street + 0.10), 0.10, 0.09, facecolor="#EDF0F2",
                       edgecolor="none", zorder=6))
building(ix + 1.18, 0.36, 0.75, "#6E7780", 3)
building(ix + 1.60, 0.32, 0.48, "#98A1A8", 3)
# tree
tx = ix + 1.985
ax.plot([tx, tx], [street + 0.02, street + 0.16], color="#6B5B45", lw=2.2, zorder=5)
ax.add_patch(Circle((tx, street + 0.26), 0.105, facecolor="#8FAF8B", edgecolor="#6E8F6A",
                    lw=0.7, zorder=5))

# heat-pump link glyph: small teal squares at two buildings
for bx in (ix + 0.23, ix + 1.36):
    ax.add_patch(Rectangle((bx - 0.045, street + 0.015), 0.09, 0.075, facecolor=GEN,
                           edgecolor="white", lw=0.6, zorder=7))

# ---------- stage chips around the illustration ----------
def chip(cx, cy, line1, line2, color, cw=1.24, ch=0.56):
    ax.add_patch(FancyBboxPatch((cx - cw / 2, cy - ch / 2), cw, ch,
                                boxstyle="round,pad=0.03,rounding_size=0.22",
                                facecolor=color, edgecolor="white", lw=1.2, zorder=8))
    ax.text(cx, cy + 0.10, line1, ha="center", va="center", fontsize=9.0,
            fontweight="bold", color="white", zorder=9)
    ax.text(cx, cy - 0.13, line2, ha="center", va="center", fontsize=8.2,
            color="white", zorder=9)

cA = (x0 + 0.78, 3.04)
cB6 = (x0 + w0 - 0.78, 3.04)
cM = (x0 + w0 - 0.78, 1.42)
cC = (x0 + 0.78, 1.42)
chip(*cA, "A · Construction", "2025", "#3E5C76")
chip(*cB6, "B6 · Operation", "2026–2075", GEN)
chip(*cM, "B2–B4 · Maint.", "2040–2070", "#B08A3E")
chip(*cC, "C · End of life", "2075", "#A65A4E")

def arrow(p, q, rad, color="#8B97A3"):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=16, color=color,
                                 lw=1.8, zorder=7))

arrow((cA[0] + 0.68, cA[1] + 0.14), (cB6[0] - 0.68, cB6[1] + 0.14), -0.25)
arrow((cB6[0], cB6[1] - 0.34), (cM[0], cM[1] + 0.34), 0.0)
arrow((cM[0] - 0.68, cM[1] - 0.14), (cC[0] + 0.68, cC[1] - 0.14), -0.25)
# dashed C->A: residual life credited only in Module D sensitivity (S6)
ax.add_patch(FancyArrowPatch((cC[0], cC[1] + 0.34), (cA[0], cA[1] - 0.34),
                             connectionstyle="arc3,rad=0.0", arrowstyle="-|>",
                             mutation_scale=13, color="#B6BFC8", lw=1.3, ls=(0, (4, 3)),
                             zorder=7))
ax.text(cA[0] - 0.40, (cA[1] + cC[1]) / 2, "Module D\n(S6 only)",
        fontsize=7.3, color="#8B97A3", ha="center", va="center", style="italic",
        linespacing=1.25)

ax.text(x0 + w0 / 2, y0 + 0.42,
        "GEN: shared borefield + heat pumps  ·  REF: gas/oil furnaces + electric AC",
        ha="center", fontsize=8.0, color=MUTED)
ax.text(x0 + w0 / 2, y0 + 0.18,
        "identical delivered service: 89.375 GWh$_{th}$ · 37 buildings · 50 years",
        ha="center", fontsize=8.8, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F1F5F4", edgecolor="#D5DEDC"))

# ================================================================ ACT 2
x1, w1 = 5.42, 3.02
card(x1, y0, w1, h0, "One inventory, three methods:\nthe result depends on time")
cy = y0 + 2.34
ty = cy + 0.52
ax.plot([x1 + 0.56, x1 + w1 - 0.56], [ty, ty], color="#9AA6B2", lw=2.2, solid_capstyle="round")
for yr in np.linspace(x1 + 0.56, x1 + w1 - 0.56, 6):
    ax.scatter([yr], [ty], s=24, color=HDR, zorder=5)
ax.text(x1 + 0.30, ty, "2025", fontsize=7.5, color=HDR, ha="right", va="center")
ax.text(x1 + w1 - 0.30, ty, "2075", fontsize=7.5, color=HDR, ha="left", va="center")
ax.text(x1 + w1 / 2, cy + 0.24, "every exchange dated and linked to a", ha="center",
        fontsize=8.4, color=MUTED)
ax.text(x1 + w1 / 2, cy + 0.06, "decarbonizing grid (premise · SSP2)", ha="center",
        fontsize=8.4, color=MUTED)

rows = [("Conventional static", "frozen 2025 grid", "− 44%", "#55575E"),
        ("Time-explicit static", "future grids, static factors", "− 28%", "#C7A465"),
        ("Dynamic climate", "future grids + emission timing", "− 42%", GEN)]
ry = cy - 0.30
for name, sub, val, color in rows:
    ax.add_patch(FancyBboxPatch((x1 + 0.24, ry - 0.27), w1 - 0.48, 0.56,
                                boxstyle="round,pad=0.02,rounding_size=0.07",
                                facecolor="#F7F8FA", edgecolor=EDGE, lw=1.0, zorder=3))
    ax.add_patch(Rectangle((x1 + 0.24, ry - 0.27), 0.08, 0.60, facecolor=color,
                           edgecolor="none", zorder=4))
    ax.text(x1 + 0.44, ry + 0.10, name, fontsize=9.2, fontweight="bold", color=INK,
            va="center", zorder=5)
    ax.text(x1 + 0.44, ry - 0.13, sub, fontsize=7.4, color=MUTED, va="center", zorder=5)
    ax.text(x1 + w1 - 0.36, ry, val, fontsize=12.5, fontweight="bold", color=color,
            va="center", ha="right", zorder=5)
    ry -= 0.70
ax.text(x1 + w1 / 2, y0 + 0.20, "GEN vs REF, GHG emissions (GWP100, kt CO$_2$-eq)",
        ha="center", va="center", fontsize=7.3, color=MUTED)

for ax_x in (5.20, 8.58):
    ax.add_patch(FancyArrowPatch((ax_x - 0.04, 2.25), (ax_x + 0.20, 2.25),
                                 arrowstyle="-|>", mutation_scale=20, color="#8B97A3", lw=2.2))

# ================================================================ ACT 3
x2, w2 = 8.82, 3.93
card(x2, y0, w2, h0, "GEN repays its construction carbon by\n$\\approx$2036 and emits 42% less by 2075")
sub = fig.add_axes([0.716, 0.16, 0.155, 0.50])
t = pd.read_csv(TRAJ)
sub.plot(t.year, t.GEN_PkBudg1000_cumulative_Mkg, color=GEN, lw=2.8)
sub.plot(t.year, t.REF_PkBudg1000_cumulative_Mkg, color=REFC, lw=2.8)
sub.fill_between(t.year, t.GEN_PkBudg1000_cumulative_Mkg, t.REF_PkBudg1000_cumulative_Mkg,
                 where=t.REF_PkBudg1000_cumulative_Mkg >= t.GEN_PkBudg1000_cumulative_Mkg,
                 color=GEN, alpha=0.12)
cross = t[t.REF_PkBudg1000_cumulative_Mkg > t.GEN_PkBudg1000_cumulative_Mkg].iloc[0]
sub.scatter([cross.year], [cross.REF_PkBudg1000_cumulative_Mkg], s=52, color=INK, zorder=6)
sub.annotate("carbon payback\n$\\approx$ 2036",
             xy=(cross.year, cross.REF_PkBudg1000_cumulative_Mkg),
             xytext=(cross.year + 2.5, cross.REF_PkBudg1000_cumulative_Mkg - 3.9),
             fontsize=9.5, fontweight="bold", color=INK,
             arrowprops=dict(arrowstyle="-", color="#777777", lw=1.0))
sub.text(2054, float(t.loc[t.year == 2054, "REF_PkBudg1000_cumulative_Mkg"].iloc[0]) + 0.75,
         "REF", color=REFC, fontsize=10.5, fontweight="bold", ha="center", va="bottom")
sub.text(2060, float(t.loc[t.year == 2060, "GEN_PkBudg1000_cumulative_Mkg"].iloc[0]) - 0.85,
         "GEN", color=GEN, fontsize=10.5, fontweight="bold", ha="center", va="top")
sub.set_xlim(2025, 2075)
sub.set_ylim(0, 12.4)
sub.set_xticks([2025, 2050, 2075])
sub.set_yticks([0, 4, 8, 12])
sub.tick_params(labelsize=7.5, colors=MUTED)
sub.set_ylabel("cumulative GHG emissions\n(kt CO$_2$-eq)", fontsize=7.4, color=MUTED, labelpad=2)
for sp in ("top", "right"):
    sub.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    sub.spines[sp].set_color("#C4CBD2")
sub.set_facecolor("none")

nx = x2 + w2 - 0.26
ax.text(nx, y0 + 2.74, "− 42%", ha="right", va="center", fontsize=24,
        fontweight="bold", color=GEN_D)
ax.text(nx, y0 + 2.28, "life-cycle GHG\nemissions by 2075", ha="right", va="center",
        fontsize=9.0, color=INK, linespacing=1.25)
ax.text(nx, y0 + 1.58, "robust across S1–S7,\npaired Monte Carlo,\nslower-policy grid",
        ha="right", va="center", fontsize=7.7, color=MUTED, linespacing=1.35)

for ext in (".pdf", ".png"):
    fig.savefig(OUT / f"graphical_abstract_elsevier{ext}", dpi=300, facecolor="white")
plt.close(fig)
print("saved v2; crossover:", int(cross.year))
