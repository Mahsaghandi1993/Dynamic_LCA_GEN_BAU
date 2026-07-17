"""SI figure: extended borefield-life trajectory (sensitivity case S6), house style."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "data" / "derived" / "database_trajectory_comparison.csv"
OUT = REPO / "figures"

GEN = "#0E7C7B"
GEN_L = "#4FA3A1"
BASE = "#8A9BA8"
REF = "#E07B39"
INK = "#333333"
_BORF_PULSE = 2 * (7.399 - 5.989) * (6.354 / 7.399)  # kt CO2-eq (= M kg), repl. pulse
FU_END, PLOT_END = 2075, 2103

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42, "axes.labelsize": 16, "xtick.labelsize": 14,
    "ytick.labelsize": 14, "legend.fontsize": 12.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

t = pd.read_csv(TRAJ).rename(columns={
    "GEN_PkBudg1000_cumulative_Mkg": "gen", "REF_PkBudg1000_cumulative_Mkg": "ref"})
within = t[t.year <= FU_END]
gen_rate = float(t.loc[t.year == 2074, "gen"].iloc[0] - t.loc[t.year == 2073, "gen"].iloc[0])
ref_rate = float(t.loc[t.year == 2074, "ref"].iloc[0] - t.loc[t.year == 2073, "ref"].iloc[0])
gen_2074 = float(t.loc[t.year == 2074, "gen"].iloc[0])
gen_2075 = float(t.loc[t.year == 2075, "gen"].iloc[0])
ref_2075 = float(t.loc[t.year == 2075, "ref"].iloc[0])
ext = np.arange(FU_END + 1, PLOT_END + 1)

gen_base_post = (gen_2075 + _BORF_PULSE) + (ext - FU_END) * gen_rate
gen_s75 = gen_2074 + (ext - 2074) * gen_rate
gen_s75_r = gen_s75.copy()
gen_s75_r[ext >= 2100] += _BORF_PULSE
gen_s100 = gen_2074 + (ext - 2074) * gen_rate
ref_post = ref_2075 + (ext - FU_END) * ref_rate

adv_2075 = (ref_2075 - (gen_2075 + _BORF_PULSE)) / ref_2075 * 100
g100_2100 = float(gen_s100[ext == 2100][0]); r_2100 = float(ref_post[ext == 2100][0])
adv_2100 = (r_2100 - g100_2100) / r_2100 * 100

fig, ax = plt.subplots(figsize=(13.2, 7.0))
fig.subplots_adjust(left=0.09, right=0.975, top=0.96, bottom=0.12)
ax.grid(axis="y", color="#E8E8E8", lw=0.8)
ax.set_axisbelow(True)
ax.axvspan(FU_END, PLOT_END + 2, color="#F0F3F7", zorder=0)
ax.axvline(FU_END, color="#888888", lw=1.2, ls="--", alpha=0.85, zorder=3)
ax.text(FU_END - 1, 14.6, "$\\leftarrow$ 50-yr functional unit", ha="right", fontsize=12, color="#777777")
ax.text(FU_END + 1, 14.6, "beyond 50-yr functional unit $\\rightarrow$", ha="left", fontsize=12, color="#777777")

# REF
ax.plot(within.year, within.ref, color=REF, lw=2.6, zorder=4, label="REF (gas/oil system)")
ax.plot(np.r_[FU_END, ext], np.r_[ref_2075, ref_post], color=REF, lw=2.6, zorder=4)
# GEN within FU
ax.plot(within.year, within.gen, color=GEN, lw=2.6, zorder=5,
        label="GEN (identical for all cases within functional unit)")
# base: replacement pulse at 2075
ax.plot([FU_END, FU_END], [gen_2075, gen_2075 + _BORF_PULSE], color=BASE, lw=1.6, ls="--", zorder=4)
ax.plot(np.r_[FU_END, ext], np.r_[gen_2075 + _BORF_PULSE, gen_base_post], color=BASE, lw=2.0,
        ls="--", zorder=4, label="GEN base — borefield replaced at year 50")
# S6 level 75 yr
s75_plot = np.r_[gen_2074, gen_s75_r]
ax.plot(np.r_[2074, ext], s75_plot, color=GEN_L, lw=2.2, zorder=5,
        label="S6 level 75 yr — replacement deferred to 2100")
# S6 level 100 yr
ax.plot(np.r_[2074, ext], np.r_[gen_2074, gen_s100], color=GEN, lw=2.4, ls=(0, (6, 2)), zorder=5,
        label="S6 level 100 yr — no replacement in window")

ax.annotate(f"base: $\\approx${adv_2075:.0f}% below REF\n(post-replacement, year 50)",
            xy=(2078, gen_2075 + _BORF_PULSE + 0.35), fontsize=12.5, color="#5A6B78",
            ha="left", va="bottom", fontweight="bold")
ax.annotate(f"100-yr life: $\\approx${adv_2100:.0f}% below REF\n(year 75, 2100)",
            xy=(2088, g100_2100 - 0.55), fontsize=12.5, color=GEN, ha="left", va="top",
            fontweight="bold")

ax.set_xlim(2025, PLOT_END + 2)
ax.set_ylim(0, 15.4)
ax.set_xlabel("Calendar year", labelpad=8)
ax.set_ylabel("Cumulative GHG emissions, GWP100 (kt CO$_2$-eq)")
ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#DDDDDD")
for ext_ in (".png", ".pdf"):
    fig.savefig(OUT / f"figS_extended_borefield{ext_}", dpi=300, bbox_inches="tight", facecolor="white")
print("saved figS_extended_borefield; adv2075=%.1f adv2100=%.1f" % (adv_2075, adv_2100))
