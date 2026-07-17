"""
run_three_method_MC_2026_06_12.py
===================================
3-method paired Monte Carlo on the 12 June inventory.

Calibrated-delta approach
  - T1 headline values anchor the three methods (static / time-explicit / dynamic).
  - S1 (fuel split) and S2 (methane leakage) deltas are computed analytically
    from the same B6 physical inventory functions used in build_ref_final_package.py.
  - The same absolute direct-combustion delta is applied across all three LCIA
    methods (defensible: CO2 GWP100 = 1; CH4 GWP100 ≈ 30 in all methods).
  - Shared and system-specific multipliers provide correlated parametric uncertainty.

N = 10,000 draws, seed = 2026.

Outputs → export/manuscript_REF_final_2026_06_12/sensitivity/three_method_MC_2026_06_12/
  mc_draws_three_method.csv
  mc_summary_three_method.csv
  F5_paired_monte_carlo_three_method.png  (600 dpi)
  F5_paired_monte_carlo_three_method.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
WORKSPACE = Path(__file__).resolve().parents[2]
OUT_DIR = (
    WORKSPACE
    / "export/manuscript_REF_final_2026_06_12/sensitivity/three_method_MC_2026_06_12"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# T1 headline anchors  (12 June authoritative inventory)
# ══════════════════════════════════════════════════════════════════════════════
T1: dict[str, dict[str, float]] = {
    "static":        {"GEN": 18_556_389.0, "REF": 33_267_682.0},
    "time_explicit": {"GEN":  6_239_949.0, "REF":  8_611_517.0},
    "dynamic":       {"GEN":  6_353_957.0, "REF": 11_031_285.0},
}
METHODS = ["static", "time_explicit", "dynamic"]

# ══════════════════════════════════════════════════════════════════════════════
# B6 direct-combustion physics
# (mirrors _prepare_b6_cache / _b6_inventory / _direct_climate in
#  build_ref_final_package.py; constants from b6_case_config.yaml and
#  export/b6/comparison/gen_vs_bau_b6_annual_by_group.csv)
# ══════════════════════════════════════════════════════════════════════════════
# Annual single-family heating/DHW thermal loads (kWh_th/year)
SF_HEAT_TH = 198_984.612_25
SF_DHW_TH  = 22_092.841_919_676_226
# Annual commercial gas (kWh_fuel/year; fixed regardless of SF fuel split)
COMM_GAS_FUEL_ANN = 193_545.013_190_732

# Fuel conversion efficiencies (kWh_fuel per kWh_th delivered)
C_GAS_HEAT = 1.0 / 0.82
C_OIL_HEAT = 1.0 / 0.80
C_GAS_DHW  = 1.0 / 0.62
C_OIL_DHW  = 1.0 / 0.60

# Config constants
SERVICE_LIFE            = 50.0   # years
GAS_LHV_KWH_PER_KG_CH4 = 13.9   # kWh/kg CH4 (for leakage mass conversion)

# Emission factors (kg substance per kWh fuel)
EF_GAS_CO2 = 0.181
EF_GAS_CH4 = 3.6e-6
EF_GAS_N2O = 3.6e-7
EF_OIL_CO2 = 0.267
EF_OIL_CH4 = 6.0e-7
EF_OIL_N2O = 6.0e-7

# GWP100 characterization factors (AR6; same scalar used in build_ref_final_package.py)
GWP_CO2 = 1.0
GWP_CH4 = 29.8
GWP_N2O = 273.0

# Base parameter values
BASE_GAS_SHARE = 0.70
BASE_LEAK_RATE = 0.0204

# ══════════════════════════════════════════════════════════════════════════════
# MC settings
# ══════════════════════════════════════════════════════════════════════════════
N_DRAWS   = 10_000
SEED      = 2026
MILLION   = 1.0e6
SANITY_TOL = 0.05  # 5 % median tolerance vs T1


# ══════════════════════════════════════════════════════════════════════════════
# Analytical direct-climate computation (vectorized)
# ══════════════════════════════════════════════════════════════════════════════

def _direct_climate(gas_share: np.ndarray, leak_rate: np.ndarray) -> np.ndarray:
    """
    Vectorized GWP100 of direct combustion + leakage for REF Stage B (kg CO2e).
    Mirrors the scalar _b6_inventory / _direct_climate pair in
    build_ref_final_package.py.
    """
    oil_share = 1.0 - gas_share

    sf_gas = (SF_HEAT_TH * gas_share * C_GAS_HEAT
              + SF_DHW_TH * gas_share * C_GAS_DHW)
    sf_oil = (SF_HEAT_TH * oil_share * C_OIL_HEAT
              + SF_DHW_TH * oil_share * C_OIL_DHW)

    gas_fuel = (COMM_GAS_FUEL_ANN + sf_gas) * SERVICE_LIFE
    oil_fuel = sf_oil * SERVICE_LIFE

    co2 = gas_fuel * EF_GAS_CO2 + oil_fuel * EF_OIL_CO2
    ch4 = (gas_fuel * EF_GAS_CH4 + oil_fuel * EF_OIL_CH4
           + gas_fuel * leak_rate / GAS_LHV_KWH_PER_KG_CH4)
    n2o = gas_fuel * EF_GAS_N2O + oil_fuel * EF_OIL_N2O

    return co2 * GWP_CO2 + ch4 * GWP_CH4 + n2o * GWP_N2O


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

def run_mc() -> pd.DataFrame:
    """
    Sample N_DRAWS parameter sets, compute GEN/REF for all three methods,
    return a long-format DataFrame with 3 × N_DRAWS rows.
    """
    rng = np.random.default_rng(SEED)

    # S1: single-family gas share (uniform)
    gas_share = rng.uniform(0.60, 0.75, N_DRAWS)
    # S2: methane leakage rate (triangular)
    leak_rate = rng.triangular(0.010, BASE_LEAK_RATE, 0.030, N_DRAWS)
    # Shared systematic uncertainty (applied to both GEN and REF, all methods)
    common = rng.normal(1.0, 0.06, N_DRAWS)
    # System-specific idiosyncratic noise
    gen_mult = rng.normal(1.0, 0.05, N_DRAWS)
    ref_mult = rng.normal(1.0, 0.05, N_DRAWS)

    # Analytical S1/S2 delta for REF (same absolute value across all LCIA methods;
    # direct CO2 has GWP100 = 1 in all methods; CH4 GWP100 ≈ 30 in all methods)
    base_dc = _direct_climate(
        np.full(N_DRAWS, BASE_GAS_SHARE),
        np.full(N_DRAWS, BASE_LEAK_RATE),
    )
    delta_dc = _direct_climate(gas_share, leak_rate) - base_dc  # (N_DRAWS,)

    frames: list[pd.DataFrame] = []
    for method in METHODS:
        base_gen = T1[method]["GEN"]
        base_ref = T1[method]["REF"]

        # GEN: geothermal — no sensitivity to gas/oil parameters (S1/S2)
        gen_draws = base_gen * common * gen_mult

        # REF: direct-combustion delta added before shared scaling
        ref_draws = (base_ref + delta_dc) * common * ref_mult

        frames.append(pd.DataFrame({
            "draw_id":              np.arange(N_DRAWS, dtype=np.int32),
            "method":               method,
            "GEN_kgCO2e":           gen_draws,
            "REF_kgCO2e":           ref_draws,
            "REF_minus_GEN_kgCO2e": ref_draws - gen_draws,
        }))

    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(draws: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = draws[draws["method"] == method]
        g = sub["GEN_kgCO2e"].to_numpy()
        r = sub["REF_kgCO2e"].to_numpy()
        d = sub["REF_minus_GEN_kgCO2e"].to_numpy()
        rows.append({
            "method":       method,
            "GEN_median":   float(np.median(g)),
            "GEN_p5":       float(np.percentile(g,  5)),
            "GEN_p95":      float(np.percentile(g, 95)),
            "REF_median":   float(np.median(r)),
            "REF_p5":       float(np.percentile(r,  5)),
            "REF_p95":      float(np.percentile(r, 95)),
            "diff_median":  float(np.median(d)),
            "diff_p5":      float(np.percentile(d,  5)),
            "diff_p95":     float(np.percentile(d, 95)),
            "P_GEN_lt_REF": float((g < r).mean()),
            "n_draws":      int(len(g)),
        })
    return pd.DataFrame(rows)


def sanity_check(summary: pd.DataFrame) -> None:
    print("\n  Sanity checks (median vs T1, tolerance ±5%):")
    for method in METHODS:
        row = summary[summary["method"] == method].iloc[0]
        for case in ("GEN", "REF"):
            med = float(row[f"{case}_median"])
            t1  = T1[method][case]
            pct = abs(med - t1) / t1
            flag = "OK" if pct <= SANITY_TOL else "FAIL"
            print(f"    [{method:15s} {case}]  median={med:>14,.0f}  T1={t1:>14,.0f}"
                  f"  diff={100*pct:.2f}%  [{flag}]")
            if flag == "FAIL":
                raise RuntimeError(
                    f"Sanity FAIL: {method} {case} — median {med:,.0f} vs T1 {t1:,.0f}"
                )
    print("  All sanity checks passed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Publication figure  (2 rows × 3 columns)
# ══════════════════════════════════════════════════════════════════════════════
C_GEN  = "#2F7369"   # teal
C_REF  = "#C0603A"   # orange
C_HIST = "#4C7A70"   # histogram fill
C_ZERO = "#CC2222"   # red "REF = GEN" line

METHOD_LABELS = {
    "static":        "Conventional static\n(frozen 2025 grid)",
    "time_explicit": "Time-explicit static\n(SSP2-PkBudg1000)",
    "dynamic":       "Dynamic GWP100\n(IPCC AR6 IRF)",
}


def _violin_pair(ax: plt.Axes, gen_m: np.ndarray, ref_m: np.ndarray,
                 t1_gen: float, t1_ref: float, title: str) -> None:
    """Side-by-side violin for GEN / REF in millions kg CO2e."""
    vp = ax.violinplot(
        [gen_m, ref_m], positions=[0, 1], widths=0.58,
        showmedians=True, showextrema=False,
    )
    for body, color in zip(vp["bodies"], [C_GEN, C_REF]):
        body.set_facecolor(color)
        body.set_alpha(0.78)
        body.set_edgecolor("none")
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2.2)

    # 5–95 % whiskers
    for pos, vals, color in [(0, gen_m, C_GEN), (1, ref_m, C_REF)]:
        q5, q95 = np.percentile(vals, [5, 95])
        ax.plot([pos, pos], [q5, q95], color=color, lw=1.8, zorder=4)

    # T1 dashes
    ax.hlines(t1_gen / MILLION, -0.28, 0.28, colors=C_GEN,
              lw=1.6, linestyles="--", zorder=6)
    ax.hlines(t1_ref / MILLION, 0.72, 1.28, colors=C_REF,
              lw=1.6, linestyles="--", zorder=6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GEN", "REF"], fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=7)
    ax.set_ylabel("GHG emissions (million kg CO₂e)", fontsize=10.5)
    ax.grid(axis="y", color="#d8d1c4", lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _histogram(ax: plt.Axes, diff_m: np.ndarray, gen_arr: np.ndarray,
               ref_arr: np.ndarray) -> None:
    """Histogram of REF − GEN difference (million kg CO2e)."""
    med   = float(np.median(diff_m))
    p5, p95 = float(np.percentile(diff_m, 5)), float(np.percentile(diff_m, 95))
    p_lt  = float((gen_arr < ref_arr).mean())

    ax.hist(diff_m, bins=65, color=C_HIST, alpha=0.85, edgecolor="none")
    ax.axvline(0.0,  color=C_ZERO, lw=2.0, zorder=5, label="REF = GEN")
    ax.axvline(med,  color="#111111", lw=1.5, linestyle="--", zorder=5, label="Median")

    txt = (f"P(GEN < REF) = {p_lt:.4f}\n"
           f"Median Δ = {med:.2f} M kg CO₂e\n"
           f"90 % CI [{p5:.2f}, {p95:.2f}] M")
    ax.text(0.97, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", lw=0.8))

    ax.set_xlabel("REF − GEN (million kg CO₂e)", fontsize=10.5)
    ax.set_ylabel("Monte Carlo draws", fontsize=10.5)
    ax.grid(axis="y", color="#d8d1c4", lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def build_figure(draws: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        12,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "xtick.labelsize":  10.5,
        "ytick.labelsize":  10.5,
    })

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.subplots_adjust(
        hspace=0.48, wspace=0.32,
        left=0.07, right=0.97, top=0.91, bottom=0.09,
    )

    for col, method in enumerate(METHODS):
        sub = draws[draws["method"] == method]
        gen_m  = sub["GEN_kgCO2e"].to_numpy()  / MILLION
        ref_m  = sub["REF_kgCO2e"].to_numpy()  / MILLION
        diff_m = sub["REF_minus_GEN_kgCO2e"].to_numpy() / MILLION

        _violin_pair(
            axes[0, col], gen_m, ref_m,
            T1[method]["GEN"], T1[method]["REF"],
            METHOD_LABELS[method],
        )
        _histogram(
            axes[1, col], diff_m,
            sub["GEN_kgCO2e"].to_numpy(),
            sub["REF_kgCO2e"].to_numpy(),
        )

    fig.suptitle(
        "Three-method paired Monte Carlo — GEN vs REF "
        f"(N = {N_DRAWS:,}, seed = {SEED})",
        fontsize=13, fontweight="bold", y=0.97,
    )

    stem = OUT_DIR / "F5_paired_monte_carlo_three_method"
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf",          bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {stem}.png")
    print(f"  → {stem}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary_table(summary: pd.DataFrame) -> None:
    print("\n  Summary table (million kg CO₂e):")
    hdr = (f"  {'Method':<16} {'GEN med':>8} {'GEN p5':>8} {'GEN p95':>9}"
           f"  {'REF med':>8} {'REF p5':>8} {'REF p95':>9}"
           f"  {'Δ med':>8} {'Δ p5':>7} {'Δ p95':>8}  {'P(G<R)':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in summary.iterrows():
        s = 1e6
        print(
            f"  {row['method']:<16}"
            f" {row['GEN_median']/s:>8.2f} {row['GEN_p5']/s:>8.2f} {row['GEN_p95']/s:>9.2f}"
            f"  {row['REF_median']/s:>8.2f} {row['REF_p5']/s:>8.2f} {row['REF_p95']/s:>9.2f}"
            f"  {row['diff_median']/s:>8.2f} {row['diff_p5']/s:>7.2f} {row['diff_p95']/s:>8.2f}"
            f"  {row['P_GEN_lt_REF']:>7.4f}"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("3-method paired MC (12 June inventory, calibrated-delta)")
    print(f"N = {N_DRAWS:,}   seed = {SEED}")
    print("=" * 70)

    print("\nStep 1 — sampling parameters and computing draws …")
    draws = run_mc()
    print(f"  {len(draws):,} rows × {len(draws.columns)} columns")

    print("Step 2 — computing summary statistics …")
    summary = build_summary(draws)

    sanity_check(summary)
    _print_summary_table(summary)

    print("\nStep 3 — writing CSVs …")
    draws_path   = OUT_DIR / "mc_draws_three_method.csv"
    summary_path = OUT_DIR / "mc_summary_three_method.csv"
    draws.to_csv(draws_path,   index=False)
    summary.to_csv(summary_path, index=False)
    print(f"  → {draws_path}")
    print(f"  → {summary_path}")

    print("\nStep 4 — building figure …")
    build_figure(draws)

    print("\n" + "=" * 70)
    print("Absolute output paths:")
    for p in [draws_path, summary_path,
              OUT_DIR / "F5_paired_monte_carlo_three_method.png",
              OUT_DIR / "F5_paired_monte_carlo_three_method.pdf"]:
        print(f"  {p.resolve()}")
    print("=" * 70)
    print("\nDone.")
