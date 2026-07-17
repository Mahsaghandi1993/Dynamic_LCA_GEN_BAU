from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        validate_exists,
        write_markdown,
    )
else:
    from .common import WORKSPACE_ROOT, ensure_workspace_tree, validate_exists, write_markdown


STATIC_ROOT = WORKSPACE_ROOT / "export" / "static_lca"
TIME_ROOT = WORKSPACE_ROOT / "export" / "full_dynamic_lcia"
OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "method_comparison"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
QA_ROOT = OUTPUT_ROOT / "qa"
DOCS_ROOT = WORKSPACE_ROOT / "docs"
CASE_DISPLAY_LABELS = {"GEN": "GEN", "BAU": "REF"}

STAGE_FILE_STUBS = {
    "A": "stage_A",
    "B": "stage_B",
    "C": "stage_C",
    "total": "total",
}


def _ensure_output_tree() -> None:
    ensure_workspace_tree()
    for folder in [OUTPUT_ROOT, FIGURE_ROOT, SUMMARY_ROOT, QA_ROOT]:
        folder.mkdir(parents=True, exist_ok=True)


def _archive_legacy_climate_figures() -> None:
    archive_root = FIGURE_ROOT / "archive_legacy"
    archive_root.mkdir(parents=True, exist_ok=True)
    legacy_stems = [
        "stage_A_method_comparison_climate",
        "stage_B_method_comparison_climate",
        "stage_C_method_comparison_climate",
        "total_GEN_method_comparison_climate",
        "total_BAU_method_comparison_climate",
        "total_GEN_vs_BAU_method_comparison_overview",
    ]
    for stem in legacy_stems:
        for suffix in [".png", ".pdf"]:
            source = FIGURE_ROOT / f"{stem}{suffix}"
            target = archive_root / f"{stem}{suffix}"
            if source.exists():
                shutil.move(str(source), str(target))


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save(fig: plt.Figure, basepath: Path) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basepath.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _case_color(case: str) -> str:
    return "#1b9e77" if case == "GEN" else "#d95f02"


def _case_label(case: str) -> str:
    return CASE_DISPLAY_LABELS.get(case, case)


def _stage_label(stage: str) -> str:
    return "Total system" if stage == "total" else f"Stage {stage}"


def _climate_triplet(table: pd.DataFrame, case: str) -> dict[str, float]:
    climate = table.loc[(table["case"] == case) & (table["metric_family"] == "climate")]
    return {
        "conventional_static": float(
            climate.loc[
                (climate["metric_key"] == "climate_static")
                & (climate["method_mode"] == "conventional_static"),
                "score_case_total",
            ].iloc[0]
        ),
        "time_explicit_static": float(
            climate.loc[
                (climate["metric_key"] == "climate_static")
                & (climate["method_mode"] == "time_explicit_static"),
                "score_case_total",
            ].iloc[0]
        ),
        "dynamic_global_warming": float(
            climate.loc[
                (climate["metric_key"] == "dynamic_GWP100")
                & (climate["method_mode"] == "dynamic_climate"),
                "score_case_total",
            ].iloc[0]
        ),
        "dynamic_rf": float(
            climate.loc[climate["metric_key"] == "dynamic_RF", "score_case_total"].iloc[0]
        ),
    }


def _gwi_subtitle() -> str:
    return (
        "Conventional static = fixed 2025 background; time-explicit static = static characterization of the timed inventory; "
        "dynamic climate = climate-only dynamic characterization (dynamic GWP100 basis in exported tables)."
    )


def _load_static(stage: str, case: str) -> pd.DataFrame:
    return pd.read_csv(validate_exists(STATIC_ROOT / f"{STAGE_FILE_STUBS[stage]}_static_{case}.csv", f"static {stage} {case}"))


def _load_time_explicit_climate(stage: str, case: str) -> dict[str, float]:
    stub = STAGE_FILE_STUBS[stage]
    climate_static = pd.read_csv(validate_exists(TIME_ROOT / stub / f"annual_climate_static_{case}.csv", f"time-explicit static climate {stage} {case}"))
    dynamic_ghg = pd.read_csv(validate_exists(TIME_ROOT / stub / f"cumulative_dynamic_GHG_{case}.csv", f"dynamic ghg {stage} {case}"))
    dynamic_rf = pd.read_csv(validate_exists(TIME_ROOT / stub / f"cumulative_dynamic_RF_{case}.csv", f"dynamic rf {stage} {case}"))
    return {
        "time_explicit_static_climate": float(climate_static["annual_time_explicit_static_climate_kgCO2e_case_total"].sum()),
        "dynamic_GWP100": float(dynamic_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "dynamic_RF": float(dynamic_rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1]),
    }


def _load_time_explicit_non_climate(stage: str, case: str) -> pd.DataFrame:
    stub = STAGE_FILE_STUBS[stage]
    return pd.read_csv(validate_exists(TIME_ROOT / stub / f"non_climate_time_explicit_static_scores_{case}.csv", f"time-explicit non-climate {stage} {case}"))


def _build_stage_table(stage: str) -> pd.DataFrame:
    rows = []
    for case in ["GEN", "BAU"]:
        static_df = _load_static(stage, case)
        time_climate = _load_time_explicit_climate(stage, case)
        time_non_climate = _load_time_explicit_non_climate(stage, case)

        climate_static = float(static_df.loc[static_df["category_key"] == "climate_static", "static_score_case_total"].iloc[0])
        rows.extend(
            [
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "climate",
                    "metric_key": "climate_static",
                    "metric_display_name": "Global warming impact",
                    "method_mode": "conventional_static",
                    "unit": "kg CO2e",
                    "score_case_total": climate_static,
                },
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "climate",
                    "metric_key": "climate_static",
                    "metric_display_name": "Global warming impact",
                    "method_mode": "time_explicit_static",
                    "unit": "kg CO2e",
                    "score_case_total": time_climate["time_explicit_static_climate"],
                },
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "climate",
                    "metric_key": "dynamic_GWP100",
                    "metric_display_name": "Global warming impact",
                    "method_mode": "dynamic_climate",
                    "unit": "kg CO2e",
                    "score_case_total": time_climate["dynamic_GWP100"],
                },
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "climate",
                    "metric_key": "dynamic_RF",
                    "metric_display_name": "Radiative forcing",
                    "method_mode": "dynamic_climate",
                    "unit": "W*yr/m^2",
                    "score_case_total": time_climate["dynamic_RF"],
                },
            ]
        )

        static_non_climate = static_df.loc[static_df["category_key"] != "climate_static"].copy()
        compare = static_non_climate.merge(
            time_non_climate[["category_key", "category_display_name", "unit", "static_score_case_total"]],
            on=["category_key", "category_display_name", "unit"],
            how="left",
            suffixes=("_conventional", "_time_explicit"),
        )
        for record in compare.to_dict(orient="records"):
            rows.append(
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "non_climate",
                    "metric_key": record["category_key"],
                    "metric_display_name": record["category_display_name"],
                    "method_mode": "conventional_static",
                    "unit": record["unit"],
                    "score_case_total": float(record["static_score_case_total_conventional"]),
                }
            )
            rows.append(
                {
                    "stage": stage,
                    "case": case,
                    "metric_family": "non_climate",
                    "metric_key": record["category_key"],
                    "metric_display_name": record["category_display_name"],
                    "method_mode": "time_explicit_static",
                    "unit": record["unit"],
                    "score_case_total": float(record["static_score_case_total_time_explicit"]),
                }
            )
    return pd.DataFrame(rows)


def _write_stage_tables(stage_tables: dict[str, pd.DataFrame]) -> None:
    for stage, table in stage_tables.items():
        if stage in {"A", "B", "C"}:
            table.to_csv(OUTPUT_ROOT / f"stage_{stage}_method_comparison.csv", index=False)
        else:
            for case in ["GEN", "BAU"]:
                table.loc[table["case"] == case].to_csv(OUTPUT_ROOT / f"total_method_comparison_{case}.csv", index=False)


def _build_total_gen_vs_bau(total_table: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        total_table.pivot_table(
            index=["metric_family", "metric_key", "metric_display_name", "method_mode", "unit"],
            columns="case",
            values="score_case_total",
            aggfunc="sum",
        )
        .reset_index()
        .fillna(0.0)
    )
    pivot["GEN_minus_BAU"] = pivot["GEN"] - pivot["BAU"]
    pivot["GEN_div_BAU"] = np.where(np.abs(pivot["BAU"]) > 0, pivot["GEN"] / pivot["BAU"], np.nan)
    return pivot


def _climate_difference_summary(stage_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in stage_tables.items():
        for case in ["GEN", "BAU"]:
            subset = table.loc[(table["case"] == case) & (table["metric_family"] == "climate")]
            lookup = (
                subset.assign(key=subset["metric_key"] + "|" + subset["method_mode"])
                .set_index("key")["score_case_total"]
                .to_dict()
            )
            conventional = float(lookup["climate_static|conventional_static"])
            time_explicit = float(lookup["climate_static|time_explicit_static"])
            dynamic_gwp = float(lookup["dynamic_GWP100|dynamic_climate"])
            dynamic_rf = float(lookup["dynamic_RF|dynamic_climate"])
            rows.append(
                {
                    "stage": stage,
                    "case": case,
                    "conventional_static_climate_case_total": conventional,
                    "time_explicit_static_climate_case_total": time_explicit,
                    "dynamic_GWP100_case_total": dynamic_gwp,
                    "dynamic_RF_case_total": dynamic_rf,
                    "time_explicit_minus_conventional_pct": ((time_explicit - conventional) / conventional) if conventional else np.nan,
                    "dynamic_gwp_minus_conventional_pct": ((dynamic_gwp - conventional) / conventional) if conventional else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _non_climate_difference_summary(stage_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in stage_tables.items():
        for case in ["GEN", "BAU"]:
            subset = table.loc[(table["case"] == case) & (table["metric_family"] == "non_climate")]
            for metric_key, group in subset.groupby("metric_key"):
                conventional = float(group.loc[group["method_mode"] == "conventional_static", "score_case_total"].iloc[0])
                time_explicit = float(group.loc[group["method_mode"] == "time_explicit_static", "score_case_total"].iloc[0])
                rows.append(
                    {
                        "stage": stage,
                        "case": case,
                        "metric_key": metric_key,
                        "metric_display_name": str(group["metric_display_name"].iloc[0]),
                        "unit": str(group["unit"].iloc[0]),
                        "conventional_static_case_total": conventional,
                        "time_explicit_static_case_total": time_explicit,
                        "difference_case_total": time_explicit - conventional,
                        "difference_pct_of_conventional": ((time_explicit - conventional) / conventional) if conventional else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _sanity_checks(stage_tables: dict[str, pd.DataFrame], total_compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, table in stage_tables.items():
        rows.append(
            {
                "check": f"{stage} has both GEN and BAU method-comparison rows",
                "value": int(set(table["case"]) == {"GEN", "BAU"}),
                "expected": 1,
                "status": "pass" if set(table["case"]) == {"GEN", "BAU"} else "fail",
            }
        )
        climate_modes = set(table.loc[table["metric_family"] == "climate", "method_mode"])
        rows.append(
            {
                "check": f"{stage} climate includes conventional, time-explicit, and dynamic rows",
                "value": len(climate_modes),
                "expected": 3,
                "status": "pass" if climate_modes == {"conventional_static", "time_explicit_static", "dynamic_climate"} else "fail",
            }
        )
    rows.append(
        {
            "check": "Total GEN vs BAU comparison includes both climate and non-climate families",
            "value": int(set(total_compare["metric_family"]) == {"climate", "non_climate"}),
            "expected": 1,
            "status": "pass" if set(total_compare["metric_family"]) == {"climate", "non_climate"} else "fail",
        }
    )
    return pd.DataFrame(rows)


def _plot_stage_climate(stage: str, table: pd.DataFrame) -> None:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4), width_ratios=[1.45, 1.15, 0.95])
    method_positions = np.arange(3)
    method_labels = ["Conventional\nstatic", "Time-explicit\nstatic", "Dynamic\nclimate"]

    for case in ["GEN", "BAU"]:
        values = _climate_triplet(table, case)
        series = [
            values["conventional_static"],
            values["time_explicit_static"],
            values["dynamic_global_warming"],
        ]
        axes[0].plot(
            method_positions,
            series,
            color=_case_color(case),
            linewidth=2.4,
            marker="o",
            markersize=6,
            label=_case_label(case),
        )
    axes[0].set_xticks(method_positions)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel("Global warming impact (kg CO2e, case total)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_title("Method pathway")
    axes[0].legend(frameon=True, loc="best")

    step_positions = np.arange(2)
    delta_labels = ["Timed foreground +\nfuture backgrounds", "Dynamic climate\ncharacterization"]
    width = 0.36
    for idx, case in enumerate(["GEN", "BAU"]):
        values = _climate_triplet(table, case)
        delta_values = [
            values["time_explicit_static"] - values["conventional_static"],
            values["dynamic_global_warming"] - values["time_explicit_static"],
        ]
        axes[1].bar(
            step_positions + (idx - 0.5) * width,
            delta_values,
            width=width,
            color=_case_color(case),
            alpha=0.9,
            label=_case_label(case) if idx == 0 else None,
        )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0)
    axes[1].set_xticks(step_positions)
    axes[1].set_xticklabels(delta_labels)
    axes[1].set_ylabel("Change in global warming impact (kg CO2e)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_title("Why the methods differ")

    rf_values = [_climate_triplet(table, case)["dynamic_rf"] for case in ["GEN", "BAU"]]
    y_positions = np.arange(2)
    axes[2].axvline(0.0, color="#444444", linewidth=1.0)
    axes[2].hlines(y_positions, 0.0, rf_values, color=[_case_color("GEN"), _case_color("BAU")], linewidth=2.4)
    axes[2].scatter(rf_values, y_positions, color=[_case_color("GEN"), _case_color("BAU")], s=44, zorder=3)
    axes[2].set_yticks(y_positions)
    axes[2].set_yticklabels([_case_label("GEN"), _case_label("BAU")])
    axes[2].set_xlabel("Dynamic radiative forcing (W*yr/m^2)")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].set_title("Dynamic RF endpoint")

    fig.suptitle(f"{_stage_label(stage)} global warming method comparison")
    fig.text(0.5, 0.955, _gwi_subtitle(), ha="center", va="top", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = f"stage_{stage}_global_warming_method_comparison" if stage != "total" else "total_GEN_vs_BAU_global_warming_method_overview"
    _save(fig, FIGURE_ROOT / stem)


def _plot_total_case_climate(case: str, total_table: pd.DataFrame) -> None:
    _set_style()
    values = _climate_triplet(total_table, case)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.1), width_ratios=[1.15, 1.05, 0.8])

    methods = ["Conventional\nstatic", "Time-explicit\nstatic", "Dynamic\nclimate"]
    series = [
        values["conventional_static"],
        values["time_explicit_static"],
        values["dynamic_global_warming"],
    ]
    axes[0].plot(
        np.arange(3),
        series,
        color=_case_color(case),
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel("Global warming impact (kg CO2e, case total)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_title("Method pathway")

    axes[1].bar(
        ["Timed foreground +\nfuture backgrounds", "Dynamic climate\ncharacterization"],
        [
            values["time_explicit_static"] - values["conventional_static"],
            values["dynamic_global_warming"] - values["time_explicit_static"],
        ],
        color=_case_color(case),
    )
    axes[1].axhline(0.0, color="#444444", linewidth=1.0)
    axes[1].set_ylabel("Change in global warming impact (kg CO2e)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_title("Method delta decomposition")

    axes[2].hlines([0], 0.0, [values["dynamic_rf"]], color=_case_color(case), linewidth=2.4)
    axes[2].scatter([values["dynamic_rf"]], [0], color=_case_color(case), s=46, zorder=3)
    axes[2].axvline(0.0, color="#444444", linewidth=1.0)
    axes[2].set_yticks([0])
    axes[2].set_yticklabels([_case_label(case)])
    axes[2].set_xlabel("Dynamic radiative forcing (W*yr/m^2)")
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[2].set_title("Dynamic RF endpoint")

    fig.suptitle(f"Total {_case_label(case)} global warming method comparison")
    fig.text(0.5, 0.955, _gwi_subtitle(), ha="center", va="top", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, FIGURE_ROOT / f"total_{case}_global_warming_method_comparison")


def _plot_total_case_non_climate(case: str, total_table: pd.DataFrame) -> None:
    _set_style()
    data = total_table.loc[(total_table["case"] == case) & (total_table["metric_family"] == "non_climate")].copy()
    categories = list(dict.fromkeys(data["metric_display_name"]))
    n = len(categories)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14.5, 4.0 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, category in zip(axes, categories):
        subset = data.loc[data["metric_display_name"] == category]
        unit = str(subset["unit"].iloc[0])
        conv = float(subset.loc[subset["method_mode"] == "conventional_static", "score_case_total"].iloc[0])
        time_explicit = float(subset.loc[subset["method_mode"] == "time_explicit_static", "score_case_total"].iloc[0])
        ax.bar(["Conv. static", "Time-explicit static"], [conv, time_explicit], color=["#9ecae1", "#3182bd"], width=0.62)
        ax.set_title(f"{category}\n[{unit}]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Case total")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Total {_case_label(case)} non-climate time-explicit static method comparison")
    fig.text(
        0.5,
        0.955,
        "Non-climate categories are static characterization only: conventional static versus static characterization of the timed inventory.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, FIGURE_ROOT / f"total_{case}_method_comparison_non_climate")


def _load_dynamic_top_flows(case: str, stem: str, value_col: str) -> pd.DataFrame:
    table = pd.read_csv(
        validate_exists(
            TIME_ROOT / "total" / f"top_biosphere_flows_{stem}_{case}.csv",
            f"dynamic top {stem} flows {case}",
        )
    )
    grouped = (
        table.groupby("flow_label", as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
        .assign(abs_value=lambda df: df["value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(6)
        .sort_values("value")
    )
    return grouped


def _plot_total_case_dynamic_rf(case: str) -> None:
    _set_style()
    table = pd.read_csv(
        validate_exists(TIME_ROOT / "total" / f"cumulative_dynamic_RF_{case}.csv", f"total dynamic RF {case}")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), width_ratios=[1.35, 0.9])
    axes[0].plot(
        table["year"],
        table["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"],
        color=_case_color(case),
        linewidth=2.4,
        marker="o",
        markersize=3.0,
    )
    axes[0].set_xlabel("Calendar year")
    axes[0].set_ylabel("Cumulative radiative forcing (W*yr/m^2, case total)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_title("Cumulative RF trajectory")

    annual = table.loc[
        :,
        ["year", "annual_radiative_forcing_W_per_m2_case_total"],
    ].copy()
    annual = annual.loc[annual["annual_radiative_forcing_W_per_m2_case_total"] != 0.0]
    if annual.empty:
        annual = table.loc[:, ["year", "annual_radiative_forcing_W_per_m2_case_total"]].copy()
    axes[1].vlines(
        annual["year"],
        0.0,
        annual["annual_radiative_forcing_W_per_m2_case_total"],
        color=_case_color(case),
        linewidth=1.2,
        alpha=0.65,
    )
    axes[1].scatter(
        annual["year"],
        annual["annual_radiative_forcing_W_per_m2_case_total"],
        color=_case_color(case),
        s=16,
        zorder=3,
    )
    axes[1].set_xlabel("Calendar year")
    axes[1].set_ylabel("Annual radiative forcing (W/m^2, case total)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_title("Annual RF pulses")

    fig.suptitle(f"Total {_case_label(case)} dynamic radiative forcing comparison")
    fig.text(
        0.5,
        0.955,
        "Cumulative RF is the preferred interpretation view; annual RF pulses are retained as a diagnostic of timed events.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, FIGURE_ROOT / f"total_{case}_dynamic_RF_comparison")


def _plot_total_case_dynamic_top_flows(case: str) -> None:
    _set_style()
    ghg = _load_dynamic_top_flows(case, "GHG", "dynamic_GWP100_kgCO2e_case_total")
    rf = _load_dynamic_top_flows(case, "RF", "radiative_forcing_W_per_m2_case_total")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), width_ratios=[1.1, 1.0])

    axes[0].barh(ghg["flow_label"], ghg["value"], color=_case_color(case), alpha=0.9)
    axes[0].set_xlabel("Cumulative global warming impact (kg CO2e, case total)")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set_title("Top dynamic climate flows")

    axes[1].barh(rf["flow_label"], rf["value"], color=_case_color(case), alpha=0.9)
    axes[1].set_xlabel("Integrated RF contribution (W*yr/m^2, case total)")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].set_title("Top dynamic RF flows")

    fig.suptitle(f"Total {_case_label(case)} dynamic top climate flows")
    fig.text(
        0.5,
        0.955,
        "Aggregated flow totals replace the noisier year-by-year diagnostics for manuscript interpretation.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, FIGURE_ROOT / f"total_{case}_dynamic_top_climate_flows")


def _method_comparison_reaudit(stage_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for stage, table in stage_tables.items():
        for case in ["GEN", "BAU"]:
            values = _climate_triplet(table, case)
            timing_background_delta = values["time_explicit_static"] - values["conventional_static"]
            dynamic_characterization_delta = values["dynamic_global_warming"] - values["time_explicit_static"]
            if stage == "A":
                interpretation = "Year-0 pulse; conventional static and time-explicit static should match closely."
                figure_action = "replace_grouped_bar_with_method_pathway"
            elif stage == "B":
                interpretation = "Spread operational and replacement events; future background relinking lowers time-explicit static relative to fixed-year static."
                figure_action = "replace_grouped_bar_with_method_pathway_and_deltas"
            elif stage == "C":
                interpretation = "Late-life end-of-life pulse; dynamic climate can shift sign or magnitude relative to static because the pulse occurs near the end of the analysis horizon."
                figure_action = "replace_grouped_bar_with_signed_method_pathway"
            else:
                interpretation = "Total system aggregates Stage A upfront burdens, Stage B timed operations, and Stage C terminal events."
                figure_action = "replace_grouped_bar_with_overview_pathway"
            rows.append(
                {
                    "stage": stage,
                    "case": case,
                    "conventional_static_global_warming_kgCO2e": values["conventional_static"],
                    "time_explicit_static_global_warming_kgCO2e": values["time_explicit_static"],
                    "dynamic_global_warming_kgCO2e": values["dynamic_global_warming"],
                    "dynamic_radiative_forcing_Wyr_per_m2": values["dynamic_rf"],
                    "timing_background_delta_kgCO2e": timing_background_delta,
                    "dynamic_characterization_delta_kgCO2e": dynamic_characterization_delta,
                    "time_explicit_vs_conventional_ratio": values["time_explicit_static"] / values["conventional_static"] if values["conventional_static"] else np.nan,
                    "dynamic_vs_time_explicit_ratio": values["dynamic_global_warming"] / values["time_explicit_static"] if values["time_explicit_static"] else np.nan,
                    "scientifically_usable": "yes",
                    "interpretation_note": interpretation,
                    "figure_action": figure_action,
                }
            )
    return pd.DataFrame(rows)


def _write_qa_notes(reaudit: pd.DataFrame) -> None:
    stage_c = reaudit.loc[reaudit["stage"] == "C", ["case", "timing_background_delta_kgCO2e", "dynamic_characterization_delta_kgCO2e"]]
    total = reaudit.loc[reaudit["stage"] == "total", ["case", "conventional_static_global_warming_kgCO2e", "time_explicit_static_global_warming_kgCO2e", "dynamic_global_warming_kgCO2e"]]
    redesign_note = (
        "The original climate method-comparison figures used grouped bars for conventional static, time-explicit static, "
        "and dynamic climate results. Those bars were numerically correct but weak as scientific communication because they "
        "compressed three distinct methodological steps into one visual grammar.\n\n"
        "The revised figures therefore use:\n"
        "- a method pathway panel to show the ordered progression from conventional static to time-explicit static to dynamic climate,\n"
        "- a delta panel to separate timing/prospective-background effects from dynamic-characterization effects, and\n"
        "- a dedicated dynamic RF endpoint panel so RF is no longer visually hidden behind the global-warming bars.\n\n"
        "Stage C needed the largest redesign because signed late-life results are particularly misleading in a grouped-bar format. "
        f"Current Stage C deltas are {stage_c.to_dict(orient='records')}.\n\n"
        "The revised total-case figures also add aggregated top-flow views and case-specific RF figures so the accepted dynamic "
        "results can be interpreted without relying on the noisier year-by-year contributor traces."
    )
    write_markdown(QA_ROOT / "method_comparison_figure_redesign_note.md", redesign_note)


def _write_summary(climate_diff: pd.DataFrame, non_climate_diff: pd.DataFrame) -> None:
    total_gen = climate_diff.loc[(climate_diff["stage"] == "total") & (climate_diff["case"] == "GEN")].iloc[0]
    total_bau = climate_diff.loc[(climate_diff["stage"] == "total") & (climate_diff["case"] == "BAU")].iloc[0]
    max_non_climate = non_climate_diff.loc[non_climate_diff["stage"] == "total"].assign(abs_pct=lambda df: df["difference_pct_of_conventional"].abs()).sort_values("abs_pct", ascending=False).iloc[0]
    text = (
        "Conventional static LCA in this study means a fixed 2025 background, no timed foreground exchanges, "
        "and the same synchronized HEATNETS-authoritative denominator used in the time-explicit workflow. "
        "Time-explicit static LCIA keeps the timed foreground and prospective relinking through bw_timex, but "
        "still applies static characterization factors. Dynamic climate LCIA then adds dynamic characterization "
        "for climate only, reported here as global warming impact on a dynamic GWP100 basis plus cumulative radiative forcing.\n\n"
        f"For the total system, GEN changes from `{total_gen['conventional_static_climate_case_total']:,.3f} kg CO2e` "
        f"under conventional static climate to `{total_gen['dynamic_GWP100_case_total']:,.3f} kg CO2e` under dynamic climate, "
        f"while REF changes from `{total_bau['conventional_static_climate_case_total']:,.3f} kg CO2e` to "
        f"`{total_bau['dynamic_GWP100_case_total']:,.3f} kg CO2e`. The method comparison matters because the geothermal "
        "and REF systems differ strongly in when burdens occur: GEN has higher early construction pulses, while REF "
        "accumulates more operational climate burden over time. The largest total non-climate conventional-versus-time-explicit "
        f"difference occurs for `{max_non_climate['metric_display_name']}` in `{max_non_climate['case']}`, at "
        f"`{max_non_climate['difference_pct_of_conventional']:.2%}` relative to the conventional static result."
    )
    write_markdown(SUMMARY_ROOT / "method_comparison_overview.md", text)


def run() -> dict[str, Path]:
    _ensure_output_tree()
    _archive_legacy_climate_figures()

    stage_tables = {stage: _build_stage_table(stage) for stage in ["A", "B", "C", "total"]}
    _write_stage_tables(stage_tables)

    total_compare = _build_total_gen_vs_bau(stage_tables["total"])
    total_compare.to_csv(OUTPUT_ROOT / "total_method_comparison_GEN_vs_BAU.csv", index=False)

    climate_diff = _climate_difference_summary(stage_tables)
    climate_diff.to_csv(OUTPUT_ROOT / "climate_method_difference_summary.csv", index=False)

    non_climate_diff = _non_climate_difference_summary(stage_tables)
    non_climate_diff.to_csv(OUTPUT_ROOT / "non_climate_method_difference_summary.csv", index=False)

    sanity = _sanity_checks(stage_tables, total_compare)
    sanity.to_csv(OUTPUT_ROOT / "method_comparison_sanity_checks.csv", index=False)
    reaudit = _method_comparison_reaudit(stage_tables)
    reaudit.to_csv(QA_ROOT / "method_comparison_reaudit.csv", index=False)
    _write_qa_notes(reaudit)

    for stage in ["A", "B", "C"]:
        _plot_stage_climate(stage, stage_tables[stage])
    _plot_stage_climate("total", stage_tables["total"])
    _plot_total_case_climate("GEN", stage_tables["total"])
    _plot_total_case_climate("BAU", stage_tables["total"])
    _plot_total_case_non_climate("GEN", stage_tables["total"])
    _plot_total_case_non_climate("BAU", stage_tables["total"])
    _plot_total_case_dynamic_rf("GEN")
    _plot_total_case_dynamic_rf("BAU")
    _plot_total_case_dynamic_top_flows("GEN")
    _plot_total_case_dynamic_top_flows("BAU")

    _write_summary(climate_diff, non_climate_diff)

    return {
        "stage_A_method_comparison": OUTPUT_ROOT / "stage_A_method_comparison.csv",
        "stage_B_method_comparison": OUTPUT_ROOT / "stage_B_method_comparison.csv",
        "stage_C_method_comparison": OUTPUT_ROOT / "stage_C_method_comparison.csv",
        "total_method_comparison_GEN": OUTPUT_ROOT / "total_method_comparison_GEN.csv",
        "total_method_comparison_BAU": OUTPUT_ROOT / "total_method_comparison_BAU.csv",
        "total_method_comparison_GEN_vs_BAU": OUTPUT_ROOT / "total_method_comparison_GEN_vs_BAU.csv",
        "climate_method_difference_summary": OUTPUT_ROOT / "climate_method_difference_summary.csv",
        "non_climate_method_difference_summary": OUTPUT_ROOT / "non_climate_method_difference_summary.csv",
        "method_comparison_sanity_checks": OUTPUT_ROOT / "method_comparison_sanity_checks.csv",
        "method_comparison_reaudit": QA_ROOT / "method_comparison_reaudit.csv",
        "method_comparison_figure_redesign_note": QA_ROOT / "method_comparison_figure_redesign_note.md",
        "method_comparison_summary": SUMMARY_ROOT / "method_comparison_overview.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare conventional static, time-explicit static, and dynamic climate results for the synchronized Framingham workflow.")
    parser.parse_args()
    outputs = run()
    print("Method comparison outputs:")
    for key, path in outputs.items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
