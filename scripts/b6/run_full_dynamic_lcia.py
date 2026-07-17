from __future__ import annotations

import argparse
import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[3] / "Dynamic_LCA_GEN_BAU" / ".mplconfig").resolve()),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str((Path(__file__).resolve().parents[3] / "Dynamic_LCA_GEN_BAU" / ".cache").resolve()),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.build_full_stage_timex_wrappers import (  # type: ignore
        build as build_stage_wrappers,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
        write_markdown,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_b6_dynamic_lcia import (  # type: ignore
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _annual_impacts_from_method,
        _build_annual_ghg_table,
        _build_annual_rf_table,
        _build_characterization_functions,
        _build_database_dates,
        _choose_climate_method,
        _comparison_tables,
        _install_safe_dynamic_biosphere_builder,
        _install_safe_interdatabase_mapping,
        _install_safe_temporalis_init,
        _patch_temporal_market_shares,
        _plot_compare_lines,
        _plot_stacked_bars,
        _plot_top_flow_lines,
        _require_ready,
        _save_figure,
        _set_plot_style,
        _title_suffix,
        _top_flow_table,
        _top_source_table,
    )
else:
    from .build_full_stage_timex_wrappers import build as build_stage_wrappers
    from .common import (
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
        write_markdown,
    )
    from .run_b6_dynamic_lcia import (
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _annual_impacts_from_method,
        _build_annual_ghg_table,
        _build_annual_rf_table,
        _build_characterization_functions,
        _build_database_dates,
        _choose_climate_method,
        _comparison_tables,
        _install_safe_dynamic_biosphere_builder,
        _install_safe_interdatabase_mapping,
        _install_safe_temporalis_init,
        _patch_temporal_market_shares,
        _plot_compare_lines,
        _plot_stacked_bars,
        _plot_top_flow_lines,
        _require_ready,
        _save_figure,
        _set_plot_style,
        _title_suffix,
        _top_flow_table,
        _top_source_table,
    )


OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "full_dynamic_lcia"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
SUMMARY_ROOT = OUTPUT_ROOT / "summaries"
QA_ROOT = OUTPUT_ROOT / "qa"
CASE_DISPLAY_LABELS = {"GEN": "GEN", "BAU": "REF"}
ALL_CASES = ("GEN", "BAU")
ALL_STAGES = ("A", "B", "C", "total")
STAGE_ROOTS = {
    "A": OUTPUT_ROOT / "stage_A",
    "B": OUTPUT_ROOT / "stage_B",
    "C": OUTPUT_ROOT / "stage_C",
    "total": OUTPUT_ROOT / "total",
}


def _case_label(case: str) -> str:
    return CASE_DISPLAY_LABELS.get(case, case)
STAGE_MANIFEST_PATH = QA_ROOT / "stage_wrapper_manifest.json"
B6_MANIFEST_PATH = WORKSPACE_ROOT / "export" / "b6" / "bw_timex_inputs" / "b6_inventory_manifest.json"
B6_DYNAMIC_MANIFEST_PATH = WORKSPACE_ROOT / "export" / "b6_dynamic_lcia" / "b6_dynamic_lcia_manifest.json"
B6_DYNAMIC_SUMMARY_PATH = WORKSPACE_ROOT / "export" / "b6_dynamic_lcia" / "denominator_and_method_summary.csv"


NON_CLIMATE_METHOD_SPECS: dict[str, dict[str, Any]] = {
    "acidification": {
        "method": (
            "TRACI v2.1 no LT",
            "acidification no LT",
            "acidification potential (AP) no LT",
        ),
        "family": "TRACI v2.1 no LT",
        "display_name": "Acidification",
        "unit": "kg SO2 eq",
    },
    "eutrophication": {
        "method": (
            "TRACI v2.1 no LT",
            "eutrophication no LT",
            "eutrophication potential no LT",
        ),
        "family": "TRACI v2.1 no LT",
        "display_name": "Eutrophication",
        "unit": "kg N eq",
    },
    "ozone_depletion": {
        "method": (
            "TRACI v2.1 no LT",
            "ozone depletion no LT",
            "ozone depletion potential (ODP) no LT",
        ),
        "family": "TRACI v2.1 no LT",
        "display_name": "Ozone Depletion",
        "unit": "kg CFC-11 eq",
    },
    "particulate_matter": {
        "method": (
            "TRACI v2.1 no LT",
            "particulate matter formation no LT",
            "particulate matter formation potential (PMFP) no LT",
        ),
        "family": "TRACI v2.1 no LT",
        "display_name": "Particulate Matter Formation",
        "unit": "kg PM2.5 eq",
    },
    "photochemical_oxidant_formation": {
        "method": (
            "TRACI v2.1 no LT",
            "photochemical oxidant formation no LT",
            "maximum incremental reactivity (MIR) no LT",
        ),
        "family": "TRACI v2.1 no LT",
        "display_name": "Photochemical Oxidant Formation",
        "unit": "kg O3 eq",
    },
    "fossil_energy_demand": {
        "method": (
            "Cumulative Energy Demand (CED)",
            "energy resources: non-renewable, fossil",
            "energy content (HHV)",
        ),
        "family": "Cumulative Energy Demand (CED)",
        "display_name": "Fossil Energy Demand",
        "unit": "MJ",
        "supplemental": True,
    },
}


@dataclass(frozen=True)
class FullContext:
    project: str
    normalization_mode: str
    foreground_set: str
    eol_scenario: str
    denominator_lifetime_kwh_th: float
    denominator_annual_kwh_th: float
    service_life_years: int
    background_dbs: list[str]
    wrappers: dict[str, dict[str, tuple[str, str]]]
    total_wrappers: dict[str, tuple[str, str]]
    climate_method: tuple[str, ...]


@dataclass
class FullCaseResult:
    case: str
    stage: str
    fg_db: str
    wrapper_key: tuple[str, str]
    tlca: Any
    inv_df: pd.DataFrame
    climate_static_annual: pd.DataFrame
    annual_ghg: pd.DataFrame
    annual_rf: pd.DataFrame
    top_gwp_flows: pd.DataFrame
    top_rf_flows: pd.DataFrame
    top_gwp_sources: pd.DataFrame
    climate_summary: pd.DataFrame
    non_climate_scores: pd.DataFrame
    non_climate_annual: pd.DataFrame


def _ensure_output_tree() -> None:
    ensure_workspace_tree()
    for folder in [OUTPUT_ROOT, FIGURE_ROOT, SUMMARY_ROOT, QA_ROOT, *STAGE_ROOTS.values()]:
        folder.mkdir(parents=True, exist_ok=True)


def _remove_stale_outputs() -> None:
    # Clean out superseded figure names so reruns don't leave behind ambiguous artifacts.
    stale_figure_stems = [
        "stage_A_non_climate_static_GEN_vs_BAU",
        "stage_B_non_climate_static_GEN_vs_BAU",
        "stage_C_non_climate_static_GEN_vs_BAU",
        "total_non_climate_static_GEN_vs_BAU",
    ]
    for stem in stale_figure_stems:
        for suffix in [".png", ".pdf"]:
            path = FIGURE_ROOT / f"{stem}{suffix}"
            if path.exists():
                path.unlink()


def _load_context() -> FullContext:
    config = load_b6_case_config()
    b6_manifest = read_json(B6_MANIFEST_PATH)
    stage_manifest = read_json(STAGE_MANIFEST_PATH)
    b6_dynamic_summary = pd.read_csv(validate_exists(B6_DYNAMIC_SUMMARY_PATH, "B6 dynamic summary"))

    normalization_mode = str(
        b6_manifest.get("selected_normalization_mode", b6_manifest.get("normalization_mode", ""))
    )
    foreground_set = str(
        b6_manifest.get("selected_foreground_set", b6_manifest.get("foreground_set", ""))
    )
    if normalization_mode != "heatnets_case_service_loads":
        raise RuntimeError(
            f"Full LCIA must use `heatnets_case_service_loads`, but manifest selected `{normalization_mode}`."
        )
    if foreground_set != "heatnets_authoritative_sync":
        raise RuntimeError(
            f"Full LCIA must use `heatnets_authoritative_sync`, but manifest selected `{foreground_set}`."
        )

    def metric_row(key: str) -> str:
        row = b6_dynamic_summary.loc[b6_dynamic_summary["key"] == key]
        if row.empty:
            raise RuntimeError(f"Could not find `{key}` in {relpath(B6_DYNAMIC_SUMMARY_PATH)}.")
        return str(row.iloc[0]["value"])

    def as_key(payload: Any) -> tuple[str, str]:
        if not isinstance(payload, list) or len(payload) != 2:
            raise RuntimeError(f"Expected two-item activity key, got {payload!r}")
        return (str(payload[0]), str(payload[1]))

    try:
        climate_method = tuple(metric_row("static_climate_method").split(" | "))
    except Exception as exc:
        raise RuntimeError(
            "Could not recover the static climate method from the accepted B6 dynamic summary."
        ) from exc

    wrappers = {
        system: {stage: as_key(key) for stage, key in stage_map.items()}
        for system, stage_map in stage_manifest["stage_wrapper_keys"].items()
    }
    total_wrappers = {
        "GEN": as_key(b6_manifest["gen_total_with_b6_key"]),
        "BAU": as_key(b6_manifest["bau_total_with_b6_key"]),
    }

    return FullContext(
        project=str(config["project"]["brightway_project"]),
        normalization_mode=normalization_mode,
        foreground_set=foreground_set,
        eol_scenario=str(config.get("end_of_life", {}).get("default_scenario", "unspecified")),
        denominator_lifetime_kwh_th=float(metric_row("denominator_lifetime_kwh_th")),
        denominator_annual_kwh_th=float(metric_row("denominator_annual_kwh_th")),
        service_life_years=int(float(metric_row("service_life_years"))),
        background_dbs=[str(db) for db in config["project"]["background_dbs"]],
        wrappers=wrappers,
        total_wrappers=total_wrappers,
        climate_method=climate_method,
    )


def _validate_methods() -> pd.DataFrame:
    import bw2data as bd

    rows: list[dict[str, Any]] = []
    climate_method = _choose_climate_method(bd)
    climate_cf_count = len(list(bd.Method(climate_method).load()))
    rows.append(
        {
            "method_group": "climate_static",
            "category_key": "climate_static",
            "method_tuple": " | ".join(climate_method),
            "family": climate_method[0],
            "cf_count": climate_cf_count,
            "status": "pass",
        }
    )
    rows.extend(
        [
            {
                "method_group": "dynamic_climate",
                "category_key": "dynamic_GWP100",
                "method_tuple": "dynamic_characterization | IPCC AR6 | GWP100",
                "family": "dynamic_characterization",
                "cf_count": np.nan,
                "status": "pass",
                "note": "dynamic climate characterization functions loaded in-project",
            },
            {
                "method_group": "dynamic_climate",
                "category_key": "dynamic_RF",
                "method_tuple": "dynamic_characterization | IPCC AR6 | radiative_forcing",
                "family": "dynamic_characterization",
                "cf_count": np.nan,
                "status": "pass",
                "note": "dynamic climate characterization functions loaded in-project",
            },
        ]
    )

    for category_key, spec in NON_CLIMATE_METHOD_SPECS.items():
        method = tuple(spec["method"])
        status = "pass"
        cf_count = 0
        note = ""
        try:
            if method not in bd.methods:
                raise KeyError("method tuple not installed")
            cf_count = len(list(bd.Method(method).load()))
            if cf_count == 0:
                raise ValueError("method has zero CFs")
        except Exception as exc:
            status = "fail"
            note = str(exc)
        rows.append(
            {
                "method_group": "non_climate_static",
                "category_key": category_key,
                "method_tuple": " | ".join(method),
                "family": spec["family"],
                "cf_count": cf_count,
                "status": status,
                "note": note,
            }
        )
    out = pd.DataFrame(rows)
    failures = out.loc[out["status"] != "pass"]
    if not failures.empty:
        failed = failures[["category_key", "method_tuple"]].to_dict(orient="records")
        raise RuntimeError(f"Required LCIA methods are unavailable: {failed}")
    return out


def _run_case(
    case: str,
    stage: str,
    fg_db: str,
    wrapper_key: tuple[str, str],
    context: FullContext,
    t0: dt.datetime,
    horizon_years: int,
) -> FullCaseResult:
    import bw2data as bd
    from bw_timex import TimexLCA
    import dynamic_characterization.dynamic_characterization as dc

    bio_db = "biosphere3" if "biosphere3" in bd.databases else "biosphere"
    _install_safe_temporalis_init()
    tlca = TimexLCA(
        {wrapper_key: 1.0},
        context.climate_method,
        _build_database_dates(fg_db, context.background_dbs, bio_db),
    )
    _install_safe_interdatabase_mapping(tlca)
    _install_safe_dynamic_biosphere_builder()
    tlca.build_timeline(
        starting_datetime=t0.strftime("%Y-%m-%d"),
        temporal_grouping="year",
        interpolation_type="linear",
        cutoff=1e-12,
        max_calc=400000,
    )
    _patch_temporal_market_shares(tlca, bd=bd, bg_dbs=context.background_dbs)

    tlca.lci()
    tlca.static_lcia()

    inv_df = tlca.dynamic_inventory_df.copy()
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    climate_static_annual = _annual_impacts_from_method(inv_df, method=context.climate_method, bd=bd)

    characterization_functions = _build_characterization_functions(inv_df, bd=bd)
    tlca.dynamic_lcia(
        metric="GWP",
        time_horizon=horizon_years,
        characterization_functions=characterization_functions,
    )

    gwp_df = dc.characterize(
        dynamic_inventory_df=inv_df[["date", "flow", "activity", "amount"]].copy(),
        metric="GWP",
        characterization_functions=characterization_functions,
        time_horizon=horizon_years,
        fixed_time_horizon=True,
        time_horizon_start=t0,
    )
    rf_df = dc.characterize(
        dynamic_inventory_df=inv_df[["date", "flow", "activity", "amount"]].copy(),
        metric="radiative_forcing",
        characterization_functions=characterization_functions,
        time_horizon=horizon_years,
        fixed_time_horizon=True,
        time_horizon_start=t0,
    )

    for frame in [gwp_df, rf_df]:
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year

    annual_gwp = gwp_df.groupby("year", as_index=False)["amount"].sum().sort_values("year")
    annual_rf = rf_df.groupby("year", as_index=False)["amount"].sum().sort_values("year")

    annual_ghg = _build_annual_ghg_table(
        static_annual=climate_static_annual,
        dynamic_annual=annual_gwp,
        denominator=context.denominator_lifetime_kwh_th,
        start_year=t0.year,
        horizon_years=horizon_years,
        normalization_mode=context.normalization_mode,
        foreground_set=context.foreground_set,
    )
    annual_rf_table = _build_annual_rf_table(
        annual_rf=annual_rf,
        denominator=context.denominator_lifetime_kwh_th,
        start_year=t0.year,
        horizon_years=horizon_years,
        normalization_mode=context.normalization_mode,
        foreground_set=context.foreground_set,
    )

    top_gwp_flows = _top_flow_table(
        characterized_df=gwp_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="dynamic_GWP100_kgCO2e",
    )
    top_rf_flows = _top_flow_table(
        characterized_df=rf_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="radiative_forcing_W_per_m2",
    )
    top_gwp_sources = _top_source_table(
        characterized_df=gwp_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="dynamic_GWP100_kgCO2e",
    )

    non_climate_scores = []
    non_climate_annual_frames = []
    for category_key, spec in NON_CLIMATE_METHOD_SPECS.items():
        method = tuple(spec["method"])
        annual = _annual_impacts_from_method(inv_df, method=method, bd=bd).rename(
            columns={"impact": "annual_static_score_per_fu"}
        )
        annual["annual_static_score_case_total"] = (
            annual["annual_static_score_per_fu"] * context.denominator_lifetime_kwh_th
        )
        annual["case"] = case
        annual["stage"] = stage
        annual["category_key"] = category_key
        annual["category_display_name"] = spec["display_name"]
        annual["unit"] = spec["unit"]
        annual["method_family"] = spec["family"]
        annual["method_tuple"] = " | ".join(method)
        non_climate_annual_frames.append(annual)

        non_climate_scores.append(
            {
                "case": case,
                "stage": stage,
                "category_key": category_key,
                "category_display_name": spec["display_name"],
                "method_family": spec["family"],
                "method_tuple": " | ".join(method),
                "unit": spec["unit"],
                "supplemental_metric": bool(spec.get("supplemental", False)),
                "static_score_per_fu": float(annual["annual_static_score_per_fu"].sum()),
                "static_score_case_total": float(annual["annual_static_score_case_total"].sum()),
            }
        )

    climate_summary = pd.DataFrame(
        [
            {
                "case": case,
                "stage": stage,
                "project": context.project,
                "fg_db": fg_db,
                "wrapper_key": str(wrapper_key),
                "normalization_mode": context.normalization_mode,
                "foreground_set": context.foreground_set,
                "denominator_lifetime_kwh_th": context.denominator_lifetime_kwh_th,
                "base_score_static_climate_per_fu": float(tlca.base_score),
                "static_score_timex_climate_per_fu": float(tlca.static_score),
                "dynamic_score_gwp_per_fu": float(tlca.dynamic_score),
                "dynamic_GWP100_total_kgCO2e_per_fu": float(
                    annual_ghg["annual_dynamic_GWP100_kgCO2e_per_fu"].sum()
                ),
                "dynamic_GWP100_total_kgCO2e_case_total": float(
                    annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()
                ),
                "final_cumulative_rf_Wyr_per_m2_per_fu": float(
                    annual_rf_table["cumulative_discrete_annual_rf_Wyr_per_m2_per_fu"].iloc[-1]
                ),
                "final_cumulative_rf_Wyr_per_m2_case_total": float(
                    annual_rf_table["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1]
                ),
                "time_horizon_years": horizon_years,
                "time_horizon_start": t0.strftime("%Y-%m-%d"),
            }
        ]
    )

    return FullCaseResult(
        case=case,
        stage=stage,
        fg_db=fg_db,
        wrapper_key=wrapper_key,
        tlca=tlca,
        inv_df=inv_df,
        climate_static_annual=climate_static_annual,
        annual_ghg=annual_ghg,
        annual_rf=annual_rf_table,
        top_gwp_flows=top_gwp_flows,
        top_rf_flows=top_rf_flows,
        top_gwp_sources=top_gwp_sources,
        climate_summary=climate_summary,
        non_climate_scores=pd.DataFrame(non_climate_scores),
        non_climate_annual=pd.concat(non_climate_annual_frames, ignore_index=True),
    )


def _write_stage_case_exports(result: FullCaseResult) -> dict[str, Path]:
    stage_root = STAGE_ROOTS[result.stage]
    case = result.case
    outputs = {
        "annual_climate_static": stage_root / f"annual_climate_static_{case}.csv",
        "cumulative_dynamic_ghg": stage_root / f"cumulative_dynamic_GHG_{case}.csv",
        "cumulative_dynamic_rf": stage_root / f"cumulative_dynamic_RF_{case}.csv",
        "non_climate": stage_root / f"non_climate_time_explicit_static_scores_{case}.csv",
        "non_climate_legacy": stage_root / f"non_climate_static_scores_{case}.csv",
        "top_ghg_flows": stage_root / f"top_biosphere_flows_GHG_{case}.csv",
        "top_rf_flows": stage_root / f"top_biosphere_flows_RF_{case}.csv",
    }
    result.annual_ghg[
        [
            "year",
            "annual_time_explicit_static_climate_kgCO2e_per_fu",
            "annual_time_explicit_static_climate_kgCO2e_case_total",
            "cumulative_time_explicit_static_climate_kgCO2e_per_fu",
            "cumulative_time_explicit_static_climate_kgCO2e_case_total",
            "normalization_mode",
            "foreground_set",
            "denominator_lifetime_kwh_th",
        ]
    ].to_csv(outputs["annual_climate_static"], index=False)
    result.annual_ghg[
        [
            "year",
            "annual_dynamic_GWP100_kgCO2e_per_fu",
            "annual_dynamic_GWP100_kgCO2e_case_total",
            "cumulative_dynamic_GWP100_kgCO2e_per_fu",
            "cumulative_dynamic_GWP100_kgCO2e_case_total",
            "normalization_mode",
            "foreground_set",
            "denominator_lifetime_kwh_th",
        ]
    ].to_csv(outputs["cumulative_dynamic_ghg"], index=False)
    result.annual_rf.to_csv(outputs["cumulative_dynamic_rf"], index=False)
    result.non_climate_scores.to_csv(outputs["non_climate"], index=False)
    result.non_climate_scores.to_csv(outputs["non_climate_legacy"], index=False)
    result.top_gwp_flows.to_csv(outputs["top_ghg_flows"], index=False)
    result.top_rf_flows.to_csv(outputs["top_rf_flows"], index=False)
    return outputs


def _load_case_result_from_exports(context: FullContext, case: str, stage: str) -> FullCaseResult:
    stage_root = STAGE_ROOTS[stage]
    annual_static = pd.read_csv(validate_exists(stage_root / f"annual_climate_static_{case}.csv", f"{stage} annual static {case}"))
    annual_dynamic = pd.read_csv(validate_exists(stage_root / f"cumulative_dynamic_GHG_{case}.csv", f"{stage} dynamic GHG {case}"))
    annual_rf = pd.read_csv(validate_exists(stage_root / f"cumulative_dynamic_RF_{case}.csv", f"{stage} dynamic RF {case}"))
    non_climate_scores = pd.read_csv(
        validate_exists(stage_root / f"non_climate_time_explicit_static_scores_{case}.csv", f"{stage} non-climate scores {case}")
    )
    top_gwp_flows = pd.read_csv(validate_exists(stage_root / f"top_biosphere_flows_GHG_{case}.csv", f"{stage} top GHG flows {case}"))
    top_rf_flows = pd.read_csv(validate_exists(stage_root / f"top_biosphere_flows_RF_{case}.csv", f"{stage} top RF flows {case}"))

    annual_ghg = annual_static.merge(
        annual_dynamic,
        on=["year", "normalization_mode", "foreground_set", "denominator_lifetime_kwh_th"],
        how="outer",
    ).sort_values("year")

    if stage == "total":
        wrapper_key = context.total_wrappers[case]
        fg_db = context.total_wrappers[case][0]
    else:
        wrapper_key = context.wrappers[case][stage]
        fg_db = context.wrappers[case][stage][0]

    return FullCaseResult(
        case=case,
        stage=stage,
        fg_db=fg_db,
        wrapper_key=wrapper_key,
        tlca=None,
        inv_df=pd.DataFrame(),
        climate_static_annual=annual_static.copy(),
        annual_ghg=annual_ghg,
        annual_rf=annual_rf,
        top_gwp_flows=top_gwp_flows,
        top_rf_flows=top_rf_flows,
        top_gwp_sources=pd.DataFrame(),
        climate_summary=pd.DataFrame([{"case": case, "stage": stage, "fg_db": fg_db}]),
        non_climate_scores=non_climate_scores,
        non_climate_annual=pd.DataFrame(),
    )


def _non_climate_total_compare(gen: FullCaseResult, bau: FullCaseResult) -> pd.DataFrame:
    return gen.non_climate_scores.merge(
        bau.non_climate_scores,
        on=["stage", "category_key", "category_display_name", "method_family", "method_tuple", "unit", "supplemental_metric"],
        how="outer",
        suffixes=("_GEN", "_BAU"),
    ).fillna(0.0)


def _stage_contribution_table(
    results: dict[str, FullCaseResult],
    total: FullCaseResult,
    case: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate_rows = []
    for stage_key in ["A", "B", "C"]:
        stage_result = results[stage_key]
        metric_values = {
            "static_climate": float(
                stage_result.annual_ghg["annual_time_explicit_static_climate_kgCO2e_case_total"].sum()
            ),
            "dynamic_GWP100": float(
                stage_result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()
            ),
            "cumulative_RF": float(
                stage_result.annual_rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1]
            ),
        }
        total_values = {
            "static_climate": float(
                total.annual_ghg["annual_time_explicit_static_climate_kgCO2e_case_total"].sum()
            ),
            "dynamic_GWP100": float(total.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
            "cumulative_RF": float(
                total.annual_rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1]
            ),
        }
        units = {
            "static_climate": "kg CO2e",
            "dynamic_GWP100": "kg CO2e",
            "cumulative_RF": "W*yr/m^2",
        }
        for metric_key, value in metric_values.items():
            total_value = total_values[metric_key]
            climate_rows.append(
                {
                    "case": case,
                    "stage": stage_key,
                    "metric": metric_key,
                    "unit": units[metric_key],
                    "stage_score_case_total": value,
                    "total_score_case_total": total_value,
                    "share_of_total": (value / total_value) if total_value else np.nan,
                }
            )

    non_climate = []
    total_lookup = total.non_climate_scores.set_index("category_key")["static_score_case_total"].to_dict()
    for stage_key in ["A", "B", "C"]:
        stage_scores = results[stage_key].non_climate_scores
        for row in stage_scores.to_dict(orient="records"):
            total_value = float(total_lookup.get(row["category_key"], np.nan))
            non_climate.append(
                {
                    "case": case,
                    "stage": stage_key,
                    "category_key": row["category_key"],
                    "category_display_name": row["category_display_name"],
                    "unit": row["unit"],
                    "stage_score_case_total": float(row["static_score_case_total"]),
                    "total_score_case_total": total_value,
                    "share_of_total": (float(row["static_score_case_total"]) / total_value)
                    if total_value
                    else np.nan,
                }
            )
    return pd.DataFrame(climate_rows), pd.DataFrame(non_climate)


def _build_normalization_check(context: FullContext, results: dict[str, dict[str, FullCaseResult]]) -> pd.DataFrame:
    rows = []
    for case, stage_map in results.items():
        for stage, result in stage_map.items():
            rows.append(
                {
                    "case": case,
                    "stage": stage,
                    "fg_db": result.fg_db,
                    "wrapper_key": str(result.wrapper_key),
                    "normalization_mode": context.normalization_mode,
                    "foreground_set": context.foreground_set,
                    "denominator_lifetime_kwh_th": context.denominator_lifetime_kwh_th,
                    "denominator_annual_kwh_th": context.denominator_annual_kwh_th,
                }
            )
    return pd.DataFrame(rows)


def _finalize_outputs(
    context: FullContext,
    results: dict[str, dict[str, FullCaseResult]],
    method_check: pd.DataFrame,
    t0: dt.datetime,
    horizon_years: int,
) -> dict[str, Path]:
    total_outputs = {
        "annual_climate_static_GEN": STAGE_ROOTS["total"] / "annual_climate_static_GEN.csv",
        "annual_climate_static_BAU": STAGE_ROOTS["total"] / "annual_climate_static_BAU.csv",
        "cumulative_dynamic_GHG_GEN_vs_BAU": STAGE_ROOTS["total"] / "cumulative_dynamic_GHG_GEN_vs_BAU.csv",
        "cumulative_dynamic_RF_GEN_vs_BAU": STAGE_ROOTS["total"] / "cumulative_dynamic_RF_GEN_vs_BAU.csv",
        "non_climate_time_explicit_static_scores_GEN": STAGE_ROOTS["total"] / "non_climate_time_explicit_static_scores_GEN.csv",
        "non_climate_time_explicit_static_scores_BAU": STAGE_ROOTS["total"] / "non_climate_time_explicit_static_scores_BAU.csv",
        "non_climate_static_scores_GEN_legacy": STAGE_ROOTS["total"] / "non_climate_static_scores_GEN.csv",
        "non_climate_static_scores_BAU_legacy": STAGE_ROOTS["total"] / "non_climate_static_scores_BAU.csv",
        "non_climate_stage_contribution_GEN": STAGE_ROOTS["total"] / "non_climate_stage_contribution_GEN.csv",
        "non_climate_stage_contribution_BAU": STAGE_ROOTS["total"] / "non_climate_stage_contribution_BAU.csv",
        "climate_stage_contribution_GEN": STAGE_ROOTS["total"] / "climate_stage_contribution_GEN.csv",
        "climate_stage_contribution_BAU": STAGE_ROOTS["total"] / "climate_stage_contribution_BAU.csv",
        "top_biosphere_flows_GHG_GEN": STAGE_ROOTS["total"] / "top_biosphere_flows_GHG_GEN.csv",
        "top_biosphere_flows_GHG_BAU": STAGE_ROOTS["total"] / "top_biosphere_flows_GHG_BAU.csv",
        "top_biosphere_flows_RF_GEN": STAGE_ROOTS["total"] / "top_biosphere_flows_RF_GEN.csv",
        "top_biosphere_flows_RF_BAU": STAGE_ROOTS["total"] / "top_biosphere_flows_RF_BAU.csv",
    }

    for case in ALL_CASES:
        _write_stage_case_exports(results[case]["total"])

    ghg_total_compare, rf_total_compare = _comparison_tables(results["GEN"]["total"], results["BAU"]["total"])
    ghg_total_compare.to_csv(total_outputs["cumulative_dynamic_GHG_GEN_vs_BAU"], index=False)
    rf_total_compare.to_csv(total_outputs["cumulative_dynamic_RF_GEN_vs_BAU"], index=False)
    results["GEN"]["total"].non_climate_scores.to_csv(total_outputs["non_climate_time_explicit_static_scores_GEN"], index=False)
    results["BAU"]["total"].non_climate_scores.to_csv(total_outputs["non_climate_time_explicit_static_scores_BAU"], index=False)
    results["GEN"]["total"].non_climate_scores.to_csv(total_outputs["non_climate_static_scores_GEN_legacy"], index=False)
    results["BAU"]["total"].non_climate_scores.to_csv(total_outputs["non_climate_static_scores_BAU_legacy"], index=False)
    results["GEN"]["total"].top_gwp_flows.to_csv(total_outputs["top_biosphere_flows_GHG_GEN"], index=False)
    results["BAU"]["total"].top_gwp_flows.to_csv(total_outputs["top_biosphere_flows_GHG_BAU"], index=False)
    results["GEN"]["total"].top_rf_flows.to_csv(total_outputs["top_biosphere_flows_RF_GEN"], index=False)
    results["BAU"]["total"].top_rf_flows.to_csv(total_outputs["top_biosphere_flows_RF_BAU"], index=False)

    climate_contrib: dict[str, pd.DataFrame] = {}
    non_climate_contrib: dict[str, pd.DataFrame] = {}
    for case in ALL_CASES:
        climate_contrib[case], non_climate_contrib[case] = _stage_contribution_table(
            results=results[case],
            total=results[case]["total"],
            case=case,
        )
        climate_contrib[case].to_csv(total_outputs[f"climate_stage_contribution_{case}"], index=False)
        non_climate_contrib[case].to_csv(total_outputs[f"non_climate_stage_contribution_{case}"], index=False)

    normalization_check = _build_normalization_check(context, results)
    method_summary = _build_method_summary(context, method_check, t0=t0, horizon_years=horizon_years)
    sanity = _build_sanity_checks(context, results, normalization_check, method_check)

    method_check.to_csv(QA_ROOT / "method_availability_check.csv", index=False)
    normalization_check.to_csv(QA_ROOT / "normalization_check_full.csv", index=False)
    method_summary.to_csv(QA_ROOT / "denominator_and_method_summary_full.csv", index=False)
    sanity.to_csv(QA_ROOT / "sanity_checks_full.csv", index=False)

    stage_scope_src = validate_exists(QA_ROOT / "stage_scope_check.csv", "stage scope check")
    if stage_scope_src != QA_ROOT / "stage_scope_check.csv":
        pd.read_csv(stage_scope_src).to_csv(QA_ROOT / "stage_scope_check.csv", index=False)

    _write_markdown_summaries(context, results, climate_contrib, non_climate_contrib)
    _write_overall_summaries(context, results)

    for stage in ["A", "B", "C"]:
        ghg_compare, rf_compare = _comparison_tables(results["GEN"][stage], results["BAU"][stage])
        stage_label = f"Stage {stage}"
        _plot_compare_lines(
            ghg_compare,
            gen_col="cumulative_dynamic_GWP100_kgCO2e_case_total_GEN",
            bau_col="cumulative_dynamic_GWP100_kgCO2e_case_total_BAU",
            ylabel="Cumulative dynamic GWP100 (kg CO2e, case total)",
            title=f"{stage_label} cumulative dynamic GWP100, GEN vs REF\n{_title_suffix(context)}",
            basepath=FIGURE_ROOT / f"stage_{stage}_cumulative_dynamic_GHG_GEN_vs_BAU",
        )
        _plot_compare_lines(
            rf_compare,
            gen_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN",
            bau_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU",
            ylabel="Cumulative annual RF sum (W*yr/m^2, case total)",
            title=f"{stage_label} cumulative dynamic RF, GEN vs REF\n{_title_suffix(context)}",
            basepath=FIGURE_ROOT / f"stage_{stage}_cumulative_dynamic_RF_GEN_vs_BAU",
        )
        _plot_non_climate_compare(
            _non_climate_total_compare(results["GEN"][stage], results["BAU"][stage]),
            title=f"{stage_label} time-explicit static LCIA (non-climate), GEN vs REF\n{_title_suffix(context)}",
            basepath=FIGURE_ROOT / f"stage_{stage}_non_climate_time_explicit_static_GEN_vs_BAU",
        )
        for case in ALL_CASES:
            _plot_top_flow_lines(
                results[case][stage].top_gwp_flows,
                flow_col="flow_label",
                value_col="dynamic_GWP100_kgCO2e_case_total",
                title=f"{stage_label} top GHG biosphere flows, {_case_label(case)}\n{_title_suffix(context)}",
                ylabel="Dynamic GWP100 contribution (kg CO2e, case total)",
                basepath=FIGURE_ROOT / f"stage_{stage}_top_GHG_biosphere_flows_{case}",
            )
            _plot_top_flow_lines(
                results[case][stage].top_rf_flows,
                flow_col="flow_label",
                value_col="radiative_forcing_W_per_m2_case_total",
                title=f"{stage_label} top RF biosphere flows, {_case_label(case)}\n{_title_suffix(context)}",
                ylabel="Radiative forcing contribution (W/m^2, case total)",
                basepath=FIGURE_ROOT / f"stage_{stage}_top_RF_biosphere_flows_{case}",
            )

    _plot_compare_lines(
        ghg_total_compare,
        gen_col="cumulative_dynamic_GWP100_kgCO2e_case_total_GEN",
        bau_col="cumulative_dynamic_GWP100_kgCO2e_case_total_BAU",
        ylabel="Cumulative dynamic GWP100 (kg CO2e, case total)",
        title=f"Total system cumulative dynamic GWP100, GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_cumulative_dynamic_GHG_GEN_vs_BAU",
    )
    _plot_compare_lines(
        rf_total_compare,
        gen_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN",
        bau_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU",
        ylabel="Cumulative annual RF sum (W*yr/m^2, case total)",
        title=f"Total system cumulative dynamic RF, GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_cumulative_dynamic_RF_GEN_vs_BAU",
    )
    _plot_non_climate_compare(
        _non_climate_total_compare(results["GEN"]["total"], results["BAU"]["total"]),
        title=f"Total system time-explicit static LCIA (non-climate), GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_non_climate_time_explicit_static_GEN_vs_BAU",
    )
    _plot_stacked_share(
        climate_contrib["GEN"],
        category_col="metric",
        title=f"GEN stage shares of total climate results\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_climate_stage_shares_GEN",
    )
    _plot_stacked_share(
        climate_contrib["BAU"],
        category_col="metric",
        title=f"REF stage shares of total climate results\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_climate_stage_shares_BAU",
    )
    _plot_stacked_share(
        non_climate_contrib["GEN"],
        category_col="category_display_name",
        title=f"GEN stage shares of total non-climate results\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_non_climate_stage_shares_GEN",
    )
    _plot_stacked_share(
        non_climate_contrib["BAU"],
        category_col="category_display_name",
        title=f"REF stage shares of total non-climate results\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "total_non_climate_stage_shares_BAU",
    )
    for case in ALL_CASES:
        _plot_top_flow_lines(
            results[case]["total"].top_gwp_flows,
            flow_col="flow_label",
            value_col="dynamic_GWP100_kgCO2e_case_total",
            title=f"Total system top GHG biosphere flows, {_case_label(case)}\n{_title_suffix(context)}",
            ylabel="Dynamic GWP100 contribution (kg CO2e, case total)",
            basepath=FIGURE_ROOT / f"total_top_GHG_biosphere_flows_{case}",
        )
        _plot_top_flow_lines(
            results[case]["total"].top_rf_flows,
            flow_col="flow_label",
            value_col="radiative_forcing_W_per_m2_case_total",
            title=f"Total system top RF biosphere flows, {_case_label(case)}\n{_title_suffix(context)}",
            ylabel="Radiative forcing contribution (W/m^2, case total)",
            basepath=FIGURE_ROOT / f"total_top_RF_biosphere_flows_{case}",
        )

    manifest = {
        "project": context.project,
        "normalization_mode": context.normalization_mode,
        "foreground_set": context.foreground_set,
        "eol_scenario": context.eol_scenario,
        "denominator_lifetime_kwh_th": context.denominator_lifetime_kwh_th,
        "time_horizon_years": horizon_years,
        "time_horizon_start": t0.strftime("%Y-%m-%d"),
        "stage_wrappers": {case: {stage: list(key) for stage, key in stage_map.items()} for case, stage_map in context.wrappers.items()},
        "total_wrappers": {case: list(key) for case, key in context.total_wrappers.items()},
        "generated_roots": {key: relpath(path) for key, path in STAGE_ROOTS.items()},
    }
    write_json(QA_ROOT / "full_dynamic_lcia_manifest.json", manifest)

    return {
        "qa_method_summary": QA_ROOT / "denominator_and_method_summary_full.csv",
        "qa_normalization": QA_ROOT / "normalization_check_full.csv",
        "qa_stage_scope": QA_ROOT / "stage_scope_check.csv",
        "qa_methods": QA_ROOT / "method_availability_check.csv",
        "qa_sanity": QA_ROOT / "sanity_checks_full.csv",
        "manifest": QA_ROOT / "full_dynamic_lcia_manifest.json",
    }


def _build_method_summary(context: FullContext, method_check: pd.DataFrame, t0: dt.datetime, horizon_years: int) -> pd.DataFrame:
    rows = [
        {"category": "project", "key": "brightway_project", "value": context.project},
        {"category": "mode", "key": "normalization_mode", "value": context.normalization_mode},
        {"category": "mode", "key": "foreground_set", "value": context.foreground_set},
        {"category": "mode", "key": "eol_scenario", "value": context.eol_scenario},
        {"category": "mode", "key": "denominator_lifetime_kwh_th", "value": context.denominator_lifetime_kwh_th},
        {"category": "mode", "key": "denominator_annual_kwh_th", "value": context.denominator_annual_kwh_th},
        {"category": "methods", "key": "static_climate_method", "value": " | ".join(context.climate_method)},
        {"category": "methods", "key": "dynamic_ghg_method", "value": "dynamic_characterization IPCC AR6 GWP100"},
        {"category": "methods", "key": "dynamic_rf_method", "value": "dynamic_characterization IPCC AR6 radiative forcing"},
        {"category": "methods", "key": "time_horizon_years", "value": horizon_years},
        {"category": "methods", "key": "time_horizon_start", "value": t0.strftime("%Y-%m-%d")},
        {
            "category": "methods",
            "key": "non_climate_method_family",
            "value": "TRACI v2.1 no LT core categories + CED fossil supplemental metric",
        },
    ]
    for _, row in method_check.iterrows():
        rows.append(
            {
                "category": "method_check",
                "key": row["category_key"],
                "value": f"{row['status']} | {row['method_tuple']}",
            }
        )
    return pd.DataFrame(rows)


def _build_sanity_checks(
    context: FullContext,
    results: dict[str, dict[str, FullCaseResult]],
    normalization_check: pd.DataFrame,
    method_check: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append(
        {
            "check": "All runs use authoritative normalization mode",
            "value": int((normalization_check["normalization_mode"] == "heatnets_case_service_loads").all()),
            "expected": 1,
            "status": "pass"
            if (normalization_check["normalization_mode"] == "heatnets_case_service_loads").all()
            else "fail",
        }
    )
    rows.append(
        {
            "check": "All runs use synchronized authoritative foreground set",
            "value": int((normalization_check["foreground_set"] == "heatnets_authoritative_sync").all()),
            "expected": 1,
            "status": "pass"
            if (normalization_check["foreground_set"] == "heatnets_authoritative_sync").all()
            else "fail",
        }
    )
    rows.append(
        {
            "check": "All non-climate methods are available",
            "value": int((method_check["status"] == "pass").all()),
            "expected": 1,
            "status": "pass" if (method_check["status"] == "pass").all() else "fail",
        }
    )

    for case in ["GEN", "BAU"]:
        stage_map = results[case]
        total = stage_map["total"]
        stage_sum_static = sum(
            stage_map[stage].annual_ghg["annual_time_explicit_static_climate_kgCO2e_case_total"].sum()
            for stage in ["A", "B", "C"]
        )
        total_static = float(total.annual_ghg["annual_time_explicit_static_climate_kgCO2e_case_total"].sum())
        rows.append(
            {
                "check": f"{case} A+B+C static climate reconciles to total",
                "value": stage_sum_static,
                "expected": total_static,
                "status": "pass" if np.isclose(stage_sum_static, total_static, rtol=1e-6, atol=1e-8) else "fail",
            }
        )

        stage_sum_gwp = sum(
            stage_map[stage].annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()
            for stage in ["A", "B", "C"]
        )
        total_gwp = float(total.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum())
        rows.append(
            {
                "check": f"{case} A+B+C dynamic GWP reconciles to total",
                "value": stage_sum_gwp,
                "expected": total_gwp,
                "status": "pass" if np.isclose(stage_sum_gwp, total_gwp, rtol=1e-6, atol=1e-8) else "fail",
            }
        )

        stage_sum_rf = sum(
            stage_map[stage].annual_rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1]
            for stage in ["A", "B", "C"]
        )
        total_rf = float(total.annual_rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"].iloc[-1])
        rows.append(
            {
                "check": f"{case} A+B+C cumulative RF reconciles to total",
                "value": stage_sum_rf,
                "expected": total_rf,
                "status": "pass" if np.isclose(stage_sum_rf, total_rf, rtol=1e-6, atol=1e-12) else "fail",
            }
        )

        total_non_climate = total.non_climate_scores.set_index("category_key")["static_score_case_total"].to_dict()
        for category_key, total_value in total_non_climate.items():
            stage_value = sum(
                float(
                    stage_map[stage]
                    .non_climate_scores.set_index("category_key")
                    .loc[category_key, "static_score_case_total"]
                )
                for stage in ["A", "B", "C"]
            )
            rows.append(
                {
                    "check": f"{case} A+B+C non-climate {category_key} reconciles to total",
                    "value": stage_value,
                    "expected": float(total_value),
                    "status": "pass"
                    if np.isclose(stage_value, float(total_value), rtol=1e-6, atol=1e-8)
                    else "fail",
                }
            )

    return pd.DataFrame(rows)


def _write_markdown_summaries(
    context: FullContext,
    results: dict[str, dict[str, FullCaseResult]],
    climate_contrib: dict[str, pd.DataFrame],
    non_climate_contrib: dict[str, pd.DataFrame],
) -> None:
    stage_labels = {
        "A": ("Stage A", "stage_A"),
        "B": ("Stage B", "stage_B"),
        "C": ("Stage C", "stage_C"),
        "total": ("Total System", "total"),
    }
    for stage, (label, file_stub) in stage_labels.items():
        gen = results["GEN"][stage]
        bau = results["BAU"][stage]
        method_text = (
            f"{label} was evaluated on the synchronized HEATNETS-authoritative foreground mode "
            f"`{context.foreground_set}` using the audited lifetime denominator "
            f"`{context.denominator_lifetime_kwh_th:,.3f} kWh_th`. Climate results use bw_timex "
            f"time-explicit inventory relinking with prospective premise backgrounds, static climate "
            f"characterization via `{ ' | '.join(context.climate_method) }`, and dynamic characterization "
            f"with IPCC AR6 functions for GWP and radiative forcing. Non-climate results are reported as "
            "static characterization of the time-explicit inventory using TRACI v2.1 no LT core categories "
            f"plus a supplemental CED fossil-energy metric. The active end-of-life scenario is "
            f"`{context.eol_scenario}`."
        )
        write_markdown(SUMMARY_ROOT / f"method_summary_{file_stub}.md", method_text)

        if stage == "total":
            gen_stage_share = climate_contrib["GEN"].sort_values("share_of_total", ascending=False).iloc[0]
            bau_stage_share = climate_contrib["BAU"].sort_values("share_of_total", ascending=False).iloc[0]
            result_text = (
                f"Total-system dynamic GWP over the modeled horizon is "
                f"`{float(gen.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` for GEN and "
                f"`{float(bau.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` for BAU. "
                f"Final cumulative annual RF sums are "
                f"`{float(gen.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for GEN and "
                f"`{float(bau.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for BAU. "
                f"The largest climate stage share is `{gen_stage_share['stage']}` for GEN and "
                f"`{bau_stage_share['stage']}` for BAU, based on the exported contribution tables."
            )
        else:
            gen_top = gen.top_gwp_flows.groupby("flow_label")["dynamic_GWP100_kgCO2e_case_total"].sum().abs().sort_values(ascending=False)
            bau_top = bau.top_gwp_flows.groupby("flow_label")["dynamic_GWP100_kgCO2e_case_total"].sum().abs().sort_values(ascending=False)
            result_text = (
                f"{label} dynamic GWP totals are "
                f"`{float(gen.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` for GEN and "
                f"`{float(bau.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` for BAU. "
                f"Final cumulative RF sums are "
                f"`{float(gen.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for GEN and "
                f"`{float(bau.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for BAU. "
                f"The largest GHG biosphere-flow contributor is `{gen_top.index[0]}` for GEN and "
                f"`{bau_top.index[0]}` for BAU in the exported contributor tables."
            )
        write_markdown(SUMMARY_ROOT / f"result_summary_{file_stub}.md", result_text)


def _write_overall_summaries(
    context: FullContext,
    results: dict[str, dict[str, FullCaseResult]],
) -> None:
    total_gen = results["GEN"]["total"]
    total_bau = results["BAU"]["total"]
    method_text = (
        f"The synchronized Framingham workflow uses the Brightway project `{context.project}` with "
        f"prospective premise backgrounds `{', '.join(context.background_dbs)}` and the authoritative "
        f"foreground set `{context.foreground_set}`. The common lifetime denominator is "
        f"`{context.denominator_lifetime_kwh_th:,.3f} kWh_th`, and the active end-of-life scenario is "
        f"`{context.eol_scenario}`. Climate results are calculated from the time-explicit inventory with "
        f"static climate characterization via `{ ' | '.join(context.climate_method) }` and dynamic "
        "characterization using IPCC AR6 functions for GWP100 and radiative forcing. Non-climate categories "
        "are not treated as dynamic LCIA; they are reported as static characterization of the time-explicit "
        "inventory using TRACI v2.1 no LT core categories plus a supplemental CED fossil-energy metric."
    )
    result_text = (
        f"Total dynamic GWP is `{float(total_gen.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` "
        f"for GEN and `{float(total_bau.annual_ghg['annual_dynamic_GWP100_kgCO2e_case_total'].sum()):,.3f} kg CO2e` for BAU. "
        f"Final cumulative RF sums are "
        f"`{float(total_gen.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for GEN and "
        f"`{float(total_bau.annual_rf['cumulative_discrete_annual_rf_Wyr_per_m2_case_total'].iloc[-1]):,.6g} W*yr/m^2` for BAU. "
        "GEN retains a larger upfront climate burden, while BAU remains dominated by Stage B because operational "
        "energy use and direct combustion accumulate across the life cycle."
    )
    discussion_text = (
        "The stage pattern is internally consistent with the synchronized foreground logic. GEN carries a higher "
        "Stage A burden because the geothermal network, borefield, and associated equipment are installed at the "
        "start of life. BAU Stage A is intentionally smaller because the modeled foreground represents incumbent "
        "HVAC-scale equipment and replacements rather than new district infrastructure or whole-building shells. "
        "BAU then dominates in Stage B because direct combustion, methane leakage, and ongoing operational energy "
        "use compound over the service life. Stage C is a small share of total impact and should be interpreted "
        "primarily as an end-of-life routing sensitivity rather than a dominant driver."
    )
    limitations_text = (
        "Confidence is highest for climate results, especially Stage B and total dynamic climate comparisons, "
        "because they rely on the accepted HEATNETS-aligned denominator, explicit timed foreground events, and "
        "dynamic characterization functions already validated in the project toolchain. Confidence is moderate "
        "for stage timing and stage shares because those are directly encoded in the synchronized foregrounds and "
        "verified by wrapper audit tables. Confidence is moderate to lower for non-climate categories because "
        "they are static characterization of the time-explicit inventory, and some upstream inventories remain "
        "proxy-based or sensitive to end-of-life assumptions. Stage C results remain especially sensitive to the "
        "chosen routing scenario, even after removal of unsupported generic material recycling credits, and should "
        "be framed with that limitation."
    )
    write_markdown(SUMMARY_ROOT / "method_summary_overall.md", method_text)
    write_markdown(SUMMARY_ROOT / "result_summary_overall.md", result_text)
    write_markdown(SUMMARY_ROOT / "discussion_overall.md", discussion_text)
    write_markdown(SUMMARY_ROOT / "limitations_overall.md", limitations_text)


def _plot_non_climate_compare(compare_df: pd.DataFrame, title: str, basepath: Path) -> None:
    _set_plot_style()
    data = compare_df.copy().sort_values("category_display_name").reset_index(drop=True)
    n = len(data)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14.5, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (_, row) in zip(axes, data.iterrows()):
        ax.bar([_case_label("GEN"), _case_label("BAU")], [row["static_score_case_total_GEN"], row["static_score_case_total_BAU"]], color=["#1b9e77", "#d95f02"])
        ax.set_title(f"{row['category_display_name']}\n[{row['unit']}]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Case total")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save_figure(fig, basepath)


def _plot_stacked_share(contrib_df: pd.DataFrame, category_col: str, title: str, basepath: Path) -> None:
    _set_plot_style()
    pivot = (
        contrib_df.pivot(index=category_col, columns="stage", values="share_of_total")
        .fillna(0.0)
        .reindex(columns=["A", "B", "C"], fill_value=0.0)
    )
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bottom = np.zeros(len(pivot.index))
    colors = {"A": "#1b9e77", "B": "#7570b3", "C": "#d95f02"}
    for stage in ["A", "B", "C"]:
        values = pivot[stage].to_numpy()
        ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            width=0.72,
            color=colors[stage],
            label=f"Stage {stage}",
        )
        bottom += values
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of total impact")
    ax.set_title(title)
    ax.legend(frameon=True, loc="best")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, basepath)


def run(
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    t0: dt.datetime = DEFAULT_T0,
    cases: list[str] | None = None,
    stages: list[str] | None = None,
    assemble_from_exports: bool = False,
) -> dict[str, Path]:
    _ensure_output_tree()
    _remove_stale_outputs()
    ready = _require_ready()
    if not assemble_from_exports:
        build_stage_wrappers(export_csv=True)
    context = _load_context()

    try:
        import bw2data as bd
    except ImportError as exc:
        raise RuntimeError("Run full dynamic LCIA from the `bw25` environment.") from exc

    bd.projects.set_current(context.project)
    method_check = _validate_methods()

    selected_cases = cases or list(ALL_CASES)
    selected_stages = stages or list(ALL_STAGES)
    invalid_cases = sorted(set(selected_cases) - set(ALL_CASES))
    invalid_stages = sorted(set(selected_stages) - set(ALL_STAGES))
    if invalid_cases:
        raise ValueError(f"Unsupported case selectors: {invalid_cases}")
    if invalid_stages:
        raise ValueError(f"Unsupported stage selectors: {invalid_stages}")

    if assemble_from_exports:
        results = {
            case: {stage: _load_case_result_from_exports(context, case, stage) for stage in ALL_STAGES}
            for case in ALL_CASES
        }
        return _finalize_outputs(context, results, method_check, t0=t0, horizon_years=horizon_years)

    results: dict[str, dict[str, FullCaseResult]] = {"GEN": {}, "BAU": {}}
    stage_key_map = {"A": "A", "B": "B", "C": "C", "total": "total"}
    fg_db_map = {
        "GEN": context.wrappers["GEN"]["A"][0],
        "BAU": context.wrappers["BAU"]["A"][0],
    }

    partial_outputs: dict[str, Path] = {}
    for case in selected_cases:
        for stage in selected_stages:
            if stage == "total":
                result = _run_case(
                    case=case,
                    stage="total",
                    fg_db=context.total_wrappers[case][0],
                    wrapper_key=context.total_wrappers[case],
                    context=context,
                    t0=t0,
                    horizon_years=horizon_years,
                )
            else:
                result = _run_case(
                    case=case,
                    stage=stage,
                    fg_db=fg_db_map[case],
                    wrapper_key=context.wrappers[case][stage_key_map[stage]],
                    context=context,
                    t0=t0,
                    horizon_years=horizon_years,
                )
            results[case][stage] = result
            for key, path in _write_stage_case_exports(result).items():
                partial_outputs[f"{case}_{stage}_{key}"] = path

    if set(selected_cases) != set(ALL_CASES) or set(selected_stages) != set(ALL_STAGES):
        return partial_outputs

    return _finalize_outputs(context, results, method_check, t0=t0, horizon_years=horizon_years)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full Stage A/B/C + total time-explicit LCIA for GEN vs BAU on the synchronized HEATNETS-authoritative foregrounds."
    )
    parser.add_argument(
        "--time-horizon-years",
        type=int,
        default=DEFAULT_HORIZON_YEARS,
        help="Dynamic characterization horizon in years.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(ALL_CASES),
        help="Optional subset of cases to run. Use with `--stages` for smaller batches.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(ALL_STAGES),
        help="Optional subset of stages to run. Use `total` to refresh only total-system outputs.",
    )
    parser.add_argument(
        "--assemble-from-exports",
        action="store_true",
        help="Skip Timex reruns and assemble comparisons/figures/QA from existing stage and total exports.",
    )
    args = parser.parse_args()

    outputs = run(
        horizon_years=args.time_horizon_years,
        t0=DEFAULT_T0,
        cases=args.cases,
        stages=args.stages,
        assemble_from_exports=args.assemble_from_exports,
    )
    print("Full dynamic LCIA outputs:")
    for key, path in outputs.items():
        print(f" - {key}: {relpath(path)}")


if __name__ == "__main__":
    main()
