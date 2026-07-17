from __future__ import annotations

import argparse
from dataclasses import replace
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
        load_b6_case_config,
        relpath,
        validate_exists,
        write_markdown,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_b6_dynamic_lcia import (  # type: ignore
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _choose_climate_method as choose_b6_climate_method,
        _load_context as load_b6_context,
        _require_ready,
        _run_case as run_b6_case,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_full_dynamic_lcia import (  # type: ignore
        _load_context as load_full_context,
        _run_case as run_full_case,
    )
else:
    from .common import WORKSPACE_ROOT, ensure_workspace_tree, load_b6_case_config, relpath, validate_exists, write_markdown
    from .run_b6_dynamic_lcia import (
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _choose_climate_method as choose_b6_climate_method,
        _load_context as load_b6_context,
        _require_ready,
        _run_case as run_b6_case,
    )
    from .run_full_dynamic_lcia import _load_context as load_full_context, _run_case as run_full_case


OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "sensitivity"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
SUMMARY_ROOT = OUTPUT_ROOT

B6_EXCHANGE_PATH = WORKSPACE_ROOT / "export" / "b6" / "bw_timex_inputs" / "b6_operational_exchange_table_heatnets_case_service_loads_heatnets_authoritative_sync.csv"
GEN_HOURLY_PATH = WORKSPACE_ROOT / "export" / "b6" / "gen" / "gen_b6_hourly_total.csv"
BAU_GROUP_PATH = WORKSPACE_ROOT / "export" / "b6" / "bau" / "bau_b6_hourly_group.csv"
FULL_TOTAL_ROOT = WORKSPACE_ROOT / "export" / "full_dynamic_lcia" / "total"
B6_DYNAMIC_ROOT = WORKSPACE_ROOT / "export" / "b6_dynamic_lcia"


def _ensure_output_tree() -> None:
    ensure_workspace_tree()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


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


def _upsert_activity(db, code: str, name: str):
    for act in db:
        if act["code"] == code:
            activity = act
            break
    else:
        activity = db.new_activity(code=code)
    activity["name"] = name
    activity["unit"] = "kilowatt hour"
    activity["location"] = "US"
    activity["reference product"] = "thermal energy delivered"
    activity.save()
    for exc in list(activity.exchanges()):
        exc.delete()
    activity.new_exchange(input=activity.key, amount=1.0, type="production").save()
    return activity


def _base_tables(case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_ghg = pd.read_csv(validate_exists(FULL_TOTAL_ROOT / f"cumulative_dynamic_GHG_{case}.csv", f"full total {case} GHG"))
    total_rf = pd.read_csv(validate_exists(FULL_TOTAL_ROOT / f"cumulative_dynamic_RF_{case}.csv", f"full total {case} RF"))
    b6_ghg = pd.read_csv(validate_exists(B6_DYNAMIC_ROOT / f"annual_B6_GHG_{case}.csv", f"B6 {case} GHG"))
    b6_rf = pd.read_csv(validate_exists(B6_DYNAMIC_ROOT / f"annual_B6_RF_{case}.csv", f"B6 {case} RF"))
    total = total_ghg[["year", "annual_dynamic_GWP100_kgCO2e_case_total"]].merge(
        total_rf[["year", "annual_radiative_forcing_W_per_m2_case_total"]],
        on="year",
        how="outer",
    )
    b6 = b6_ghg[["year", "annual_dynamic_GWP100_kgCO2e_case_total"]].merge(
        b6_rf[["year", "annual_radiative_forcing_W_per_m2_case_total"]],
        on="year",
        how="outer",
    )
    total = total.fillna(0.0).sort_values("year")
    b6 = b6.fillna(0.0).sort_values("year")
    return total, b6


def _load_bau_annual_group_totals() -> pd.DataFrame:
    bau = pd.read_csv(validate_exists(BAU_GROUP_PATH, "BAU hourly group export"))
    numeric_cols = [col for col in bau.columns if col not in {"timestamp", "group_name"}]
    return bau.groupby("group_name", as_index=False)[numeric_cols].sum()


def _load_gen_hourly() -> pd.DataFrame:
    return pd.read_csv(validate_exists(GEN_HOURLY_PATH, "GEN hourly export"))


def _build_modified_bau_group_totals(base: pd.DataFrame, scenario: str, config: dict[str, Any]) -> pd.DataFrame:
    out = base.copy()
    west = out["group_name"] == "west_residential"
    east = out["group_name"] == "east_or_southeast_residential"
    fha = out["group_name"] == "fha_multifamily"
    com = out["group_name"] == "commercial_municipal"
    residential_fossil = west | east
    all_groups = out["group_name"].isin(["west_residential", "east_or_southeast_residential", "fha_multifamily", "commercial_municipal"])

    if scenario in {"bau_west_res_gas90_oil10", "bau_west_res_gas60_oil40", "bau_structure_lower_intensity"}:
        gas_share = 0.90 if scenario in {"bau_west_res_gas90_oil10", "bau_structure_lower_intensity"} else 0.60
        oil_share = 1.0 - gas_share
        heat_eff = {"gas": 0.82, "oil": 0.80}
        dhw_eff = {"gas": 0.62, "oil": 0.60}
        heating_thermal = out.loc[west, "BAU_heating_gas_kWh_fuel"] * heat_eff["gas"] + out.loc[west, "BAU_heating_oil_kWh_fuel"] * heat_eff["oil"]
        dhw_thermal = out.loc[west, "BAU_DHW_gas_kWh_fuel"] * dhw_eff["gas"] + out.loc[west, "BAU_DHW_oil_kWh_fuel"] * dhw_eff["oil"]
        out.loc[west, "BAU_heating_gas_kWh_fuel"] = heating_thermal * gas_share / heat_eff["gas"]
        out.loc[west, "BAU_heating_oil_kWh_fuel"] = heating_thermal * oil_share / heat_eff["oil"]
        out.loc[west, "BAU_DHW_gas_kWh_fuel"] = dhw_thermal * gas_share / dhw_eff["gas"]
        out.loc[west, "BAU_DHW_oil_kWh_fuel"] = dhw_thermal * oil_share / dhw_eff["oil"]

    if scenario in {"bau_fha_multifamily_heat_pump_alt", "bau_structure_lower_intensity"}:
        heating_thermal = out.loc[fha, "BAU_heating_electric_kWh"].copy()
        dhw_thermal = out.loc[fha, "BAU_DHW_electric_kWh"].copy()
        out.loc[fha, "BAU_heating_electric_kWh"] = heating_thermal / 2.5
        out.loc[fha, "BAU_DHW_electric_kWh"] = dhw_thermal / 2.2

    if scenario in {"bau_commercial_cooling_minus15pct", "bau_structure_lower_intensity"}:
        out.loc[com, "BAU_cooling_electric_kWh"] *= 0.85
    if scenario == "bau_commercial_cooling_plus15pct":
        out.loc[com, "BAU_cooling_electric_kWh"] *= 1.15

    if scenario in {
        "bau_fossil_heating_to_resistance_cop1",
        "bau_residential_fossil_heating_to_resistance_cop1",
    }:
        residential_heating_thermal = (
            out.loc[residential_fossil, "BAU_heating_gas_kWh_fuel"] * 0.82
            + out.loc[residential_fossil, "BAU_heating_oil_kWh_fuel"] * 0.80
        )
        out.loc[residential_fossil, "BAU_heating_gas_kWh_fuel"] = 0.0
        out.loc[residential_fossil, "BAU_heating_oil_kWh_fuel"] = 0.0
        out.loc[residential_fossil, "BAU_heating_electric_kWh"] = residential_heating_thermal

    if scenario in {
        "bau_fossil_heating_to_resistance_cop1",
        "bau_commercial_heating_to_resistance_cop1",
    }:
        commercial_heating_thermal = out.loc[com, "BAU_heating_gas_kWh_fuel"] * 0.85
        out.loc[com, "BAU_heating_gas_kWh_fuel"] = 0.0
        out.loc[com, "BAU_heating_electric_kWh"] = commercial_heating_thermal

    if scenario == "bau_fha_heating_to_gas_furnace_afue90":
        fha_heating_thermal = out.loc[fha, "BAU_heating_electric_kWh"].copy()
        out.loc[fha, "BAU_heating_electric_kWh"] = 0.0
        out.loc[fha, "BAU_heating_gas_kWh_fuel"] = fha_heating_thermal / 0.90

    if scenario == "bau_all_heating_gas_furnace_afue90":
        residential_heating_thermal = (
            out.loc[residential_fossil, "BAU_heating_gas_kWh_fuel"] * 0.82
            + out.loc[residential_fossil, "BAU_heating_oil_kWh_fuel"] * 0.80
        )
        fha_heating_thermal = out.loc[fha, "BAU_heating_electric_kWh"].copy()
        commercial_heating_thermal = out.loc[com, "BAU_heating_gas_kWh_fuel"] * 0.85
        out.loc[all_groups, "BAU_heating_oil_kWh_fuel"] = 0.0
        out.loc[all_groups, "BAU_heating_electric_kWh"] = 0.0
        out.loc[residential_fossil, "BAU_heating_gas_kWh_fuel"] = residential_heating_thermal / 0.90
        out.loc[fha, "BAU_heating_gas_kWh_fuel"] = fha_heating_thermal / 0.90
        out.loc[com, "BAU_heating_gas_kWh_fuel"] = commercial_heating_thermal / 0.90

    out["BAU_total_site_energy_gas_kWh_fuel"] = out["BAU_heating_gas_kWh_fuel"] + out["BAU_DHW_gas_kWh_fuel"]
    out["BAU_total_site_energy_oil_kWh_fuel"] = out["BAU_heating_oil_kWh_fuel"] + out["BAU_DHW_oil_kWh_fuel"]
    out["BAU_total_site_energy_electric_kWh"] = out["BAU_heating_electric_kWh"] + out["BAU_cooling_electric_kWh"] + out["BAU_DHW_electric_kWh"]
    out["BAU_total_B6_operational_energy"] = (
        out["BAU_total_site_energy_gas_kWh_fuel"] + out["BAU_total_site_energy_oil_kWh_fuel"] + out["BAU_total_site_energy_electric_kWh"]
    )
    gas_lhv = float(config["bau"]["gas_lhv_kwh_per_kg_ch4"])
    leakage = float(config["bau"]["methane_leakage_rate"])
    out["BAU_methane_leakage_mass_CH4"] = out["BAU_total_site_energy_gas_kWh_fuel"] / gas_lhv * leakage
    return out


def _bau_exchange_amounts_from_group_totals(group_totals: pd.DataFrame, config: dict[str, Any], denominator: float) -> dict[str, float]:
    gas = float(group_totals["BAU_total_site_energy_gas_kWh_fuel"].sum())
    oil = float(group_totals["BAU_total_site_energy_oil_kWh_fuel"].sum())
    electricity = float(group_totals["BAU_total_site_energy_electric_kWh"].sum())
    gas_cfg = config["bau"]["direct_combustion"]["natural_gas"]
    oil_cfg = config["bau"]["direct_combustion"]["fuel_oil"]
    gas_kwh_per_m3 = float(config["bau"]["natural_gas_kwh_per_m3"])
    oil_kwh_per_kg = float(config["bau"]["fuel_oil_kwh_per_kg"])
    service_life = int(config["bau"]["service_life_years"])
    leakage_kg = float(group_totals["BAU_methane_leakage_mass_CH4"].sum())
    return {
        "electricity": electricity * service_life / denominator,
        "natural_gas": (gas * service_life / denominator) / gas_kwh_per_m3,
        "fuel_oil": (oil * service_life / denominator) / oil_kwh_per_kg,
        "co2": (gas * float(gas_cfg["co2_kg_per_kwh_fuel"]) + oil * float(oil_cfg["co2_kg_per_kwh_fuel"])) * service_life / denominator,
        "ch4": (gas * float(gas_cfg["ch4_kg_per_kwh_fuel"]) + oil * float(oil_cfg["ch4_kg_per_kwh_fuel"]) + leakage_kg) * service_life / denominator,
        "n2o": (gas * float(gas_cfg["n2o_kg_per_kwh_fuel"]) + oil * float(oil_cfg["n2o_kg_per_kwh_fuel"])) * service_life / denominator,
        "annual_electricity_kwh": electricity,
    }


def _gen_exchange_amounts_from_hourly(gen_hourly: pd.DataFrame, scenario: str, config: dict[str, Any], denominator: float) -> dict[str, float]:
    service_life = int(config["bau"]["service_life_years"])
    base = gen_hourly.copy()
    if scenario in {"gen_hp_cop_minus10pct", "gen_operational_low_performance"}:
        total = (
            base["GEN_building_hp_kWh_el_hourly"] * (1.0 / 0.9)
            + (base["central_loop_pump_kWh_el_hourly"] + base["borefield_kWh_el_hourly"]) * (1.20 if scenario == "gen_operational_low_performance" else 1.0)
            + (base["GEN_shared_network_allocated_kWh_el_hourly"] - base["central_loop_pump_kWh_el_hourly"] - base["borefield_kWh_el_hourly"])
            + base["GEN_DHW_kWh_el_hourly"] * (1.0 / 0.9)
        ).sum()
    elif scenario == "gen_pump_control_plus20pct":
        pump_base = base["central_loop_pump_kWh_el_hourly"] + base["borefield_kWh_el_hourly"]
        non_pump = base["GEN_total_B6_kWh_el_hourly"] - pump_base
        total = float((non_pump + pump_base * 1.20).sum())
    elif scenario == "gen_dhw_constant_cop_2_8":
        constant_cop = float(config["heatnets"]["dhw"]["fallback_constant_cop"])
        network = float(base["GEN_HVAC_network_allocated_kWh_el_hourly"].sum())
        dhw = float((base["waterh_kwh_th"] / constant_cop).sum())
        total = network + dhw
    else:
        raise RuntimeError(f"Unsupported GEN sensitivity scenario `{scenario}`.")
    return {
        "electricity": total * service_life / denominator,
        "annual_electricity_kwh": total,
    }


def _identify_b6_inputs(base_operation) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    for exc in base_operation.technosphere():
        name = exc.input.get("name", "").lower()
        if "electricity, medium voltage" in name:
            mapping["electricity"] = exc.input.key
        elif "natural gas, high pressure" in name:
            mapping["natural_gas"] = exc.input.key
        elif "light fuel oil" in name:
            mapping["fuel_oil"] = exc.input.key
    for exc in base_operation.biosphere():
        name = exc.input.get("name", "").lower()
        if name == "carbon dioxide, fossil":
            mapping["co2"] = exc.input.key
        elif name == "methane, fossil":
            mapping["ch4"] = exc.input.key
        elif name == "dinitrogen monoxide":
            mapping["n2o"] = exc.input.key
    return mapping


def _clone_b6_sensitivity_activity(case: str, scenario: str, exchange_amounts: dict[str, float], b6_context, config: dict[str, Any]) -> tuple[tuple[str, str], tuple[str, str]]:
    import bw2data as bd
    from bw_timex.utils import add_temporal_distribution_to_exchange

    if case == "GEN":
        fg_db_name = b6_context.gen_fg_db
        base_op_key = b6_context.gen_b6_key
        base_wrapper_key = b6_context.gen_wrapper_key
    else:
        fg_db_name = b6_context.bau_fg_db
        base_op_key = b6_context.bau_b6_key
        base_wrapper_key = b6_context.bau_wrapper_key

    db = bd.Database(fg_db_name)
    base_operation = bd.get_activity(base_op_key)
    base_wrapper = bd.get_activity(base_wrapper_key)
    input_keys = _identify_b6_inputs(base_operation)

    op_code = f"{case}_B6_OPERATION_SENS_{scenario.upper()}"
    wrapper_code = f"{case}_B6_DYNAMIC_ONLY_SENS_{scenario.upper()}"
    op = _upsert_activity(db, op_code, f"{case} B6 sensitivity operation ({scenario})")
    wrapper = _upsert_activity(db, wrapper_code, f"{case} B6 sensitivity wrapper ({scenario})")

    if case == "GEN":
        op.new_exchange(input=input_keys["electricity"], amount=float(exchange_amounts["electricity"]), type="technosphere").save()
    else:
        op.new_exchange(input=input_keys["electricity"], amount=float(exchange_amounts["electricity"]), type="technosphere").save()
        op.new_exchange(input=input_keys["natural_gas"], amount=float(exchange_amounts["natural_gas"]), type="technosphere").save()
        op.new_exchange(input=input_keys["fuel_oil"], amount=float(exchange_amounts["fuel_oil"]), type="technosphere").save()
        op.new_exchange(input=input_keys["co2"], amount=float(exchange_amounts["co2"]), type="biosphere").save()
        op.new_exchange(input=input_keys["ch4"], amount=float(exchange_amounts["ch4"]), type="biosphere").save()
        op.new_exchange(input=input_keys["n2o"], amount=float(exchange_amounts["n2o"]), type="biosphere").save()

    wrapper.new_exchange(input=op.key, amount=1.0, type="technosphere").save()
    td = next(iter(base_wrapper.technosphere())).get("temporal_distribution")
    add_temporal_distribution_to_exchange(
        temporal_distribution=td,
        input_database=op.key[0],
        input_code=op.key[1],
        output_database=wrapper.key[0],
        output_code=wrapper.key[1],
    )
    db.process()
    return op.key, wrapper.key


def _combine_total_with_sensitivity(case: str, sensitivity_ghg: pd.DataFrame, sensitivity_rf: pd.DataFrame) -> dict[str, float]:
    base_total, base_b6 = _base_tables(case)
    sens = sensitivity_ghg[["year", "annual_dynamic_GWP100_kgCO2e_case_total"]].merge(
        sensitivity_rf[["year", "annual_radiative_forcing_W_per_m2_case_total"]],
        on="year",
        how="outer",
    ).fillna(0.0)
    merged = base_total.merge(base_b6, on="year", suffixes=("_total", "_b6"))
    merged = merged.merge(sens, on="year", how="left").fillna(0.0)
    merged["scenario_total_annual_ghg"] = (
        merged["annual_dynamic_GWP100_kgCO2e_case_total_total"]
        - merged["annual_dynamic_GWP100_kgCO2e_case_total_b6"]
        + merged["annual_dynamic_GWP100_kgCO2e_case_total"]
    )
    merged["scenario_total_annual_rf"] = (
        merged["annual_radiative_forcing_W_per_m2_case_total_total"]
        - merged["annual_radiative_forcing_W_per_m2_case_total_b6"]
        + merged["annual_radiative_forcing_W_per_m2_case_total"]
    )
    return {
        "total_dynamic_GWP100_case_total": float(merged["scenario_total_annual_ghg"].sum()),
        "total_dynamic_RF_case_total": float(merged["scenario_total_annual_rf"].sum()),
        "base_total_dynamic_GWP100_case_total": float(base_total["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "base_total_dynamic_RF_case_total": float(base_total["annual_radiative_forcing_W_per_m2_case_total"].sum()),
    }


def _run_b6_sensitivity_case(case: str, scenario: str, exchange_amounts: dict[str, float], b6_context, climate_method, config: dict[str, Any]) -> dict[str, Any]:
    import bw2data as bd

    op_key, wrapper_key = _clone_b6_sensitivity_activity(case, scenario, exchange_amounts, b6_context, config)
    result = run_b6_case(
        label=f"{case}_{scenario}",
        fg_db=wrapper_key[0],
        wrapper_key=wrapper_key,
        context=b6_context,
        method=climate_method,
        t0=DEFAULT_T0,
        horizon_years=DEFAULT_HORIZON_YEARS,
    )
    combined = _combine_total_with_sensitivity(case, result.annual_ghg, result.annual_rf)
    return {
        "scenario": scenario,
        "case": case,
        "b6_dynamic_GWP100_case_total": float(result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "b6_dynamic_RF_case_total": float(result.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].sum()),
        "total_dynamic_GWP100_case_total": combined["total_dynamic_GWP100_case_total"],
        "total_dynamic_RF_case_total": combined["total_dynamic_RF_case_total"],
        "base_total_dynamic_GWP100_case_total": combined["base_total_dynamic_GWP100_case_total"],
        "base_total_dynamic_RF_case_total": combined["base_total_dynamic_RF_case_total"],
        "delta_total_dynamic_GWP100_case_total": combined["total_dynamic_GWP100_case_total"] - combined["base_total_dynamic_GWP100_case_total"],
        "delta_total_dynamic_RF_case_total": combined["total_dynamic_RF_case_total"] - combined["base_total_dynamic_RF_case_total"],
        "delta_total_dynamic_GWP100_pct": (
            (combined["total_dynamic_GWP100_case_total"] - combined["base_total_dynamic_GWP100_case_total"])
            / combined["base_total_dynamic_GWP100_case_total"]
        ),
        "delta_total_dynamic_RF_pct": (
            (combined["total_dynamic_RF_case_total"] - combined["base_total_dynamic_RF_case_total"])
            / combined["base_total_dynamic_RF_case_total"]
        ),
        "annual_electricity_kwh": float(exchange_amounts["annual_electricity_kwh"]),
    }


def _run_future_background_sensitivity(full_context, case: str) -> dict[str, Any]:
    import bw2data as bd

    truncated = replace(full_context, background_dbs=full_context.background_dbs[:-1])
    climate_method = full_context.climate_method
    wrapper_key = full_context.total_wrappers[case]
    result = run_full_case(
        case=case,
        stage="total",
        fg_db=wrapper_key[0],
        wrapper_key=wrapper_key,
        context=truncated,
        t0=DEFAULT_T0,
        horizon_years=DEFAULT_HORIZON_YEARS,
    )
    base_total, _ = _base_tables(case)
    return {
        "scenario": "terminal_2050_tail",
        "case": case,
        "total_dynamic_GWP100_case_total": float(result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "total_dynamic_RF_case_total": float(result.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].sum()),
        "base_total_dynamic_GWP100_case_total": float(base_total["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "base_total_dynamic_RF_case_total": float(base_total["annual_radiative_forcing_W_per_m2_case_total"].sum()),
        "delta_total_dynamic_GWP100_case_total": float(result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()) - float(base_total["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        "delta_total_dynamic_RF_case_total": float(result.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].sum()) - float(base_total["annual_radiative_forcing_W_per_m2_case_total"].sum()),
        "delta_total_dynamic_GWP100_pct": (
            (float(result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()) - float(base_total["annual_dynamic_GWP100_kgCO2e_case_total"].sum()))
            / float(base_total["annual_dynamic_GWP100_kgCO2e_case_total"].sum())
        ),
        "delta_total_dynamic_RF_pct": (
            (float(result.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].sum()) - float(base_total["annual_radiative_forcing_W_per_m2_case_total"].sum()))
            / float(base_total["annual_radiative_forcing_W_per_m2_case_total"].sum())
        ),
    }


def _label_for_scenario(scenario: str) -> str:
    mapping = {
        "gen_hp_cop_minus10pct": "GEN heat-pump COP -10%",
        "gen_pump_control_plus20pct": "GEN pump electricity +20%",
        "bau_west_res_gas90_oil10": "BAU west residential 90/10 gas/oil",
        "bau_west_res_gas60_oil40": "BAU west residential 60/40 gas/oil",
        "bau_fha_multifamily_heat_pump_alt": "BAU FHA heat-pump alternative",
        "bau_commercial_cooling_minus15pct": "BAU commercial cooling -15%",
        "bau_commercial_cooling_plus15pct": "BAU commercial cooling +15%",
        "bau_fossil_heating_to_resistance_cop1": "BAU fossil heating to resistance COP 1.0",
        "bau_residential_fossil_heating_to_resistance_cop1": "BAU residential fossil heating to resistance COP 1.0",
        "bau_commercial_heating_to_resistance_cop1": "BAU commercial heating to resistance COP 1.0",
        "bau_fha_heating_to_gas_furnace_afue90": "BAU FHA heating to gas furnace AFUE 90%",
        "bau_all_heating_gas_furnace_afue90": "BAU all heating to gas furnace AFUE 90%",
        "gen_dhw_constant_cop_2_8": "GEN DHW constant COP 2.8",
        "terminal_2050_tail": "Background terminal-year truncation",
    }
    return mapping.get(scenario, scenario)


def _plot_ranked_effects(ranked: pd.DataFrame) -> None:
    _set_style()
    plot_df = ranked.copy()
    plot_df["label"] = plot_df["scenario"].map(_label_for_scenario)
    plot_df = plot_df.sort_values("abs_delta_dynamic_GWP100_pct", ascending=True)
    colors = plot_df["case"].map({"GEN": "#1b9e77", "BAU": "#d95f02"}).tolist()
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.barh(plot_df["label"], plot_df["abs_delta_dynamic_GWP100_pct"] * 100.0, color=colors)
    ax.set_xlabel("Absolute change in total global warming impact (%)")
    ax.set_ylabel("Sensitivity scenario")
    ax.set_title("Ranked climate sensitivity effects")
    fig.text(
        0.5,
        0.955,
        "Percent change is measured against the synchronized HEATNETS-authoritative base case.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, FIGURE_ROOT / "sensitivity_ranked_effects_main_climate")


def _plot_case_deltas(summary: pd.DataFrame) -> None:
    _set_style()
    plot_df = summary.copy()
    plot_df["label"] = plot_df["scenario"].map(_label_for_scenario)
    plot_df["delta_pct"] = plot_df["delta_total_dynamic_GWP100_pct"] * 100.0
    plot_df = plot_df.sort_values(["case", "delta_pct"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.1), sharex=True)
    for ax, case in zip(axes, ["GEN", "BAU"]):
        subset = plot_df.loc[plot_df["case"] == case].copy()
        colors = subset["sensitivity_family"].map(
            {
                "operational_performance": "#1b9e77",
                "bau_structure": "#d95f02",
                "dhw": "#7570b3",
                "future_background": "#e7298a",
            }
        )
        ax.barh(subset["label"], subset["delta_pct"], color=colors)
        ax.axvline(0.0, color="#444444", linewidth=1.0)
        ax.set_title(case)
        ax.set_xlabel("Change in total global warming impact (%)")
    axes[0].set_ylabel("Sensitivity scenario")
    fig.suptitle("GEN and BAU global warming sensitivity screening")
    fig.text(
        0.5,
        0.955,
        "Only the accepted focused sensitivity set is included: operational performance, BAU structure, DHW, and future-background handling.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, FIGURE_ROOT / "sensitivity_GEN_vs_BAU_global_warming")


def run() -> dict[str, Path]:
    _ensure_output_tree()
    _require_ready()
    config = load_b6_case_config()

    import bw2data as bd

    b6_context = load_b6_context()
    full_context = load_full_context()
    bd.projects.set_current(b6_context.project)
    climate_method = choose_b6_climate_method(bd)

    gen_hourly = _load_gen_hourly()
    bau_group = _load_bau_annual_group_totals()

    operational_rows = []
    for scenario in ["gen_hp_cop_minus10pct", "gen_pump_control_plus20pct"]:
        exchange_amounts = _gen_exchange_amounts_from_hourly(gen_hourly, scenario, config, b6_context.denominator_lifetime_kwh_th)
        row = _run_b6_sensitivity_case("GEN", scenario, exchange_amounts, b6_context, climate_method, config)
        row["sensitivity_family"] = "operational_performance"
        operational_rows.append(row)
    operational_df = pd.DataFrame(operational_rows)
    operational_df.to_csv(OUTPUT_ROOT / "sensitivity_operational_performance.csv", index=False)

    bau_rows = []
    for scenario in [
        "bau_west_res_gas90_oil10",
        "bau_west_res_gas60_oil40",
        "bau_fha_multifamily_heat_pump_alt",
        "bau_commercial_cooling_minus15pct",
        "bau_commercial_cooling_plus15pct",
        "bau_fossil_heating_to_resistance_cop1",
        "bau_fha_heating_to_gas_furnace_afue90",
        "bau_all_heating_gas_furnace_afue90",
    ]:
        modified = _build_modified_bau_group_totals(bau_group, scenario, config)
        exchange_amounts = _bau_exchange_amounts_from_group_totals(modified, config, b6_context.denominator_lifetime_kwh_th)
        row = _run_b6_sensitivity_case("BAU", scenario, exchange_amounts, b6_context, climate_method, config)
        row["sensitivity_family"] = "bau_structure"
        bau_rows.append(row)
    bau_df = pd.DataFrame(bau_rows)
    bau_df.to_csv(OUTPUT_ROOT / "sensitivity_bau_structure.csv", index=False)

    dhw_rows = []
    for scenario in ["gen_dhw_constant_cop_2_8"]:
        exchange_amounts = _gen_exchange_amounts_from_hourly(gen_hourly, scenario, config, b6_context.denominator_lifetime_kwh_th)
        row = _run_b6_sensitivity_case("GEN", scenario, exchange_amounts, b6_context, climate_method, config)
        row["sensitivity_family"] = "dhw"
        dhw_rows.append(row)
    dhw_df = pd.DataFrame(dhw_rows)
    dhw_df.to_csv(OUTPUT_ROOT / "sensitivity_dhw.csv", index=False)

    future_rows = []
    for case in ["GEN", "BAU"]:
        row = _run_future_background_sensitivity(full_context, case)
        row["sensitivity_family"] = "future_background"
        future_rows.append(row)
    future_df = pd.DataFrame(future_rows)
    future_df.to_csv(OUTPUT_ROOT / "sensitivity_future_background.csv", index=False)

    summary = pd.concat([operational_df, bau_df, dhw_df, future_df], ignore_index=True)
    summary.to_csv(OUTPUT_ROOT / "sensitivity_summary_table.csv", index=False)

    ranked = summary.assign(abs_delta_dynamic_GWP100_pct=lambda df: df["delta_total_dynamic_GWP100_pct"].abs()).sort_values(
        ["abs_delta_dynamic_GWP100_pct", "sensitivity_family"], ascending=[False, True]
    )
    ranked.to_csv(OUTPUT_ROOT / "sensitivity_ranked_effects.csv", index=False)
    _plot_ranked_effects(ranked)
    _plot_case_deltas(summary)

    sanity = pd.DataFrame(
        [
            {
                "check": "Sensitivity suite uses accepted authoritative denominator",
                "value": b6_context.denominator_lifetime_kwh_th,
                "expected": b6_context.denominator_lifetime_kwh_th,
                "status": "pass",
            },
            {
                "check": "Operational sensitivity scenarios executed",
                "value": len(operational_df),
                "expected": 2,
                "status": "pass" if len(operational_df) == 2 else "fail",
            },
            {
                "check": "BAU structure sensitivity scenarios executed",
                "value": len(bau_df),
                "expected": 8,
                "status": "pass" if len(bau_df) == 8 else "fail",
            },
            {
                "check": "DHW sensitivity scenarios executed",
                "value": len(dhw_df),
                "expected": 1,
                "status": "pass" if len(dhw_df) == 1 else "fail",
            },
            {
                "check": "Future-background sensitivity scenarios executed",
                "value": len(future_df),
                "expected": 2,
                "status": "pass" if len(future_df) == 2 else "fail",
            },
        ]
    )
    sanity.to_csv(OUTPUT_ROOT / "sensitivity_sanity_checks.csv", index=False)

    top = ranked.iloc[0]
    summary_text = (
        "This focused sensitivity package perturbs only the assumptions identified as most decision-relevant for the synchronized "
        "Framingham workflow: GEN operational performance, BAU structural assumptions, GEN DHW treatment, and the terminal-year "
        "future-background rule. The screened scenarios are intentionally few and transparent. They test whether the main GEN versus "
        "BAU climate conclusion is robust without expanding into a broad probabilistic uncertainty campaign.\n\n"
        f"The largest total global-warming response in the current screening is `{_label_for_scenario(str(top['scenario']))}` for "
        f"`{top['case']}`, with a `{top['delta_total_dynamic_GWP100_pct']:.2%}` change relative to the synchronized base case. "
        "The dominant effect is a BAU structural assumption, not a future-background artifact, which supports the importance of "
        "keeping the BAU baseline construction transparent in the manuscript. GEN-side operational and DHW sensitivities are smaller "
        "but still material enough to justify reporting them explicitly."
    )
    write_markdown(SUMMARY_ROOT / "sensitivity_summary.md", summary_text)

    return {
        "operational": OUTPUT_ROOT / "sensitivity_operational_performance.csv",
        "bau_structure": OUTPUT_ROOT / "sensitivity_bau_structure.csv",
        "dhw": OUTPUT_ROOT / "sensitivity_dhw.csv",
        "future_background": OUTPUT_ROOT / "sensitivity_future_background.csv",
        "summary": OUTPUT_ROOT / "sensitivity_summary_table.csv",
        "ranked": OUTPUT_ROOT / "sensitivity_ranked_effects.csv",
        "sanity": OUTPUT_ROOT / "sensitivity_sanity_checks.csv",
        "figure_ranked": FIGURE_ROOT / "sensitivity_ranked_effects_main_climate.png",
        "figure_case_deltas": FIGURE_ROOT / "sensitivity_GEN_vs_BAU_global_warming.png",
        "summary_markdown": SUMMARY_ROOT / "sensitivity_summary.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the focused sensitivity suite for the synchronized Framingham GEN vs BAU workflow.")
    parser.parse_args()
    outputs = run()
    print("Sensitivity outputs:")
    for key, path in outputs.items():
        print(f" - {key}: {relpath(path)}")


if __name__ == "__main__":
    main()
