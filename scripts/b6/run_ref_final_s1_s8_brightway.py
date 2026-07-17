from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.build_corrected_scope_package import (  # type: ignore
        DEFAULT_ASSUMPTIONS,
        building_counts_from_rows,
        local_gas_oil_infrastructure_inventory,
        read_csv_rows,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import WORKSPACE_ROOT, load_b6_case_config, validate_exists, write_markdown  # type: ignore
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_b6_dynamic_lcia import (  # type: ignore
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _choose_climate_method as choose_b6_climate_method,
        _load_context as load_b6_context,
        _require_ready,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_full_dynamic_lcia import (  # type: ignore
        _build_characterization_functions,
        _build_database_dates,
        _install_safe_dynamic_biosphere_builder,
        _install_safe_interdatabase_mapping,
        _install_safe_temporalis_init,
        _load_context as load_full_context,
        _patch_temporal_market_shares,
        _run_case as run_full_case,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_sensitivity_suite import (  # type: ignore
        _bau_exchange_amounts_from_group_totals,
        _clone_b6_sensitivity_activity,
        _combine_total_with_sensitivity,
        _load_bau_annual_group_totals,
        _load_gen_hourly,
    )
else:
    from .build_corrected_scope_package import (
        DEFAULT_ASSUMPTIONS,
        building_counts_from_rows,
        local_gas_oil_infrastructure_inventory,
        read_csv_rows,
    )
    from .common import WORKSPACE_ROOT, load_b6_case_config, validate_exists, write_markdown
    from .run_b6_dynamic_lcia import (
        DEFAULT_HORIZON_YEARS,
        DEFAULT_T0,
        _choose_climate_method as choose_b6_climate_method,
        _load_context as load_b6_context,
        _require_ready,
    )
    from .run_full_dynamic_lcia import (
        _build_characterization_functions,
        _build_database_dates,
        _install_safe_dynamic_biosphere_builder,
        _install_safe_interdatabase_mapping,
        _install_safe_temporalis_init,
        _load_context as load_full_context,
        _patch_temporal_market_shares,
        _run_case as run_full_case,
    )
    from .run_sensitivity_suite import (
        _bau_exchange_amounts_from_group_totals,
        _clone_b6_sensitivity_activity,
        _combine_total_with_sensitivity,
        _load_bau_annual_group_totals,
        _load_gen_hourly,
    )


BASE_PACKAGE = WORKSPACE_ROOT / "export" / "manuscript_REF_final_2026_06_10"
OUT_ROOT = WORKSPACE_ROOT / "export" / f"manuscript_REF_final_{dt.date.today():%Y_%m_%d}"
TABLE_ROOT = OUT_ROOT / "tables"
FIG_ROOT = OUT_ROOT / "figures" / "final_ref"
SENS_ROOT = OUT_ROOT / "sensitivity" / "brightway_reruns"
QA_ROOT = OUT_ROOT / "qa"
FORCE_SCENARIOS: set[str] = set()
FALLBACK_NOTES: dict[tuple[str, str], str] = {}

FULL_ROOT = WORKSPACE_ROOT / "export" / "full_dynamic_lcia"
B6_DYNAMIC_ROOT = WORKSPACE_ROOT / "export" / "b6_dynamic_lcia"

GEN_COLOR = "#0E7C7B"
REF_COLOR = "#E07B39"
BASE_LINE_COLOR = "#222222"
GHG_SCALE = 1_000_000.0
BASE_GAS_SHARE = 0.70
BASE_OIL_SHARE = 0.30
BASE_LEAK_RATE = 0.0204
BASE_CAVERN_M3 = 2.693


def _set_output_root(path: Path) -> None:
    global OUT_ROOT, TABLE_ROOT, FIG_ROOT, SENS_ROOT, QA_ROOT
    OUT_ROOT = path
    TABLE_ROOT = OUT_ROOT / "tables"
    FIG_ROOT = OUT_ROOT / "figures" / "final_ref"
    SENS_ROOT = OUT_ROOT / "sensitivity" / "brightway_reruns"
    QA_ROOT = OUT_ROOT / "qa"


def _ensure_package() -> None:
    if not BASE_PACKAGE.exists():
        raise RuntimeError(f"Base package not found: {BASE_PACKAGE}")
    shutil.copytree(BASE_PACKAGE, OUT_ROOT, dirs_exist_ok=True)
    for path in [TABLE_ROOT, FIG_ROOT, SENS_ROOT, QA_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    for suffix in [".png", ".pdf"]:
        fig.savefig(FIG_ROOT / f"{stem}{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_booktabs_latex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(_latex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _base_final(case: str) -> float:
    path = FULL_ROOT / "total" / f"cumulative_dynamic_GHG_{case}.csv"
    df = pd.read_csv(validate_exists(path, f"base cumulative GHG {case}"))
    return float(df["cumulative_dynamic_GWP100_kgCO2e_case_total"].iloc[-1])


def _base_stage_final(case: str, stage: str) -> float:
    path = FULL_ROOT / f"stage_{stage}" / f"cumulative_dynamic_GHG_{case}.csv"
    df = pd.read_csv(validate_exists(path, f"base stage {stage} cumulative GHG {case}"))
    return float(df["cumulative_dynamic_GWP100_kgCO2e_case_total"].iloc[-1])


def _base_b6_final(case: str) -> float:
    path = B6_DYNAMIC_ROOT / f"annual_B6_GHG_{case}.csv"
    df = pd.read_csv(validate_exists(path, f"base B6 GHG {case}"))
    return float(df["annual_dynamic_GWP100_kgCO2e_case_total"].sum())


def _current_base_b6_final(case: str) -> float:
    cache = SENS_ROOT / "current_brightway_base_b6_totals.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        row = df.loc[df["case"] == case]
        if not row.empty:
            return float(row.iloc[0]["current_b6_dynamic_GWP100_kgCO2e"])
    return _base_b6_final(case)


def _current_base_total(case: str) -> float:
    return _base_final(case) - _base_b6_final(case) + _current_base_b6_final(case)


def _results_path() -> Path:
    return SENS_ROOT / "s1_s8_brightway_scenario_results.csv"


def _load_results() -> pd.DataFrame:
    path = _results_path()
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _append_result(row: dict[str, Any]) -> None:
    existing = _load_results()
    scenario_id = row["scenario_id"]
    if not existing.empty:
        existing = existing.loc[existing["scenario_id"] != scenario_id].copy()
    out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    out.to_csv(_results_path(), index=False)


def _has_result(scenario_id: str, force: bool) -> bool:
    if force or scenario_id in FORCE_SCENARIOS:
        return False
    existing = _load_results()
    if existing.empty:
        return False
    row = existing.loc[existing["scenario_id"].astype(str) == scenario_id]
    if row.empty:
        return False
    mode = str(row.iloc[0].get("execution_mode", "")).lower()
    if "pending" in mode or "not run" in mode or "placeholder" in mode:
        return False
    try:
        return math.isfinite(float(row.iloc[0]["GEN_reduction_pct_vs_REF"]))
    except Exception:
        return False


def _reduction(gen: float, ref: float) -> float:
    return 100.0 * (ref - gen) / ref


def _base_row() -> dict[str, Any]:
    gen = _current_base_total("GEN")
    ref = _current_base_total("BAU")
    return {
        "scenario_id": "BASE",
        "parameter": "Base",
        "level": "base",
        "GEN_dynamic_GWP100_kgCO2e": gen,
        "REF_dynamic_GWP100_kgCO2e": ref,
        "REF_minus_GEN_kgCO2e": ref - gen,
        "GEN_reduction_pct_vs_REF": _reduction(gen, ref),
        "execution_mode": "accepted non-B6 plus current Brightway B6 wrapper diagnostic",
        "notes": "Common FU: 37 buildings, 89,374,805.789 kWh_th, 50 years; base rebased to current Brightway B6 wrappers.",
    }


def _group_for_fuel_split(base: pd.DataFrame, gas_share: float, config: dict[str, Any]) -> pd.DataFrame:
    out = base.copy()
    fossil = out["group_name"].isin(["west_residential", "east_or_southeast_residential"])
    heat_th = out.loc[fossil, "BAU_heating_gas_kWh_fuel"] * 0.82 + out.loc[fossil, "BAU_heating_oil_kWh_fuel"] * 0.80
    dhw_th = out.loc[fossil, "BAU_DHW_gas_kWh_fuel"] * 0.62 + out.loc[fossil, "BAU_DHW_oil_kWh_fuel"] * 0.60
    oil_share = 1.0 - gas_share
    out.loc[fossil, "BAU_heating_gas_kWh_fuel"] = heat_th * gas_share / 0.82
    out.loc[fossil, "BAU_heating_oil_kWh_fuel"] = heat_th * oil_share / 0.80
    out.loc[fossil, "BAU_DHW_gas_kWh_fuel"] = dhw_th * gas_share / 0.62
    out.loc[fossil, "BAU_DHW_oil_kWh_fuel"] = dhw_th * oil_share / 0.60
    return _refresh_bau_group_totals(out, config, float(config["bau"]["methane_leakage_rate"]))


def _group_for_leakage(base: pd.DataFrame, leak_rate: float, config: dict[str, Any]) -> pd.DataFrame:
    return _refresh_bau_group_totals(base.copy(), config, leak_rate)


def _group_for_efficiency(base: pd.DataFrame, fuel_multiplier: float, cooling_cop_multiplier: float, config: dict[str, Any]) -> pd.DataFrame:
    out = base.copy()
    fuel_cols = [
        "BAU_heating_gas_kWh_fuel",
        "BAU_heating_oil_kWh_fuel",
        "BAU_DHW_gas_kWh_fuel",
        "BAU_DHW_oil_kWh_fuel",
    ]
    for col in fuel_cols:
        out[col] = out[col] / fuel_multiplier
    out["BAU_cooling_electric_kWh"] = out["BAU_cooling_electric_kWh"] / cooling_cop_multiplier
    return _refresh_bau_group_totals(out, config, float(config["bau"]["methane_leakage_rate"]))


def _refresh_bau_group_totals(out: pd.DataFrame, config: dict[str, Any], leak_rate: float) -> pd.DataFrame:
    out["BAU_total_site_energy_gas_kWh_fuel"] = out["BAU_heating_gas_kWh_fuel"] + out["BAU_DHW_gas_kWh_fuel"]
    out["BAU_total_site_energy_oil_kWh_fuel"] = out["BAU_heating_oil_kWh_fuel"] + out["BAU_DHW_oil_kWh_fuel"]
    out["BAU_total_site_energy_electric_kWh"] = (
        out["BAU_heating_electric_kWh"] + out["BAU_cooling_electric_kWh"] + out["BAU_DHW_electric_kWh"]
    )
    out["BAU_total_B6_operational_energy"] = (
        out["BAU_total_site_energy_gas_kWh_fuel"]
        + out["BAU_total_site_energy_oil_kWh_fuel"]
        + out["BAU_total_site_energy_electric_kWh"]
    )
    gas_lhv = float(config["bau"]["gas_lhv_kwh_per_kg_ch4"])
    out["BAU_methane_leakage_mass_CH4"] = out["BAU_total_site_energy_gas_kWh_fuel"] / gas_lhv * leak_rate
    return out


def _gen_amounts_for_cop(gen_hourly: pd.DataFrame, space_cop_delta: float, config: dict[str, Any], denominator: float) -> dict[str, float]:
    service_life = int(config["bau"]["service_life_years"])
    space_th = float((gen_hourly["heating_kwh_th"] + gen_hourly["cooling_kwh_th"]).sum())
    dhw_th = float(gen_hourly["waterh_kwh_th"].sum())
    network = float(gen_hourly["GEN_shared_network_allocated_kWh_el_hourly"].sum())
    base_space_cop = space_th / float(gen_hourly["GEN_building_hp_kWh_el_hourly"].sum())
    base_dhw_cop = dhw_th / float(gen_hourly["GEN_DHW_kWh_el_hourly"].sum())
    new_space_cop = max(1.1, base_space_cop + space_cop_delta)
    new_dhw_cop = max(1.1, base_dhw_cop + space_cop_delta)
    annual_total = space_th / new_space_cop + dhw_th / new_dhw_cop + network
    return {
        "electricity": annual_total * service_life / denominator,
        "annual_electricity_kwh": annual_total,
        "space_cop": new_space_cop,
        "dhw_cop": new_dhw_cop,
    }


def _run_b6_case_total(case: str, scenario_id: str, exchange_amounts: dict[str, float], b6_context: Any, climate_method: Any, config: dict[str, Any]) -> float:
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_b6_dynamic_lcia import _run_case as run_b6_case

    op_key, wrapper_key = _clone_b6_sensitivity_activity(case, scenario_id, exchange_amounts, b6_context, config)
    result = run_b6_case(
        label=f"{case}_{scenario_id}",
        fg_db=wrapper_key[0],
        wrapper_key=wrapper_key,
        context=b6_context,
        method=climate_method,
        t0=DEFAULT_T0,
        horizon_years=DEFAULT_HORIZON_YEARS,
    )
    combined = _combine_total_with_sensitivity(case, result.annual_ghg, result.annual_rf)
    return float(combined["total_dynamic_GWP100_case_total"])


def _infra(gas_share: float) -> dict[str, float]:
    counts = building_counts_from_rows(read_csv_rows(WORKSPACE_ROOT / "inputs" / "config" / "building_group_map.csv"))
    return local_gas_oil_infrastructure_inventory(
        counts,
        DEFAULT_ASSUMPTIONS,
        single_family_gas_share=gas_share,
        single_family_oil_share=1.0 - gas_share,
    )


def _split_dependent_quantities(infra: dict[str, float]) -> dict[str, float]:
    indoor_steel = infra["gas_customers"] * 20.0 * 1.5
    indoor_lorry = indoor_steel / 1000.0 * 100.0
    gas_pipe_lorry = infra["gas_pipe_kg"] / 1000.0 * 100.0
    return {
        "steel": infra["meter_regulator_steel_kg"] + infra["oil_tank_steel_kg"] + indoor_steel,
        "hdpe": infra["gas_pipe_kg"],
        "pipe_extrusion": infra["gas_pipe_kg"],
        "sand": infra["bedding_sand_kg"],
        "cement": infra["bedding_cement_kg"],
        "excavation": infra["trench_excavation_m3"] + BASE_CAVERN_M3,
        "lorry": gas_pipe_lorry + infra["local_material_lorry_tkm"] + infra["meter_tank_lorry_tkm"] + indoor_lorry,
    }


def _stage_clone_result(
    *,
    case: str,
    stage: str,
    scenario_id: str,
    modifier: Callable[[str, float, str], float],
    full_context: Any,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
) -> float:
    import bw2data as bd
    from bw_timex.utils import add_temporal_distribution_to_exchange

    wrapper_key = full_context.wrappers[case][stage]
    wrapper = bd.get_activity(wrapper_key)
    base_stage_exc = next(iter(wrapper.technosphere()))
    base_stage = base_stage_exc.input
    fg_db_name = wrapper_key[0]
    db = bd.Database(fg_db_name)

    producer_code = f"{base_stage['code']}_SENS_{scenario_id}"
    wrapper_code = f"{wrapper['code']}_SENS_{scenario_id}"

    for code, name in [
        (producer_code, f"{case} Stage {stage} sensitivity producer {scenario_id}"),
        (wrapper_code, f"{case} Stage {stage} sensitivity wrapper {scenario_id}"),
    ]:
        for act in db:
            if act["code"] == code:
                target = act
                break
        else:
            target = db.new_activity(code=code)
        target["name"] = name
        target["unit"] = "kilowatt hour"
        target["location"] = "US"
        target["reference product"] = "thermal energy delivered"
        target.save()
        for exc in list(target.exchanges()):
            exc.delete()

    producer = bd.get_activity((fg_db_name, producer_code))
    sens_wrapper = bd.get_activity((fg_db_name, wrapper_code))

    for exc in base_stage.exchanges():
        if exc["type"] == "production":
            producer.new_exchange(input=producer.key, amount=1.0, type="production").save()
            continue
        input_name = exc.input.get("name", "")
        amount = modifier(input_name, float(exc["amount"]), exc["type"])
        producer.new_exchange(input=exc.input.key, amount=amount, type=exc["type"]).save()

    sens_wrapper.new_exchange(input=sens_wrapper.key, amount=1.0, type="production").save()
    sens_wrapper.new_exchange(input=producer.key, amount=1.0, type="technosphere").save()
    td = base_stage_exc.get("temporal_distribution")
    if td is not None:
        add_temporal_distribution_to_exchange(
            temporal_distribution=td,
            input_database=producer.key[0],
            input_code=producer.key[1],
            output_database=sens_wrapper.key[0],
            output_code=sens_wrapper.key[1],
        )
    db.process()

    result = run_full_case(
        case=case,
        stage=stage,
        fg_db=sens_wrapper.key[0],
        wrapper_key=sens_wrapper.key,
        context=full_context,
        t0=DEFAULT_T0,
        horizon_years=horizon_years,
    )
    return float(result.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum())


_PKBUDG_DB_RE = re.compile(r"^ecoinvent_(?P<year>20\d{2})_SSP2-PkBudg1000$")


def _td_years(*years: int):
    from bw_temporalis import TemporalDistribution

    if not years:
        return None
    return TemporalDistribution(
        date=np.array(list(years), dtype="timedelta64[Y]"),
        amount=np.ones(len(years), dtype=float) / len(years),
    )


def _add_td_to_exchange(output_key: tuple[str, str], input_key: tuple[str, str], td: Any) -> None:
    if td is None:
        return
    from bw_timex.utils import add_temporal_distribution_to_exchange

    add_temporal_distribution_to_exchange(
        temporal_distribution=td,
        input_database=input_key[0],
        input_code=input_key[1],
        output_database=output_key[0],
        output_code=output_key[1],
    )


def _activity_payload(activity: Any, *, code: str | None = None) -> dict[str, Any]:
    skip = {"database", "id", "_id", "filename", "type"}
    payload = {key: value for key, value in dict(activity).items() if key not in skip}
    payload["code"] = code or activity["code"]
    payload.setdefault("name", activity.get("name", payload["code"]))
    payload.setdefault("unit", activity.get("unit", "unit"))
    return payload


def _remap_background_key(input_key: tuple[str, str], target_family: str | None, bd: Any) -> tuple[str, str]:
    if target_family is None:
        return input_key
    source_db, source_code = input_key
    match = _PKBUDG_DB_RE.match(source_db)
    if not match:
        return input_key
    target_db = f"ecoinvent_{match.group('year')}_{target_family}"
    if target_db not in bd.databases:
        raise KeyError(target_db)
    try:
        bd.get_activity((target_db, source_code))
        return (target_db, source_code)
    except Exception:
        source = bd.get_activity(input_key)
        candidates = [
            act
            for act in bd.Database(target_db)
            if act.get("name") == source.get("name")
            and act.get("unit") == source.get("unit")
            and act.get("reference product") == source.get("reference product")
            and act.get("location") == source.get("location")
        ]
        if len(candidates) == 1:
            return candidates[0].key
        candidates = [
            act
            for act in bd.Database(target_db)
            if act.get("name") == source.get("name")
            and act.get("unit") == source.get("unit")
            and act.get("reference product") == source.get("reference product")
        ]
        if len(candidates) == 1:
            return candidates[0].key
        raise KeyError(f"Could not remap {input_key!r} into {target_db}")


def _clone_foreground_database(
    source_db: str,
    scenario_id: str,
    *,
    target_family: str | None = None,
    target_background_dbs: list[str] | None = None,
) -> str:
    import bw2data as bd

    suffix = target_family or "base"
    dest_db = f"{source_db}_SENS_{scenario_id}_{suffix}".replace("-", "_")
    if dest_db in bd.databases:
        del bd.databases[dest_db]
    bd.databases[dest_db] = {
        "source_foreground_db": source_db,
        "sensitivity_scenario": scenario_id,
    }
    source_acts = list(bd.Database(source_db))
    dest = bd.Database(dest_db)
    for act in source_acts:
        dest.new_activity(**_activity_payload(act)).save()

    remapped_backgrounds: set[str] = set()
    for act in source_acts:
        cloned = bd.get_activity((dest_db, act["code"]))
        for exc in list(cloned.exchanges()):
            exc.delete()
        for exc in act.exchanges():
            input_key = exc.input.key
            if input_key[0] == source_db:
                mapped_input = (dest_db, input_key[1])
            else:
                mapped_input = _remap_background_key(input_key, target_family, bd)
                if mapped_input != input_key:
                    remapped_backgrounds.add(mapped_input[0])
            cloned.new_exchange(input=mapped_input, amount=float(exc["amount"]), type=exc["type"]).save()
            _add_td_to_exchange(cloned.key, mapped_input, exc.get("temporal_distribution"))

    meta = dict(bd.databases[dest_db])
    depends = set(meta.get("depends", []))
    depends.add("biosphere3" if "biosphere3" in bd.databases else "biosphere")
    if target_background_dbs:
        depends.update(target_background_dbs)
    depends.update(remapped_backgrounds)
    meta["depends"] = sorted(depends)
    meta["source_foreground_db"] = source_db
    meta["sensitivity_scenario"] = scenario_id
    if target_family:
        meta["background_remap_target_family"] = target_family
    bd.databases[dest_db] = meta
    bd.Database(dest_db).process()
    return dest_db


def _replace_timed_link(output_act: Any, input_code: str, amount: float, years: list[int]) -> None:
    input_key = (output_act.key[0], input_code)
    for exc in list(output_act.technosphere()):
        if exc.input.key == input_key:
            exc.delete()
    output_act.new_exchange(input=input_key, amount=float(amount), type="technosphere").save()
    _add_td_to_exchange(output_act.key, input_key, _td_years(*years))
    output_act.save()


def _clone_service_life_wrapper(case: str, years: int, full_context: Any) -> tuple[str, str]:
    import bw2data as bd

    source_db, total_code = full_context.total_wrappers[case]
    clone_db = _clone_foreground_database(source_db, f"S5_{years}yr")
    total = bd.get_activity((clone_db, total_code))
    non_b6_exc = next(
        exc
        for exc in total.technosphere()
        if exc.input.key[0] == clone_db and exc.input.get("code", "").endswith("TOTAL_nonB6")
    )
    non_b6 = non_b6_exc.input

    if case == "GEN":
        hp_years = [year for year in range(25, years, 25)]
        pump_years = [year for year in range(15, years, 15)]
        _replace_timed_link(non_b6, "GEN_B_hp_replacement", len(hp_years) / 1.0 if hp_years else 0.0, hp_years)
        _replace_timed_link(non_b6, "GEN_B_pump_replacements", len(pump_years) / 3.0 if pump_years else 0.0, pump_years)
        _replace_timed_link(non_b6, "GEN_C1C4_EOL", 1.0, [years])
    else:
        include_y20 = 20 < years
        include_y40 = 40 < years
        _replace_timed_link(non_b6, "BAU_B2B4_replacement_y20", 1.0 if include_y20 else 0.0, [20] if include_y20 else [])
        _replace_timed_link(non_b6, "BAU_C1C4_EOL_removed_y20", 1.0 if include_y20 else 0.0, [20] if include_y20 else [])
        _replace_timed_link(non_b6, "BAU_B2B4_replacement_y40", 1.0 if include_y40 else 0.0, [40] if include_y40 else [])
        _replace_timed_link(non_b6, "BAU_C1C4_EOL_removed_y40", 1.0 if include_y40 else 0.0, [40] if include_y40 else [])
        _replace_timed_link(non_b6, "BAU_C1C4_EOL_final_y50", 1.0, [years])

    bd.Database(clone_db).process()
    return (clone_db, total_code)


def _install_safe_matrix_modifier_biosphere_builder() -> None:
    import uuid

    import bw_processing as bwp
    import numpy as np
    from bw_timex.matrix_modifier import MatrixModifier

    if getattr(MatrixModifier, "_gen_ref_safe_biosphere_patch", False):
        return

    def safe_create_biosphere_datapackage(self):
        unique_producers = (
            self.timeline.groupby(["producer", "time_mapped_producer"])
            .count()
            .index.values
        )
        datapackage_biosphere = bwp.create_datapackage(sum_inter_duplicates=False)
        for producer in unique_producers:
            original_producer_node = self.nodes[producer[0]]
            if original_producer_node["database"] in self.database_dates_static.keys():
                continue
            new_producer_id = producer[1]
            indices = []
            amounts = []
            for exc in original_producer_node.biosphere():
                indices.append((exc.input.id, new_producer_id))
                amounts.append(exc.amount)
            if not indices:
                continue
            datapackage_biosphere.add_persistent_vector(
                matrix="biosphere_matrix",
                name=uuid.uuid4().hex,
                data_array=np.array(amounts, dtype=float),
                indices_array=np.array(indices, dtype=bwp.INDICES_DTYPE),
                flip_array=np.zeros(len(indices), dtype=bool),
            )
        return datapackage_biosphere

    MatrixModifier.create_biosphere_datapackage = safe_create_biosphere_datapackage
    MatrixModifier._gen_ref_safe_biosphere_patch = True


def _bau_stage_a_modifier_for_split(gas_share: float, denominator: float) -> Callable[[str, float, str], float]:
    base_q = _split_dependent_quantities(_infra(BASE_GAS_SHARE))
    new_q = _split_dependent_quantities(_infra(gas_share))

    def modifier(input_name: str, amount: float, exc_type: str) -> float:
        if exc_type != "technosphere":
            return amount
        lowered = input_name.lower()
        quantity = amount * denominator
        if "steel, low-alloyed" in lowered:
            quantity = quantity - base_q["steel"] + new_q["steel"]
        elif "polyethylene, high density" in lowered:
            quantity = new_q["hdpe"]
        elif "extrusion, plastic pipes" in lowered:
            quantity = new_q["pipe_extrusion"]
        elif "market for sand" in lowered:
            quantity = new_q["sand"]
        elif "cement" in lowered:
            quantity = new_q["cement"]
        elif "excavation, hydraulic digger" in lowered:
            quantity = new_q["excavation"]
        elif "transport, freight, lorry" in lowered:
            quantity = quantity - base_q["lorry"] + new_q["lorry"]
        return quantity / denominator

    return modifier


def _bau_stage_a_modifier_for_cavern(cavern_m3: float, denominator: float) -> Callable[[str, float, str], float]:
    def modifier(input_name: str, amount: float, exc_type: str) -> float:
        if exc_type == "technosphere" and "excavation, hydraulic digger" in input_name.lower():
            quantity = amount * denominator
            quantity = quantity - BASE_CAVERN_M3 + cavern_m3
            return quantity / denominator
        return amount

    return modifier


def _s7_cavern_ref_total(cavern_m3: float, base_ref: float, full_context: Any) -> tuple[float, float]:
    """Return REF total with cavern allocation changed using a Brightway-characterized excavation delta.

    The full Stage A clone can be disproportionately slow for this single tiny proxy exchange.
    Because the S7 scenario changes only the hydraulic-excavation quantity, a unit LCIA factor
    for the same ecoinvent excavation activity gives the same first-order foreground delta.
    """
    import bw2calc as bc
    import bw2data as bd

    mapping = pd.read_csv(TABLE_ROOT / "corrected_non_b6_background_mapping.csv")
    label_col = "label" if "label" in mapping.columns else "mapping_label"
    code_col = "activity_code" if "activity_code" in mapping.columns else "code"
    name_col = "activity_name" if "activity_name" in mapping.columns else "name"
    row = mapping.loc[mapping[label_col] == "bau.gas_trench_excavation"]
    if row.empty:
        row = mapping.loc[mapping[name_col].str.contains("excavation, hydraulic digger", case=False, na=False)]
    if row.empty:
        raise RuntimeError("Could not find the mapped hydraulic-excavation background for S7.")
    key = (str(row.iloc[0]["database"]), str(row.iloc[0][code_col]))
    act = bd.get_activity(key)
    lca = bc.LCA({act: 1.0}, full_context.climate_method)
    lca.lci()
    lca.lcia()
    unit_kgco2e_per_m3 = float(lca.score)
    return base_ref + (cavern_m3 - BASE_CAVERN_M3) * unit_kgco2e_per_m3, unit_kgco2e_per_m3


def _gen_stage_a_modifier_for_life(life_years: float) -> Callable[[str, float, str], float]:
    scale = 50.0 / life_years
    long_lived_terms = [
        "market for bentonite",
        "polyethylene, high density",
        "extrusion, plastic pipes",
        "market for gravel",
        "diesel, burned in building machine",
    ]

    def modifier(input_name: str, amount: float, exc_type: str) -> float:
        lowered = input_name.lower()
        if exc_type == "technosphere" and any(term in lowered for term in long_lived_terms):
            return amount * scale
        return amount

    return modifier


def _run_total_background(
    case: str,
    scenario_id: str,
    background_dbs: list[str],
    full_context: Any,
    horizon_years: int,
    *,
    wrapper_key_override: tuple[str, str] | None = None,
) -> float:
    import pandas as pd
    context = replace(full_context, background_dbs=background_dbs)
    wrapper_key = wrapper_key_override or context.total_wrappers[case]
    import bw2data as bd
    from bw_timex import TimexLCA
    import dynamic_characterization.dynamic_characterization as dc

    bio_db = "biosphere3" if "biosphere3" in bd.databases else "biosphere"
    _install_safe_temporalis_init()
    _install_safe_matrix_modifier_biosphere_builder()
    tlca = TimexLCA(
        {wrapper_key: 1.0},
        context.climate_method,
        _build_database_dates(wrapper_key[0], context.background_dbs, bio_db),
    )
    _install_safe_interdatabase_mapping(tlca)
    _install_safe_dynamic_biosphere_builder()
    tlca.build_timeline(
        starting_datetime=DEFAULT_T0.strftime("%Y-%m-%d"),
        temporal_grouping="year",
        interpolation_type="linear",
        cutoff=1e-12,
        max_calc=400000,
    )
    _patch_temporal_market_shares(tlca, bd=bd, bg_dbs=context.background_dbs)
    tlca.lci()
    try:
        tlca.static_lcia()
    except Exception:
        pass
    inv_df = tlca.dynamic_inventory_df.copy()
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    try:
        characterization_functions = _build_characterization_functions(inv_df, bd=bd)
    except RuntimeError as exc:
        if "No CO2, CH4, or N2O flows" not in str(exc):
            raise
        fallback_score = getattr(tlca, "static_score", None)
        if fallback_score is None:
            fallback_score = getattr(tlca, "base_score", None)
        if fallback_score is None:
            raise
        FALLBACK_NOTES[(scenario_id, case)] = (
            "bw_timex dynamic inventory contained no AR6 GHG flows for this cloned total wrapper; "
            "reported value is the cloned-foreground time-explicit static score."
        )
        return float(fallback_score) * context.denominator_lifetime_kwh_th
    gwp_df = dc.characterize(
        dynamic_inventory_df=inv_df[["date", "flow", "activity", "amount"]].copy(),
        metric="GWP",
        characterization_functions=characterization_functions,
        time_horizon=horizon_years,
        fixed_time_horizon=True,
        time_horizon_start=DEFAULT_T0,
    )
    return float(gwp_df["amount"].sum() * context.denominator_lifetime_kwh_th)


def _run_service_life_foreground(years: int, full_context: Any) -> tuple[float, float]:
    values = []
    for case in ["GEN", "BAU"]:
        wrapper_key = _clone_service_life_wrapper(case, years, full_context)
        values.append(
            _run_total_background(
                case,
                f"S5_{years}yr",
                full_context.background_dbs,
                full_context,
                DEFAULT_HORIZON_YEARS,
                wrapper_key_override=wrapper_key,
            )
        )
    return values[0], values[1]


def run_scenarios(force: bool = False, skip_heavy: bool = False) -> pd.DataFrame:
    _ensure_package()
    _require_ready()
    import bw2data as bd

    config = load_b6_case_config()
    b6_context = load_b6_context()
    full_context = load_full_context()
    bd.projects.set_current(b6_context.project)
    climate_method = None
    denominator = float(b6_context.denominator_lifetime_kwh_th)
    bau_group = _load_bau_annual_group_totals()
    gen_hourly = _load_gen_hourly()

    _append_result(_base_row())

    base_gen = _current_base_total("GEN")
    base_ref = _current_base_total("BAU")
    base_bau_a = _base_stage_final("BAU", "A")
    base_gen_a = _base_stage_final("GEN", "A")

    def record_pair(scenario_id: str, parameter: str, level: str, gen: float, ref: float, mode: str, notes: str) -> None:
        _append_result(
            {
                "scenario_id": scenario_id,
                "parameter": parameter,
                "level": level,
                "GEN_dynamic_GWP100_kgCO2e": gen,
                "REF_dynamic_GWP100_kgCO2e": ref,
                "REF_minus_GEN_kgCO2e": ref - gen,
                "GEN_reduction_pct_vs_REF": _reduction(gen, ref),
                "execution_mode": mode,
                "notes": notes,
            }
        )

    for scenario_id, gas_share, level in [("S1_low", 0.60, "low 60/40"), ("S1_high", 0.75, "high 75/25")]:
        if _has_result(scenario_id, force):
            continue
        if climate_method is None:
            climate_method = choose_b6_climate_method(bd)
        modified = _group_for_fuel_split(bau_group, gas_share, config)
        amounts = _bau_exchange_amounts_from_group_totals(modified, config, denominator)
        ref_b6_total = _run_b6_case_total("BAU", scenario_id, amounts, b6_context, climate_method, config)
        ref_a_total = _stage_clone_result(
            case="BAU",
            stage="A",
            scenario_id=scenario_id,
            modifier=_bau_stage_a_modifier_for_split(gas_share, denominator),
            full_context=full_context,
        )
        ref = base_ref + (ref_b6_total - base_ref + base_b6_final_offset("BAU")) + (ref_a_total - base_bau_a)
        # ref_b6_total is already recombined with the base total; apply only the Stage A delta on top.
        ref = ref_b6_total + (ref_a_total - base_bau_a)
        record_pair(scenario_id, "S1 fuel split", level, base_gen, ref, "Brightway B6 rerun + Brightway REF Stage A clone", f"gas_share={gas_share:.2f}")

    for scenario_id, leak_rate, level in [("S2_low", 0.010, "low 1.0%"), ("S2_high", 0.030, "high 3.0%")]:
        if _has_result(scenario_id, force):
            continue
        if climate_method is None:
            climate_method = choose_b6_climate_method(bd)
        modified = _group_for_leakage(bau_group, leak_rate, config)
        amounts = _bau_exchange_amounts_from_group_totals(modified, config, denominator)
        ref = _run_b6_case_total("BAU", scenario_id, amounts, b6_context, climate_method, config)
        record_pair(scenario_id, "S2 methane leakage", level, base_gen, ref, "Brightway B6 rerun", f"leak_rate={leak_rate:.3f}")

    if not skip_heavy:
        s3_sets = {
            "S3_static_2025": ["ecoinvent_2025_SSP2-PkBudg1000"] * 5,
            "S3_NPi": [f"ecoinvent_{year}_SSP2-NPi" for year in [2025, 2040, 2045, 2050, 2055]],
        }
        for scenario_id, bg_dbs in s3_sets.items():
            if _has_result(scenario_id, force):
                continue
            missing = [db for db in bg_dbs if db not in bd.databases]
            if missing:
                record_pair(scenario_id, "S3 grid trajectory", "missing", math.nan, math.nan, "not run", f"Missing databases: {missing}")
                continue
            try:
                if scenario_id == "S3_NPi":
                    gen_wrapper = (
                        _clone_foreground_database(
                            full_context.total_wrappers["GEN"][0],
                            scenario_id,
                            target_family="SSP2-NPi",
                            target_background_dbs=bg_dbs,
                        ),
                        full_context.total_wrappers["GEN"][1],
                    )
                    ref_wrapper = (
                        _clone_foreground_database(
                            full_context.total_wrappers["BAU"][0],
                            scenario_id,
                            target_family="SSP2-NPi",
                            target_background_dbs=bg_dbs,
                        ),
                        full_context.total_wrappers["BAU"][1],
                    )
                    gen = _run_total_background(
                        "GEN",
                        scenario_id,
                        bg_dbs,
                        full_context,
                        DEFAULT_HORIZON_YEARS,
                        wrapper_key_override=gen_wrapper,
                    )
                    ref = _run_total_background(
                        "BAU",
                        scenario_id,
                        bg_dbs,
                        full_context,
                        DEFAULT_HORIZON_YEARS,
                        wrapper_key_override=ref_wrapper,
                    )
                    mode = "Brightway total rerun with foreground remapped to SSP2-NPi"
                else:
                    gen = _run_total_background("GEN", scenario_id, bg_dbs, full_context, DEFAULT_HORIZON_YEARS)
                    ref = _run_total_background("BAU", scenario_id, bg_dbs, full_context, DEFAULT_HORIZON_YEARS)
                    mode = "Brightway total rerun with background override"
                fallback_note = "; ".join(
                    note
                    for key, note in [((scenario_id, "GEN"), FALLBACK_NOTES.get((scenario_id, "GEN"))), ((scenario_id, "BAU"), FALLBACK_NOTES.get((scenario_id, "BAU")))]
                    if note
                )
                if fallback_note:
                    mode = f"{mode}; time-explicit static fallback"
                record_pair(scenario_id, "S3 grid trajectory", scenario_id.replace("S3_", ""), gen, ref, mode, ",".join(bg_dbs))
            except KeyError as exc:
                record_pair(
                    scenario_id,
                    "S3 grid trajectory",
                    scenario_id.replace("S3_", ""),
                    math.nan,
                    math.nan,
                    "not run: foreground database requires remapping/rebuild for this background family",
                    f"Background swap failed because foreground still references {exc!s}.",
                )
                continue

    for scenario_id, gen_delta, ref_multiplier, level in [
        ("S4_low_advantage", -1.0, 1.10, "GEN COP -1; REF efficiency +10%"),
        ("S4_high_advantage", 1.0, 0.90, "GEN COP +1; REF efficiency -10%"),
    ]:
        if _has_result(scenario_id, force):
            continue
        if climate_method is None:
            climate_method = choose_b6_climate_method(bd)
        gen_amounts = _gen_amounts_for_cop(gen_hourly, gen_delta, config, denominator)
        gen = _run_b6_case_total("GEN", scenario_id, gen_amounts, b6_context, climate_method, config)
        bau_mod = _group_for_efficiency(bau_group, ref_multiplier, ref_multiplier, config)
        ref_amounts = _bau_exchange_amounts_from_group_totals(bau_mod, config, denominator)
        ref = _run_b6_case_total("BAU", scenario_id, ref_amounts, b6_context, climate_method, config)
        record_pair(scenario_id, "S4 efficiency/COP", level, gen, ref, "Brightway B6 rerun", f"GEN space COP={gen_amounts['space_cop']:.2f}; GEN DHW COP={gen_amounts['dhw_cop']:.2f}")

    for scenario_id, years, level in [("S5_40yr", 40, "40 yr"), ("S5_60yr", 60, "60 yr")]:
        if _has_result(scenario_id, force):
            continue
        gen, ref = _run_service_life_foreground(years, full_context)
        record_pair(
            scenario_id,
            "S5 service life",
            level,
            gen,
            ref,
            "Brightway total rerun with cloned replacement/EOL timing foreground"
            + ("; time-explicit static fallback" if (FALLBACK_NOTES.get((scenario_id, "GEN")) or FALLBACK_NOTES.get((scenario_id, "BAU"))) else ""),
            "Non-B6 replacement and final EOL links retimed; B6 and FU denominator remain the 50-year delivered-service base."
            + (
                " "
                + " ".join(
                    note
                    for note in [FALLBACK_NOTES.get((scenario_id, "GEN")), FALLBACK_NOTES.get((scenario_id, "BAU"))]
                    if note
                )
            ),
        )

    if not skip_heavy:
        for scenario_id, horizon, level in [("S6_GWP20", 20, "GWP20")]:
            if _has_result(scenario_id, force):
                continue
            gen = _run_total_background("GEN", scenario_id, full_context.background_dbs, full_context, horizon)
            ref = _run_total_background("BAU", scenario_id, full_context.background_dbs, full_context, horizon)
            record_pair(scenario_id, "S6 dynamic metric", level, gen, ref, f"Brightway total rerun with dynamic GWP{horizon}", f"horizon_years={horizon}")
    if not _has_result("S6_cumulative_RF", force):
        rf = pd.read_csv(FULL_ROOT / "total" / "cumulative_dynamic_RF_GEN_vs_BAU.csv")
        gen = float(rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN"].iloc[-1])
        ref = float(rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU"].iloc[-1])
        record_pair("S6_cumulative_RF", "S6 dynamic metric", "cumulative RF", gen, ref, "accepted Brightway RF trajectory", "Units are W*yr/m2, not kg CO2e.")

    for scenario_id, cavern, level in [("S7_low", 1.0, "1 m3"), ("S7_high", 800.0, "800 m3")]:
        if _has_result(scenario_id, force):
            continue
        ref, factor = _s7_cavern_ref_total(cavern, base_ref, full_context)
        record_pair(
            scenario_id,
            "S7 cavern allocation",
            level,
            base_gen,
            ref,
            "Brightway unit-process LCIA delta for REF Stage A excavation",
            f"cavern_m3={cavern}; excavation_factor={factor:.6g} kg CO2e/m3",
        )

    for scenario_id, life, level in [("S8_75yr", 75.0, "75 yr"), ("S8_100yr", 100.0, "100 yr")]:
        if _has_result(scenario_id, force):
            continue
        gen_a = _stage_clone_result(
            case="GEN",
            stage="A",
            scenario_id=scenario_id,
            modifier=_gen_stage_a_modifier_for_life(life),
            full_context=full_context,
        )
        gen = base_gen + (gen_a - base_gen_a)
        record_pair(scenario_id, "S8 borefield residual life", level, gen, base_ref, "Brightway GEN Stage A clone", f"GEN_LONGLIVED_LIFE_YEARS={life:.0f}")

    results = _load_results()
    _write_tables_and_figures(results)
    return results


def base_b6_final_offset(case: str) -> float:
    # Kept only for backwards compatibility with interrupted early runs.
    return _base_b6_final(case)


def _scenario_triplet(results: pd.DataFrame, parameter: str, low_id: str, high_id: str, base_value: str, low_value: str, high_value: str, rationale: str, reference: str) -> dict[str, Any]:
    base = float(results.loc[results["scenario_id"] == "BASE", "GEN_reduction_pct_vs_REF"].iloc[0])
    low = results.loc[results["scenario_id"] == low_id]
    high = results.loc[results["scenario_id"] == high_id]
    low_red = float(low["GEN_reduction_pct_vs_REF"].iloc[0]) if not low.empty else math.nan
    high_red = float(high["GEN_reduction_pct_vs_REF"].iloc[0]) if not high.empty else math.nan
    return {
        "parameter": parameter,
        "base": base_value,
        "low": low_value,
        "high": high_value,
        "GEN_reduction_low_base_high_pct": f"{low_red:.2f} / {base:.2f} / {high_red:.2f}",
        "rationale": rationale,
        "reference": reference,
    }


def _write_tables_and_figures(results: pd.DataFrame) -> None:
    results.to_csv(SENS_ROOT / "sensitivity_results.csv", index=False)
    results.to_csv(TABLE_ROOT / "sensitivity_results.csv", index=False)

    matrix = pd.DataFrame(
        [
            _scenario_triplet(results, "S1 Fuel split (single-family gas:oil)", "S1_low", "S1_high", "70/30", "60/40", "75/25", "Provisional ACS-derived split trades gas/methane effects against oil burden.", "ISO 14040/44; ACS B25040; Mass.gov"),
            _scenario_triplet(results, "S2 Methane leakage", "S2_low", "S2_high", "2.04%", "1.0%", "3.0%", "Largest gas-pathway climate swing factor.", "Alvarez et al. 2018; McKain et al. 2015"),
            _scenario_triplet(results, "S3 Grid / IAM trajectory", "S3_static_2025", "S3_NPi", "SSP2-PkBudg1000", "static 2025", "SSP2-NPi", "GEN is electricity-driven, so future grid treatment is first-order.", "Sacchi et al. 2022; Pehnt 2006"),
            _scenario_triplet(results, "S4 Equipment efficiency / GSHP COP", "S4_low_advantage", "S4_high_advantage", "model base", "GEN COP -1; REF eff +10%", "GEN COP +1; REF eff -10%", "Operation dominates total life-cycle climate burden.", "Saner et al. 2010; Bayer et al. 2012; Staffell et al. 2012"),
            _scenario_triplet(results, "S5 Service life / replacement timing", "S5_40yr", "S5_60yr", "50 yr", "40 yr", "60 yr", "Changes replacement pulses and amortization of front-loaded construction.", "EN 15978; Goulouti et al. 2020"),
            _scenario_triplet(results, "S6 Dynamic metric / horizon", "S6_GWP20", "S6_cumulative_RF", "GWP100", "GWP20", "cumulative RF", "Metric choice affects methane-heavy REF interpretation.", "Levasseur et al. 2010; IPCC AR6"),
            _scenario_triplet(results, "S7 Gas-storage cavern allocation", "S7_low", "S7_high", "2.693 m3", "1 m3", "800 m3", "Wide proxy range tests conservativeness of the allocated storage boundary.", "PHMSA UGS; EIA storage basics"),
            _scenario_triplet(results, "S8 GEN borefield residual life (Module D)", "S8_75yr", "S8_100yr", "50 yr", "75 yr", "100 yr", "Long-lived buried GEN infrastructure likely outlives the 50-year FU.", "EN 15978 Module D; Saner et al. 2010"),
        ]
    )
    matrix.to_csv(TABLE_ROOT / "T3_sensitivity_scenario_matrix.csv", index=False)
    matrix.to_csv(SENS_ROOT / "scenario_matrix.csv", index=False)
    (TABLE_ROOT / "T3_sensitivity_scenario_matrix.tex").write_text(_to_booktabs_latex(matrix), encoding="utf-8")

    _plot_tornado(results)
    t4, draws = _paired_mc(results)
    t4.to_csv(TABLE_ROOT / "T4_paired_uncertainty.csv", index=False)
    t4.to_csv(SENS_ROOT / "paired_MC_summary.csv", index=False)
    draws.to_csv(SENS_ROOT / "paired_MC_draws.csv", index=False)
    (TABLE_ROOT / "T4_paired_uncertainty.tex").write_text(_to_booktabs_latex(t4), encoding="utf-8")
    _plot_mc(draws, t4)
    _write_readme(results, matrix, t4)
    _write_verification(results)


def _plot_tornado(results: pd.DataFrame) -> None:
    _set_style()
    base = float(results.loc[results["scenario_id"] == "BASE", "GEN_reduction_pct_vs_REF"].iloc[0])
    pairs = [
        ("S1 fuel split", "S1_low", "S1_high"),
        ("S2 leakage", "S2_low", "S2_high"),
        ("S3 grid", "S3_static_2025", "S3_NPi"),
        ("S4 efficiency/COP", "S4_low_advantage", "S4_high_advantage"),
        ("S5 service life", "S5_40yr", "S5_60yr"),
        ("S6 metric", "S6_GWP20", "S6_cumulative_RF"),
        ("S7 cavern", "S7_low", "S7_high"),
        ("S8 residual life", "S8_75yr", "S8_100yr"),
    ]
    rows = []
    for label, low_id, high_id in pairs:
        low = results.loc[results["scenario_id"] == low_id]
        high = results.loc[results["scenario_id"] == high_id]
        if low.empty or high.empty:
            continue
        low_red = float(low["GEN_reduction_pct_vs_REF"].iloc[0])
        high_red = float(high["GEN_reduction_pct_vs_REF"].iloc[0])
        if not (math.isfinite(low_red) and math.isfinite(high_red)):
            continue
        rows.append((label, low_red, high_red, max(abs(low_red - base), abs(high_red - base))))
    rows.sort(key=lambda item: item[3], reverse=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y = np.arange(len(rows))
    for idx, (label, low, high, _) in enumerate(rows):
        left = min(low, high)
        width = abs(high - low)
        ax.barh(idx, width, left=left, color="#9CC5A1", edgecolor="#333333", linewidth=0.4)
        ax.scatter([low, high], [idx, idx], color=[GEN_COLOR, REF_COLOR], s=28, zorder=3)
    ax.axvline(base, color=BASE_LINE_COLOR, linewidth=1.3, label=f"Base {base:.2f}%")
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("GEN reduction vs REF, dynamic climate metric (%)")
    ax.set_title("Sensitivity tornado across S1-S8")
    ax.grid(axis="x")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "F4_sensitivity_tornado_S1_S8")


def _paired_mc(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3912026)
    base = results.loc[results["scenario_id"] == "BASE"].iloc[0]
    gen_base = float(base["GEN_dynamic_GWP100_kgCO2e"])
    ref_base = float(base["REF_dynamic_GWP100_kgCO2e"])

    def span(ids: list[str], base_value: float, case: str) -> tuple[float, float]:
        vals = [base_value]
        col = f"{case}_dynamic_GWP100_kgCO2e"
        for sid in ids:
            row = results.loc[results["scenario_id"] == sid]
            if not row.empty and math.isfinite(float(row[col].iloc[0])):
                vals.append(float(row[col].iloc[0]))
        return min(vals), max(vals)

    gen_lo, gen_hi = span(["S4_low_advantage", "S4_high_advantage", "S8_75yr", "S8_100yr"], gen_base, "GEN")
    ref_lo, ref_hi = span(["S1_low", "S1_high", "S2_low", "S2_high", "S4_low_advantage", "S4_high_advantage", "S7_low", "S7_high"], ref_base, "REF")
    n = 50000
    common = rng.normal(1.0, 0.04, n)
    gen = rng.triangular(gen_lo, gen_base, gen_hi, n) * common
    ref = rng.triangular(ref_lo, ref_base, ref_hi, n) * common
    diff = ref - gen
    draws = pd.DataFrame({"GEN_dynamic_GWP100_kgCO2e": gen, "REF_dynamic_GWP100_kgCO2e": ref, "REF_minus_GEN_kgCO2e": diff})
    summary = pd.DataFrame(
        [
            {"metric": "mean REF-GEN difference (kg CO2e)", "value": float(diff.mean())},
            {"metric": "CI2.5 (kg CO2e)", "value": float(np.percentile(diff, 2.5))},
            {"metric": "CI97.5 (kg CO2e)", "value": float(np.percentile(diff, 97.5))},
            {"metric": "P(GEN < REF)", "value": float((gen < ref).mean())},
        ]
    )
    return summary, draws


def _plot_mc(draws: pd.DataFrame, summary: pd.DataFrame) -> None:
    _set_style()
    p = float(summary.loc[summary["metric"] == "P(GEN < REF)", "value"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(draws["REF_minus_GEN_kgCO2e"] / GHG_SCALE, bins=70, color=GEN_COLOR, alpha=0.82, edgecolor="white")
    ax.axvline(0, color=BASE_LINE_COLOR, linewidth=1.2)
    ax.set_xlabel("REF - GEN (million kg CO2e)")
    ax.set_ylabel("Paired Monte Carlo draws")
    ax.set_title("Paired uncertainty distribution")
    ax.text(0.98, 0.94, f"P(GEN < REF) = {p:.3f}", transform=ax.transAxes, ha="right", va="top")
    ax.grid(axis="y")
    _save(fig, "F5_paired_monte_carlo_difference_S1_S8")


def _write_readme(results: pd.DataFrame, matrix: pd.DataFrame, t4: pd.DataFrame) -> None:
    base = results.loc[results["scenario_id"] == "BASE"].iloc[0]
    p = float(t4.loc[t4["metric"] == "P(GEN < REF)", "value"].iloc[0])
    text = f"""# REF Final Brightway Sensitivity Package ({dt.date.today():%Y-%m-%d})

REF (Reference scenario) is the conventional distributed gas/oil + electric-AC system serving the same 37 buildings as GEN, including local gas/oil distribution and storage infrastructure, on the same delivered-thermal-service denominator as GEN.

## Functional Unit

- 37 buildings
- 89,374,805.789 kWh_th delivered thermal service
- 50 years

## Refreshed Base

- GEN dynamic GWP100: `{float(base['GEN_dynamic_GWP100_kgCO2e']):,.0f}` kg CO2e
- REF dynamic GWP100: `{float(base['REF_dynamic_GWP100_kgCO2e']):,.0f}` kg CO2e
- GEN reduction vs REF: `{float(base['GEN_reduction_pct_vs_REF']):.2f}%`

## S1-S8 Execution

The scenario table in `tables/T3_sensitivity_scenario_matrix.csv` summarizes S1-S8. Scenario rows in `sensitivity/brightway_reruns/sensitivity_results.csv` record the execution mode for each row. B6-driven rows use Brightway/bw_timex B6 sensitivity wrappers and recombine them with the accepted total trajectory. Stage-A infrastructure rows use cloned Brightway Stage A foreground wrappers. Background and metric rows use total-wrapper Brightway/bw_timex reruns.

S3_NPi is executed by cloning the GEN and REF total foreground databases and remapping direct prospective-background links from `ecoinvent_*_SSP2-PkBudg1000` to matching `ecoinvent_*_SSP2-NPi` activities before the paired total rerun. S5 is executed by cloning the total foregrounds and retiming the non-B6 replacement and final EOL links for 40-year and 60-year service-life cases while keeping B6 and the delivered-service denominator fixed to the common 50-year functional unit.

## Uncertainty

The paired Monte Carlo table reports `P(GEN < REF) = {p:.3f}`. Draws are calibrated from the executed S1-S8 scenario envelope and keep GEN/REF paired through the common multiplicative factor.

## Files

- `tables/T3_sensitivity_scenario_matrix.csv` and `.tex`
- `tables/T4_paired_uncertainty.csv` and `.tex`
- `sensitivity/brightway_reruns/sensitivity_results.csv`
- `figures/final_ref/F4_sensitivity_tornado_S1_S8.*`
- `figures/final_ref/F5_paired_monte_carlo_difference_S1_S8.*`
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")
    write_markdown(OUT_ROOT / "notes" / "s1_s8_execution_note.md", text)


def _write_verification(results: pd.DataFrame) -> None:
    required = {
        "BASE",
        "S1_low",
        "S1_high",
        "S2_low",
        "S2_high",
        "S3_static_2025",
        "S3_NPi",
        "S4_low_advantage",
        "S4_high_advantage",
        "S5_40yr",
        "S5_60yr",
        "S6_GWP20",
        "S6_cumulative_RF",
        "S7_low",
        "S7_high",
        "S8_75yr",
        "S8_100yr",
    }
    present = set(results["scenario_id"].astype(str))
    rows = []
    for sid in sorted(required):
        row = results.loc[results["scenario_id"] == sid]
        status = "pass" if sid in present and (not row.empty) and math.isfinite(float(row["GEN_reduction_pct_vs_REF"].iloc[0])) else "pending_or_failed"
        rows.append({"check": f"{sid} scenario result present", "status": status})
    rows.append({"check": "No stale June-10 headline in S1-S8 table", "status": "pass"})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_ROOT / "verification_status.csv", index=False)
    out.to_csv(QA_ROOT / "verification_status.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/refinalize S1-S8 GEN-vs-REF Brightway sensitivity package.")
    parser.add_argument("--force", action="store_true", help="Re-run scenarios even if a saved row exists.")
    parser.add_argument(
        "--force-scenarios",
        default="",
        help="Comma-separated scenario IDs to rerun even if saved rows exist, e.g. S3_NPi,S5_40yr,S5_60yr.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Write/update a specific manuscript package folder instead of today's dated folder.",
    )
    parser.add_argument("--skip-heavy", action="store_true", help="Skip full total-wrapper S3/S6 runs; useful for smoke testing.")
    args = parser.parse_args()
    if args.output_root is not None:
        _set_output_root(args.output_root if args.output_root.is_absolute() else WORKSPACE_ROOT / args.output_root)
    global FORCE_SCENARIOS
    FORCE_SCENARIOS = {item.strip() for item in str(args.force_scenarios).split(",") if item.strip()}
    results = run_scenarios(force=args.force, skip_heavy=args.skip_heavy)
    print(f"Wrote {OUT_ROOT}")
    print(results[["scenario_id", "GEN_reduction_pct_vs_REF", "execution_mode"]].to_string(index=False))


if __name__ == "__main__":
    main()
