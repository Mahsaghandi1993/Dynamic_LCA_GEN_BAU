from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        EXPORT_ROOT,
        ensure_workspace_tree,
        legacy_operation_root,
        load_archetype_map,
        load_b6_case_config,
        load_bau_baseline_config,
        load_fuel_share_defaults,
        read_excel_any,
        relpath,
        validate_exists,
    )
else:
    from .common import (
        EXPORT_ROOT,
        ensure_workspace_tree,
        legacy_operation_root,
        load_archetype_map,
        load_b6_case_config,
        load_bau_baseline_config,
        load_fuel_share_defaults,
        read_excel_any,
        relpath,
        validate_exists,
    )


KBtu_TO_KWH = 0.29307107
HOURS_PER_YEAR = 8760


@dataclass
class ArchetypeMetrics:
    target_key: str
    source_family: str
    dhw_profile: pd.Series
    heating_profile: pd.Series | None
    cooling_profile: pd.Series | None
    coefficients: dict[str, float]
    dhw_to_heating_ratio: float | None
    annual_dhw_per_unit: float | None
    annual_dhw_per_sqft: float | None


def _aggregate_interval_series(series: pd.Series, steps_per_hour: int) -> pd.Series:
    values = series.astype(float).to_numpy()
    usable = len(values) - (len(values) % steps_per_hour)
    values = values[:usable]
    return pd.Series(values.reshape(-1, steps_per_hour).sum(axis=1))


def _resstock_metrics(target_key: str, baseline_path: Path, geothermal_path: Path) -> ArchetypeMetrics:
    _ = geothermal_path
    baseline_cols = {
        "timestamp",
        "units_represented",
        "out.load.heating.energy_delivered.kbtu",
        "out.load.cooling.energy_delivered.kbtu",
        "out.load.hot_water.energy_delivered.kbtu",
        "out.electricity.heating.energy_consumption.kwh",
        "out.electricity.cooling.energy_consumption.kwh",
        "out.electricity.hot_water.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.hot_water.energy_consumption.kwh",
        "out.fuel_oil.heating.energy_consumption.kwh",
        "out.fuel_oil.hot_water.energy_consumption.kwh",
    }
    df = pd.read_csv(baseline_path, usecols=lambda c: c in baseline_cols)
    steps_per_hour = max(1, len(df) // HOURS_PER_YEAR)
    heating = _aggregate_interval_series(df["out.load.heating.energy_delivered.kbtu"] * KBtu_TO_KWH, steps_per_hour)
    cooling = _aggregate_interval_series(df["out.load.cooling.energy_delivered.kbtu"] * KBtu_TO_KWH, steps_per_hour)
    dhw = _aggregate_interval_series(df["out.load.hot_water.energy_delivered.kbtu"] * KBtu_TO_KWH, steps_per_hour)

    heating_total = float(heating.sum())
    cooling_total = float(cooling.sum())
    dhw_total = float(dhw.sum())
    units_represented = float(df["units_represented"].iloc[0]) if "units_represented" in df.columns else 1.0

    def coeff(column: str, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(df[column].sum()) / denominator

    coefficients = {
        "natural_gas_heating": coeff("out.natural_gas.heating.energy_consumption.kwh", heating_total),
        "fuel_oil_heating": coeff("out.fuel_oil.heating.energy_consumption.kwh", heating_total),
        "electricity_heating": coeff("out.electricity.heating.energy_consumption.kwh", heating_total),
        "electricity_cooling": coeff("out.electricity.cooling.energy_consumption.kwh", cooling_total),
        "natural_gas_dhw": coeff("out.natural_gas.hot_water.energy_consumption.kwh", dhw_total),
        "fuel_oil_dhw": coeff("out.fuel_oil.hot_water.energy_consumption.kwh", dhw_total),
        "electricity_dhw": coeff("out.electricity.hot_water.energy_consumption.kwh", dhw_total),
    }

    return ArchetypeMetrics(
        target_key=target_key,
        source_family="resstock",
        dhw_profile=(dhw / dhw_total) if dhw_total > 0 else pd.Series(np.zeros(HOURS_PER_YEAR)),
        heating_profile=(heating / heating_total) if heating_total > 0 else None,
        cooling_profile=(cooling / cooling_total) if cooling_total > 0 else None,
        coefficients=coefficients,
        dhw_to_heating_ratio=(dhw_total / heating_total) if heating_total > 0 else None,
        annual_dhw_per_unit=(dhw_total / units_represented) if units_represented > 0 else None,
        annual_dhw_per_sqft=None,
    )


def _comstock_metrics(target_key: str, baseline_path: Path, geothermal_path: Path) -> ArchetypeMetrics:
    base_cols = {
        "timestamp",
        "floor_area_represented",
        "out.electricity.cooling.energy_consumption.kwh",
        "out.electricity.water_systems.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.natural_gas.water_systems.energy_consumption.kwh",
        "out.other_fuel.heating.energy_consumption.kwh",
        "out.other_fuel.water_systems.energy_consumption.kwh",
    }
    geo_cols = {
        "timestamp",
        "floor_area_represented",
        "out.district_cooling.cooling.energy_consumption.kwh",
        "out.district_heating.heating.energy_consumption.kwh",
        "out.district_heating.water_systems.energy_consumption.kwh",
    }
    base = pd.read_csv(baseline_path, usecols=lambda c: c in base_cols)
    geo = pd.read_csv(geothermal_path, usecols=lambda c: c in geo_cols)
    steps_per_hour = max(1, len(base) // HOURS_PER_YEAR)

    heating = _aggregate_interval_series(geo["out.district_heating.heating.energy_consumption.kwh"], steps_per_hour)
    cooling = _aggregate_interval_series(geo["out.district_cooling.cooling.energy_consumption.kwh"], steps_per_hour)
    dhw = _aggregate_interval_series(geo["out.district_heating.water_systems.energy_consumption.kwh"], steps_per_hour)

    heating_total = float(heating.sum())
    cooling_total = float(cooling.sum())
    dhw_total = float(dhw.sum())
    floor_area = float(base["floor_area_represented"].iloc[0]) if "floor_area_represented" in base.columns else 0.0

    def coeff(column: str, denominator: float, source: pd.DataFrame) -> float:
        if denominator <= 0:
            return 0.0
        return float(source[column].sum()) / denominator

    coefficients = {
        "natural_gas_heating": coeff("out.natural_gas.heating.energy_consumption.kwh", heating_total, base),
        "fuel_oil_heating": coeff("out.other_fuel.heating.energy_consumption.kwh", heating_total, base),
        "electricity_cooling": coeff("out.electricity.cooling.energy_consumption.kwh", cooling_total, base),
        "natural_gas_dhw": coeff("out.natural_gas.water_systems.energy_consumption.kwh", dhw_total, base),
        "fuel_oil_dhw": coeff("out.other_fuel.water_systems.energy_consumption.kwh", dhw_total, base),
        "electricity_dhw": coeff("out.electricity.water_systems.energy_consumption.kwh", dhw_total, base),
        "electricity_heating": 0.0,
    }

    return ArchetypeMetrics(
        target_key=target_key,
        source_family="comstock",
        dhw_profile=(dhw / dhw_total) if dhw_total > 0 else pd.Series(np.zeros(HOURS_PER_YEAR)),
        heating_profile=(heating / heating_total) if heating_total > 0 else None,
        cooling_profile=(cooling / cooling_total) if cooling_total > 0 else None,
        coefficients=coefficients,
        dhw_to_heating_ratio=(dhw_total / heating_total) if heating_total > 0 else None,
        annual_dhw_per_unit=None,
        annual_dhw_per_sqft=(dhw_total / floor_area) if floor_area > 0 else None,
    )


def _load_archetype_metrics() -> dict[str, ArchetypeMetrics]:
    op_root = legacy_operation_root(raw=True)
    mapping = load_archetype_map()
    metrics: dict[str, ArchetypeMetrics] = {}
    for row in mapping.to_dict(orient="records"):
        baseline_path = validate_exists(op_root / row["baseline_relpath"], "archetype baseline file")
        geothermal_path = validate_exists(op_root / row["geothermal_relpath"], "archetype geothermal file")
        if row["source_family"] == "resstock":
            metrics[row["target_key"]] = _resstock_metrics(row["target_key"], baseline_path, geothermal_path)
        elif row["source_family"] == "comstock":
            metrics[row["target_key"]] = _comstock_metrics(row["target_key"], baseline_path, geothermal_path)
        else:
            raise ValueError(f"Unsupported source family `{row['source_family']}` for target `{row['target_key']}`")
    return metrics


def _resolve_group_config(group_name: str, groups_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(groups_config[group_name])
    parent = config.get("inherit_from")
    if not parent:
        return config
    merged = _resolve_group_config(parent, groups_config)
    for key, value in config.items():
        if key == "inherit_from":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _fallback_dhw_total(metrics: ArchetypeMetrics, floor_area: float | None) -> float:
    if metrics.annual_dhw_per_unit is not None:
        return metrics.annual_dhw_per_unit
    if metrics.annual_dhw_per_sqft is not None and floor_area is not None and floor_area > 0:
        return metrics.annual_dhw_per_sqft * floor_area
    return 0.0


def _calc_energy_from_mode(
    load_kwh_th: float,
    mode_cfg: dict[str, Any],
    metrics: ArchetypeMetrics,
    fuel_shares: dict[str, Any],
    end_use: str,
) -> dict[str, float]:
    output = {
        "natural_gas": 0.0,
        "fuel_oil": 0.0,
        "electricity": 0.0,
    }
    if load_kwh_th <= 0:
        return output

    coefficient_mode = str(mode_cfg.get("coefficient_mode", "archetype")).lower()

    def selected_coeff(fuel_key: str) -> tuple[float, str, float]:
        archetype_coeff = float(metrics.coefficients.get(f"{fuel_key}_{end_use}", 0.0) or 0.0)
        if fuel_key == "electricity":
            fallback_coeff = 1.0 / float(mode_cfg.get("fallback_cop", 1.0))
        else:
            fallback_cfg = mode_cfg.get("fallback_efficiency", 1.0)
            if isinstance(fallback_cfg, dict):
                fallback_eff = float(fallback_cfg.get(fuel_key, 1.0))
            else:
                fallback_eff = float(fallback_cfg)
            fallback_coeff = 1.0 / fallback_eff

        if coefficient_mode == "explicit_efficiency":
            return fallback_coeff, "explicit_efficiency", archetype_coeff
        if archetype_coeff > 0:
            return archetype_coeff, "archetype", archetype_coeff
        return fallback_coeff, "fallback", archetype_coeff

    if "fuels" in mode_cfg:
        share_cfg = fuel_shares[mode_cfg["fuel_share_key"]]
        split = {
            "natural_gas": float(share_cfg.get("gas_share", 0.0)),
            "fuel_oil": float(share_cfg.get("oil_share", 0.0)),
        }
        gas_coeff, _, _ = selected_coeff("natural_gas")
        oil_coeff, _, _ = selected_coeff("fuel_oil")
        output["natural_gas"] = load_kwh_th * split["natural_gas"] * gas_coeff
        output["fuel_oil"] = load_kwh_th * split["fuel_oil"] * oil_coeff
        return output

    fuel = mode_cfg.get("fuel")
    if fuel == "electricity":
        if mode_cfg.get("mode") == "electric_resistance":
            eff = float(mode_cfg.get("efficiency", 1.0))
            output["electricity"] = load_kwh_th / eff
        else:
            coeff, _, _ = selected_coeff("electricity")
            output["electricity"] = load_kwh_th * coeff
        return output

    if fuel == "natural_gas":
        coeff, _, _ = selected_coeff("natural_gas")
        output["natural_gas"] = load_kwh_th * coeff
        return output

    if fuel == "fuel_oil":
        coeff, _, _ = selected_coeff("fuel_oil")
        output["fuel_oil"] = load_kwh_th * coeff
        return output

    return output


def _conversion_records(
    building_id: str,
    group_name: str,
    metrics: ArchetypeMetrics,
    heating_cfg: dict[str, Any],
    cooling_cfg: dict[str, Any],
    dhw_cfg: dict[str, Any],
    fuel_shares: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def archetype_coeff(end_use: str, fuel_key: str) -> float:
        return float(metrics.coefficients.get(f"{fuel_key}_{end_use}", 0.0) or 0.0)

    def add_record(
        end_use: str,
        fuel_key: str,
        selected_coeff: float,
        source: str,
        archetype_value: float,
        explicit_value: float,
        share: float = 1.0,
    ) -> None:
        implied = (1.0 / selected_coeff) if selected_coeff > 0 else np.nan
        metric_name = "implied_efficiency_or_cop"
        records.append(
            {
                "building_id": building_id,
                "group_name": group_name,
                "source_family": metrics.source_family,
                "end_use": end_use,
                "fuel_or_energy": fuel_key,
                "coefficient_mode": str(
                    {"heating": heating_cfg, "cooling": cooling_cfg, "dhw": dhw_cfg}[end_use].get("coefficient_mode", "archetype")
                ),
                "selected_coefficient_kwh_input_per_kwh_th": selected_coeff,
                "selected_source": source,
                "archetype_coefficient_kwh_input_per_kwh_th": archetype_value,
                "explicit_coefficient_kwh_input_per_kwh_th": explicit_value,
                "fuel_share": share,
                metric_name: implied,
            }
        )

    def resolve_coeff(mode_cfg: dict[str, Any], end_use: str, fuel_key: str) -> tuple[float, str, float]:
        coefficient_mode = str(mode_cfg.get("coefficient_mode", "archetype")).lower()
        arch = archetype_coeff(end_use, fuel_key)
        if fuel_key == "electricity":
            explicit = 1.0 / float(mode_cfg.get("fallback_cop", 1.0))
        else:
            fallback_cfg = mode_cfg.get("fallback_efficiency", 1.0)
            if isinstance(fallback_cfg, dict):
                explicit = 1.0 / float(fallback_cfg.get(fuel_key, 1.0))
            else:
                explicit = 1.0 / float(fallback_cfg)
        if coefficient_mode == "explicit_efficiency":
            return explicit, "explicit_efficiency", arch
        if arch > 0:
            return arch, "archetype", arch
        return explicit, "fallback", arch

    for end_use, mode_cfg in [("heating", heating_cfg), ("cooling", cooling_cfg), ("dhw", dhw_cfg)]:
        if "fuels" in mode_cfg:
            share_cfg = fuel_shares[mode_cfg["fuel_share_key"]]
            for fuel_key, share_key in [("natural_gas", "gas_share"), ("fuel_oil", "oil_share")]:
                coeff, source, arch = resolve_coeff(mode_cfg, end_use, fuel_key)
                if fuel_key == "natural_gas":
                    fallback_cfg = mode_cfg.get("fallback_efficiency", {})
                    explicit = 1.0 / float(fallback_cfg.get("natural_gas", 1.0))
                else:
                    fallback_cfg = mode_cfg.get("fallback_efficiency", {})
                    explicit = 1.0 / float(fallback_cfg.get("fuel_oil", 1.0))
                add_record(end_use, fuel_key, coeff, source, arch, explicit, float(share_cfg.get(share_key, 0.0)))
        else:
            fuel_key = str(mode_cfg.get("fuel", "electricity"))
            if mode_cfg.get("mode") == "electric_resistance":
                coeff = 1.0 / float(mode_cfg.get("efficiency", 1.0))
                add_record(end_use, fuel_key, coeff, "explicit_efficiency", archetype_coeff(end_use, fuel_key), coeff)
            else:
                coeff, source, arch = resolve_coeff(mode_cfg, end_use, fuel_key)
                if fuel_key == "electricity":
                    explicit = 1.0 / float(mode_cfg.get("fallback_cop", 1.0))
                else:
                    explicit = 1.0 / float(mode_cfg.get("fallback_efficiency", 1.0))
                add_record(end_use, fuel_key, coeff, source, arch, explicit)
    return records


def rebuild() -> dict[str, Path]:
    ensure_workspace_tree()
    case_config = load_b6_case_config()
    bau_config = load_bau_baseline_config()
    fuel_shares = load_fuel_share_defaults()

    service_path = validate_exists(EXPORT_ROOT / "gen" / "heatnets_service_loads_hourly_building.csv", "HEATNETS service-load export")
    service_loads = pd.read_csv(service_path, parse_dates=["timestamp"])
    loop_components = read_excel_any(legacy_operation_root(raw=True).parent / "HEET_LeGUp_git_Framingham" / "InputFiles" / "loop_components_info.xlsx")
    loop_components = loop_components[["ID", "floor_area"]].rename(columns={"ID": "building_id"})

    service_loads = service_loads.merge(loop_components, on="building_id", how="left")

    metrics_lookup = _load_archetype_metrics()
    groups_cfg = bau_config["groups"]
    leakage_rate = float(case_config["bau"]["methane_leakage_rate"])
    gas_lhv = float(case_config["bau"]["gas_lhv_kwh_per_kg_ch4"])

    common_rows: list[pd.DataFrame] = []
    bau_rows: list[pd.DataFrame] = []
    coeff_records: list[dict[str, Any]] = []
    conversion_records: list[dict[str, Any]] = []

    for building_id, frame in service_loads.groupby("building_id", sort=True):
        first = frame.iloc[0]
        group_name = str(first["group_name"])
        metrics = metrics_lookup.get(str(building_id)) or metrics_lookup.get(group_name)
        if metrics is None:
            raise KeyError(
                f"No archetype metrics were found for building `{building_id}` in group `{group_name}`. "
                "Check `inputs/config/archetype_mapping.csv`."
            )
        group_cfg = _resolve_group_config(group_name, groups_cfg)

        annual_heating = float(frame["heating_kwh_th"].sum())
        annual_dhw = (
            annual_heating * metrics.dhw_to_heating_ratio
            if metrics.dhw_to_heating_ratio is not None and annual_heating > 0
            else _fallback_dhw_total(metrics, float(first["floor_area"]) if not pd.isna(first["floor_area"]) else None)
        )
        dhw_hourly = metrics.dhw_profile.to_numpy() * annual_dhw

        common = frame[["timestamp", "building_id", "building_name", "group_name", "heating_kwh_th", "cooling_kwh_th"]].copy()
        common["waterh_kwh_th"] = dhw_hourly
        common["delivered_total_kwh_th"] = common["heating_kwh_th"] + common["cooling_kwh_th"] + common["waterh_kwh_th"]
        common_rows.append(common)

        bau = common.copy()
        heating_cfg = group_cfg["heating"]
        cooling_cfg = group_cfg["cooling"]
        dhw_cfg = group_cfg["dhw"]

        bau["BAU_heating_gas_kWh_fuel"] = 0.0
        bau["BAU_heating_oil_kWh_fuel"] = 0.0
        bau["BAU_heating_electric_kWh"] = 0.0
        bau["BAU_cooling_electric_kWh"] = 0.0
        bau["BAU_DHW_gas_kWh_fuel"] = 0.0
        bau["BAU_DHW_oil_kWh_fuel"] = 0.0
        bau["BAU_DHW_electric_kWh"] = 0.0

        for idx, row in bau.iterrows():
            heating_split = _calc_energy_from_mode(float(row["heating_kwh_th"]), heating_cfg, metrics, fuel_shares, "heating")
            cooling_split = _calc_energy_from_mode(float(row["cooling_kwh_th"]), cooling_cfg, metrics, fuel_shares, "cooling")
            dhw_split = _calc_energy_from_mode(float(row["waterh_kwh_th"]), dhw_cfg, metrics, fuel_shares, "dhw")

            bau.at[idx, "BAU_heating_gas_kWh_fuel"] = heating_split["natural_gas"]
            bau.at[idx, "BAU_heating_oil_kWh_fuel"] = heating_split["fuel_oil"]
            bau.at[idx, "BAU_heating_electric_kWh"] = heating_split["electricity"]
            bau.at[idx, "BAU_cooling_electric_kWh"] = cooling_split["electricity"]
            bau.at[idx, "BAU_DHW_gas_kWh_fuel"] = dhw_split["natural_gas"]
            bau.at[idx, "BAU_DHW_oil_kWh_fuel"] = dhw_split["fuel_oil"]
            bau.at[idx, "BAU_DHW_electric_kWh"] = dhw_split["electricity"]

        bau["BAU_total_site_energy_gas_kWh_fuel"] = bau["BAU_heating_gas_kWh_fuel"] + bau["BAU_DHW_gas_kWh_fuel"]
        bau["BAU_total_site_energy_oil_kWh_fuel"] = bau["BAU_heating_oil_kWh_fuel"] + bau["BAU_DHW_oil_kWh_fuel"]
        bau["BAU_total_site_energy_electric_kWh"] = (
            bau["BAU_heating_electric_kWh"] + bau["BAU_cooling_electric_kWh"] + bau["BAU_DHW_electric_kWh"]
        )
        bau["BAU_methane_leakage_mass_CH4"] = bau["BAU_total_site_energy_gas_kWh_fuel"] / gas_lhv * leakage_rate
        bau["BAU_total_B6_operational_energy"] = (
            bau["BAU_total_site_energy_gas_kWh_fuel"]
            + bau["BAU_total_site_energy_oil_kWh_fuel"]
            + bau["BAU_total_site_energy_electric_kWh"]
        )
        bau_rows.append(bau)

        coeff_record = {
            "target_key": metrics.target_key,
            "building_id": building_id,
            "group_name": group_name,
            "source_family": metrics.source_family,
            "dhw_to_heating_ratio": metrics.dhw_to_heating_ratio,
        }
        coeff_record.update(metrics.coefficients)
        coeff_records.append(coeff_record)
        conversion_records.extend(
            _conversion_records(
                building_id=building_id,
                group_name=group_name,
                metrics=metrics,
                heating_cfg=heating_cfg,
                cooling_cfg=cooling_cfg,
                dhw_cfg=dhw_cfg,
                fuel_shares=fuel_shares,
            )
        )

    common_service = pd.concat(common_rows, ignore_index=True)
    bau_hourly = pd.concat(bau_rows, ignore_index=True)

    bau_group_hourly = (
        bau_hourly.groupby(["timestamp", "group_name"], as_index=False)[
            [
                "BAU_heating_gas_kWh_fuel",
                "BAU_heating_oil_kWh_fuel",
                "BAU_heating_electric_kWh",
                "BAU_cooling_electric_kWh",
                "BAU_DHW_gas_kWh_fuel",
                "BAU_DHW_oil_kWh_fuel",
                "BAU_DHW_electric_kWh",
                "BAU_methane_leakage_mass_CH4",
                "BAU_total_site_energy_gas_kWh_fuel",
                "BAU_total_site_energy_oil_kWh_fuel",
                "BAU_total_site_energy_electric_kWh",
                "BAU_total_B6_operational_energy",
                "delivered_total_kwh_th",
            ]
        ]
        .sum()
        .sort_values(["timestamp", "group_name"])
    )

    bau_annual = (
        bau_hourly.groupby(["building_id", "building_name", "group_name"], as_index=False)[
            [
                "BAU_heating_gas_kWh_fuel",
                "BAU_heating_oil_kWh_fuel",
                "BAU_heating_electric_kWh",
                "BAU_cooling_electric_kWh",
                "BAU_DHW_gas_kWh_fuel",
                "BAU_DHW_oil_kWh_fuel",
                "BAU_DHW_electric_kWh",
                "BAU_methane_leakage_mass_CH4",
                "BAU_total_site_energy_gas_kWh_fuel",
                "BAU_total_site_energy_oil_kWh_fuel",
                "BAU_total_site_energy_electric_kWh",
                "BAU_total_B6_operational_energy",
                "delivered_total_kwh_th",
            ]
        ]
        .sum()
        .sort_values(["group_name", "building_id"])
    )

    common_out = EXPORT_ROOT / "bau" / "common_service_loads_hourly_building.csv"
    bau_out = EXPORT_ROOT / "bau" / "bau_b6_hourly_building.csv"
    group_out = EXPORT_ROOT / "bau" / "bau_b6_hourly_group.csv"
    annual_out = EXPORT_ROOT / "bau" / "bau_b6_annual_summary.csv"
    coeff_out = EXPORT_ROOT / "bau" / "bau_archetype_coefficients.csv"
    conversion_out = EXPORT_ROOT / "bau" / "bau_energy_conversion_summary.csv"

    common_service.to_csv(common_out, index=False)
    bau_hourly.to_csv(bau_out, index=False)
    bau_group_hourly.to_csv(group_out, index=False)
    bau_annual.to_csv(annual_out, index=False)
    pd.DataFrame(coeff_records).drop_duplicates().to_csv(coeff_out, index=False)
    pd.DataFrame(conversion_records).drop_duplicates().to_csv(conversion_out, index=False)

    return {
        "common_service_loads": common_out,
        "bau_hourly": bau_out,
        "bau_group_hourly": group_out,
        "bau_annual": annual_out,
        "coefficients": coeff_out,
        "conversion_summary": conversion_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild BAU hourly Stage B6 energy from ResStock and ComStock archetypes.")
    parser.parse_args()
    outputs = rebuild()
    for label, path in outputs.items():
        print(f"{label}: {relpath(path)}")


if __name__ == "__main__":
    main()
