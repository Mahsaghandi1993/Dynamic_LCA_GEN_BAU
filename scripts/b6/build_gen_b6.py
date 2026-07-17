from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        EXPORT_ROOT,
        heatnets_source_root,
        load_b6_case_config,
        load_group_map,
        read_excel_any,
        relpath,
        validate_exists,
    )
else:
    from .common import (
        EXPORT_ROOT,
        heatnets_source_root,
        load_b6_case_config,
        load_group_map,
        read_excel_any,
        relpath,
        validate_exists,
    )


TON_TO_KW = 3.517


def _load_cop_curves(cop_workbook: Path) -> dict[str, pd.DataFrame]:
    sheets = ["Commercial", "Versatec", "YorkSFH", "Climatemaster"]
    curves = {}
    for sheet in sheets:
        frame = read_excel_any(cop_workbook, sheet_name=sheet, skiprows=5)
        frame = frame[["inlet_temp", "Heating_COP", "Cooling_COP"]].copy().dropna()
        frame["inlet_temp"] = pd.to_numeric(frame["inlet_temp"])
        frame["Heating_COP"] = pd.to_numeric(frame["Heating_COP"])
        curves[sheet] = frame.sort_values("inlet_temp")
    return curves


def _heating_cop(curve: pd.DataFrame, inlet_temp: pd.Series, fallback_cop: float) -> pd.Series:
    clipped = inlet_temp.astype(float).clip(lower=float(curve["inlet_temp"].min()), upper=float(curve["inlet_temp"].max()))
    values = np.interp(clipped, curve["inlet_temp"], curve["Heating_COP"])
    values = np.where(np.isfinite(values) & (values > 0), values, fallback_cop)
    return pd.Series(values, index=inlet_temp.index)


def build() -> dict[str, Path]:
    config = load_b6_case_config()
    heatnets_hourly_path = validate_exists(
        EXPORT_ROOT / "gen" / "gen_heatnets_component_electricity_hourly.csv",
        "validated HEATNETS component electricity export",
    )
    common_service_path = validate_exists(
        EXPORT_ROOT / "bau" / "common_service_loads_hourly_building.csv",
        "common Stage B6 service-load export",
    )
    component_hourly = pd.read_csv(heatnets_hourly_path, parse_dates=["timestamp"])
    common_service = pd.read_csv(common_service_path, parse_dates=["timestamp"])

    group_map = load_group_map()[["building_id", "group_name", "heatnets_hp_type"]]
    loop_components = read_excel_any(heatnets_source_root(raw=True) / "InputFiles" / "loop_components_info.xlsx")
    loop_components = loop_components[["ID", "HP_total_size_tons", "HP_type"]].rename(columns={"ID": "building_id"})
    loop_components["HP_total_size_tons"] = pd.to_numeric(loop_components["HP_total_size_tons"], errors="coerce")

    building_ids = [col for col in component_hourly.columns if col.startswith(("C", "R"))]
    building_hp_long = component_hourly[["timestamp"] + building_ids].melt(
        id_vars="timestamp",
        var_name="building_id",
        value_name="GEN_building_hp_kWh_el_hourly",
    )

    building_hourly = common_service.merge(building_hp_long, on=["timestamp", "building_id"], how="left")
    building_hourly = building_hourly.merge(loop_components, on="building_id", how="left")
    building_hourly = building_hourly.merge(group_map, on=["building_id", "group_name"], how="left")
    building_hourly = building_hourly.merge(
        component_hourly[
            [
                "timestamp",
                "loop_avg_temperature_c",
                "central_loop_pump_kWh_el_hourly",
                "borefield_kWh_el_hourly",
                "auxiliary_kWh_el_hourly",
                "shared_network_kWh_el_hourly",
                "GEN_HVAC_network_kWh_el_hourly",
            ]
        ],
        on="timestamp",
        how="left",
    )
    building_hourly["GEN_building_hp_kWh_el_hourly"] = building_hourly["GEN_building_hp_kWh_el_hourly"].fillna(0.0)

    curves = _load_cop_curves(heatnets_source_root(raw=True) / "InputFiles" / "heat_pump_COP_curve.xlsx")
    fallback_cop = float(config["heatnets"]["dhw"]["fallback_constant_cop"])
    method = str(config["heatnets"]["dhw"]["method"])

    building_hourly["GEN_DHW_COP"] = fallback_cop
    for hp_type, curve in curves.items():
        mask = building_hourly["HP_type"].fillna(building_hourly["heatnets_hp_type"]) == hp_type
        if mask.any() and method == "heatnets_heating_curve_surrogate":
            building_hourly.loc[mask, "GEN_DHW_COP"] = _heating_cop(
                curve,
                building_hourly.loc[mask, "loop_avg_temperature_c"],
                fallback_cop,
            )

    capacity_kw = building_hourly["HP_total_size_tons"].fillna(np.inf) * TON_TO_KW
    thermal_dhw = building_hourly["waterh_kwh_th"].fillna(0.0)
    hp_served_dhw = np.minimum(thermal_dhw, capacity_kw)
    backup_served_dhw = np.maximum(thermal_dhw - capacity_kw, 0.0)
    building_hourly["GEN_DHW_kWh_el_hourly"] = (hp_served_dhw / building_hourly["GEN_DHW_COP"]) + backup_served_dhw

    building_hourly["shared_allocation_basis"] = building_hourly["GEN_building_hp_kWh_el_hourly"]
    zero_hp_mask = building_hourly.groupby("timestamp")["GEN_building_hp_kWh_el_hourly"].transform("sum") <= 0
    building_hourly.loc[zero_hp_mask, "shared_allocation_basis"] = (
        building_hourly.loc[zero_hp_mask, "heating_kwh_th"].fillna(0.0) + building_hourly.loc[zero_hp_mask, "cooling_kwh_th"].fillna(0.0)
    )
    shared_basis_total = building_hourly.groupby("timestamp")["shared_allocation_basis"].transform("sum")
    building_hourly["shared_network_allocation_share"] = np.where(
        shared_basis_total > 0,
        building_hourly["shared_allocation_basis"] / shared_basis_total,
        0.0,
    )
    building_hourly["GEN_shared_network_allocated_kWh_el_hourly"] = (
        building_hourly["shared_network_allocation_share"] * building_hourly["shared_network_kWh_el_hourly"]
    )
    building_hourly["GEN_HVAC_network_allocated_kWh_el_hourly"] = (
        building_hourly["GEN_building_hp_kWh_el_hourly"] + building_hourly["GEN_shared_network_allocated_kWh_el_hourly"]
    )
    building_hourly["GEN_total_B6_kWh_el_hourly"] = (
        building_hourly["GEN_HVAC_network_allocated_kWh_el_hourly"] + building_hourly["GEN_DHW_kWh_el_hourly"]
    )

    total_hourly = (
        building_hourly.groupby("timestamp", as_index=False)[
            [
                "GEN_building_hp_kWh_el_hourly",
                "GEN_shared_network_allocated_kWh_el_hourly",
                "GEN_HVAC_network_allocated_kWh_el_hourly",
                "GEN_DHW_kWh_el_hourly",
                "GEN_total_B6_kWh_el_hourly",
                "heating_kwh_th",
                "cooling_kwh_th",
                "waterh_kwh_th",
                "delivered_total_kwh_th",
            ]
        ]
        .sum()
        .sort_values("timestamp")
    )
    total_hourly = total_hourly.merge(
        component_hourly[
            [
                "timestamp",
                "loop_avg_temperature_c",
                "building_hp_kWh_el_hourly",
                "central_loop_pump_kWh_el_hourly",
                "borefield_kWh_el_hourly",
                "auxiliary_kWh_el_hourly",
                "shared_network_kWh_el_hourly",
                "GEN_HVAC_network_kWh_el_hourly",
            ]
        ],
        on="timestamp",
        how="left",
    )
    total_hourly.rename(
        columns={
            "GEN_HVAC_network_allocated_kWh_el_hourly": "GEN_HVAC_network_allocated_kWh_el_hourly",
            "GEN_DHW_kWh_el_hourly": "GEN_DHW_kWh_el_hourly",
            "GEN_total_B6_kWh_el_hourly": "GEN_total_B6_kWh_el_hourly",
        },
        inplace=True,
    )

    group_hourly = (
        building_hourly.groupby(["timestamp", "group_name"], as_index=False)[
            [
                "GEN_building_hp_kWh_el_hourly",
                "GEN_shared_network_allocated_kWh_el_hourly",
                "GEN_HVAC_network_allocated_kWh_el_hourly",
                "GEN_DHW_kWh_el_hourly",
                "GEN_total_B6_kWh_el_hourly",
                "heating_kwh_th",
                "cooling_kwh_th",
                "waterh_kwh_th",
                "delivered_total_kwh_th",
            ]
        ]
        .sum()
        .sort_values(["timestamp", "group_name"])
    )

    annual_total = pd.DataFrame(
        [
            {
                "GEN_building_hp_kWh_el_annual": float(total_hourly["GEN_building_hp_kWh_el_hourly"].sum()),
                "GEN_shared_network_kWh_el_annual": float(total_hourly["GEN_shared_network_allocated_kWh_el_hourly"].sum()),
                "GEN_HVAC_network_kWh_el_annual": float(total_hourly["GEN_HVAC_network_allocated_kWh_el_hourly"].sum()),
                "GEN_DHW_kWh_el_annual": float(total_hourly["GEN_DHW_kWh_el_hourly"].sum()),
                "GEN_total_B6_kWh_el_annual": float(total_hourly["GEN_total_B6_kWh_el_hourly"].sum()),
                "delivered_total_kwh_th_annual": float(total_hourly["delivered_total_kwh_th"].sum()),
                "mean_hourly_GEN_DHW_COP": float(
                    building_hourly.loc[building_hourly["waterh_kwh_th"] > 0, "GEN_DHW_COP"].mean()
                )
                if (building_hourly["waterh_kwh_th"] > 0).any()
                else fallback_cop,
                "load_weighted_GEN_DHW_COP": float(total_hourly["waterh_kwh_th"].sum() / total_hourly["GEN_DHW_kWh_el_hourly"].sum())
                if float(total_hourly["GEN_DHW_kWh_el_hourly"].sum()) > 0
                else fallback_cop,
                "dhw_method": method,
            }
        ]
    )

    annual_by_group = (
        group_hourly.groupby("group_name", as_index=False)[
            [
                "GEN_building_hp_kWh_el_hourly",
                "GEN_shared_network_allocated_kWh_el_hourly",
                "GEN_HVAC_network_allocated_kWh_el_hourly",
                "GEN_DHW_kWh_el_hourly",
                "GEN_total_B6_kWh_el_hourly",
                "heating_kwh_th",
                "cooling_kwh_th",
                "waterh_kwh_th",
                "delivered_total_kwh_th",
            ]
        ]
        .sum()
        .sort_values("group_name")
    )

    building_out = EXPORT_ROOT / "gen" / "gen_b6_hourly_building.csv"
    group_out = EXPORT_ROOT / "gen" / "gen_b6_hourly_group.csv"
    total_out = EXPORT_ROOT / "gen" / "gen_b6_hourly_total.csv"
    annual_out = EXPORT_ROOT / "gen" / "gen_b6_annual_summary.csv"
    annual_group_out = EXPORT_ROOT / "gen" / "gen_b6_annual_by_group.csv"
    dhw_cop_monthly_out = EXPORT_ROOT / "gen" / "gen_dhw_cop_monthly.csv"

    monthly_dhw = total_hourly.copy()
    monthly_dhw["month"] = monthly_dhw["timestamp"].dt.month
    monthly_dhw = monthly_dhw.groupby("month", as_index=False)[["waterh_kwh_th", "GEN_DHW_kWh_el_hourly"]].sum()
    monthly_dhw["load_weighted_GEN_DHW_COP"] = monthly_dhw["waterh_kwh_th"] / monthly_dhw["GEN_DHW_kWh_el_hourly"]

    building_hourly.to_csv(building_out, index=False)
    group_hourly.to_csv(group_out, index=False)
    total_hourly.to_csv(total_out, index=False)
    annual_total.to_csv(annual_out, index=False)
    annual_by_group.to_csv(annual_group_out, index=False)
    monthly_dhw.to_csv(dhw_cop_monthly_out, index=False)
    return {
        "building_hourly": building_out,
        "group_hourly": group_out,
        "total_hourly": total_out,
        "annual_summary": annual_out,
        "annual_by_group": annual_group_out,
        "dhw_cop_monthly": dhw_cop_monthly_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GEN Stage B6 outputs from validated HEATNETS HVAC electricity plus explicit DHW.")
    parser.parse_args()
    outputs = build()
    for label, path in outputs.items():
        print(f"{label}: {relpath(path)}")


if __name__ == "__main__":
    main()
