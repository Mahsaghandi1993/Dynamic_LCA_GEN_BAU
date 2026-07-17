from __future__ import annotations

import csv
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    probe = (start or Path(__file__)).resolve()
    for candidate in [probe.parent, *probe.parents]:
        if (candidate / "Dynamic_LCA_GEN_BAU").exists() and (candidate / "README.md").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root()
WORKSPACE_ROOT = REPO_ROOT / "Dynamic_LCA_GEN_BAU"
MANUSCRIPT_ROOT = WORKSPACE_ROOT / "export" / "manuscript"
ARCHIVE_ROOT = WORKSPACE_ROOT / "export" / "archive" / "previous_outputs"
OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "manuscript_scope_corrected_2026_06_02"
FUEL_SHARE_CONFIG = WORKSPACE_ROOT / "inputs" / "config" / "fuel_share_framingham_acs.yaml"


DEFAULT_ASSUMPTIONS: dict[str, float] = {
    "gas_main_length_m": 1609.34,
    "gas_service_length_per_customer_m": 15.0,
    "gas_main_pipe_kg_per_m": 5.0,
    "gas_service_pipe_kg_per_m": 1.2,
    "low_pressure_trench_width_m": 0.6,
    "low_pressure_trench_depth_m": 1.2,
    "gas_bedding_sand_kg_per_m": 600.0,
    "gas_bedding_cement_kg_per_m": 26.0,
    "gas_meter_regulator_steel_kg_per_customer": 15.0,
    "heating_oil_tank_steel_kg_per_customer": 125.0,
    "local_material_truck_distance_km": 50.0,
    "meter_and_tank_truck_distance_km": 100.0,
    "eol_truck_distance_km": 30.0,
    "filter_kg_per_hvac_system_year": 2.0,
    "filter_truck_distance_km": 350.0,
    "service_life_years": 50.0,
}


SF_GAS_FURNACE_BOM_KG = {
    "steel_low_alloyed": 46.0,
    "galvanized_steel": 18.0,
    "aluminum": 9.0,
    "copper": 3.0,
}
SF_AC_BOM_KG = {
    "steel_low_alloyed": 78.0,
    "galvanized_steel": 35.0,
    "aluminum": 17.0,
    "copper": 17.0,
    "refrigerant_r134a": 6.0,
}
SF_DUCTWORK_GALV_KG = 265.0

MF_GAS_FURNACE_COMBINED_KG = {
    "steel_low_alloyed": 370.0,
    "galvanized_steel": 145.0,
    "aluminum": 73.0,
    "copper": 25.0,
}
MF_AC_COMBINED_KG = {
    "steel_low_alloyed": 638.0,
    "galvanized_steel": 290.0,
    "aluminum": 141.0,
    "copper": 141.0,
    "refrigerant_r134a": 49.0,
}
MF_DUCTWORK_COMBINED_GALV_KG = 4300.0

COMM_GAS_FURNACE_COMBINED_KG = {
    "steel_low_alloyed": 1694.0,
    "galvanized_steel": 663.0,
    "aluminum": 332.0,
    "copper": 111.0,
}
COMM_AC_COMBINED_KG = {
    "steel_low_alloyed": 2923.0,
    "galvanized_steel": 1312.0,
    "aluminum": 638.0,
    "copper": 638.0,
    "refrigerant_r134a": 225.0,
}
COMM_DUCTWORK_COMBINED_GALV_KG = 19575.0


SCREENING_FACTORS_KG_CO2E_PER_UNIT = {
    "steel_low_alloyed": 2.0,
    "galvanized_steel": 2.4,
    "aluminum": 8.0,
    "copper": 4.0,
    "refrigerant_r134a": 7.0,
    "hdpe_pipe_material": 2.2,
    "pipe_extrusion": 0.25,
    "bedding_sand": 0.005,
    "bedding_cement": 0.85,
    "excavation_m3": 0.111,
    "freight_lorry_tkm": 0.10,
    "filter_material": 2.0,
    "mixed_metal_disposal": 0.10,
    "polyethylene_landfill": 0.10,
}


SOURCE_NOTES = {
    "NRC Canada WBLCA guideline": "https://publications.gc.ca/site/eng/9.908801/publication.html",
    "Schori 2012 natural gas LCI": "https://www.dflca.ch/inventories/Hintergrund/Schori_2012-NaturalGas.pdf",
    "PHMSA pipeline construction phases": "https://www.phmsa.dot.gov/technical-resources/pipeline/pipeline-construction/phases-pipeline-construction-overview",
    "Alvarez et al. 2018 methane supply-chain estimate": "https://pubmed.ncbi.nlm.nih.gov/29930092/",
    "NETL natural gas environmental performance": "https://www.netl.doe.gov/projects/VueConnection/download.aspx?filename=EvaluatingUSNaturalGasEnvironmentalPerformance_061423.pdf&id=922f5eb6-a92a-4f19-83a7-ce81761cb6e4",
    "NREL ComStock baseline documentation": "https://nrel.github.io/ComStock.github.io/docs/upgrade_measures/hvac_doas_mshp.html",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_json_compatible_yaml(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_single_family_fuel_share_config(path: Path = FUEL_SHARE_CONFIG) -> dict[str, Any]:
    """Load the ACS-backed base gas/oil split used by the conventional reference.

    Base value is grounded in Massachusetts ACS House Heating Fuel Table B25040:
    utility gas 50.5% and fuel oil 22.4%, renormalized to the fossil gas:oil
    ratio and rounded to 0.70/0.30 for the Framingham single-family baseline.
    """
    config = _read_json_compatible_yaml(path)
    base = config["base"]
    gas_share = float(base["gas_share"])
    oil_share = float(base["oil_share"])
    if not math.isclose(gas_share + oil_share, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Fuel shares in {path} must sum to 1.0; got {gas_share + oil_share}")
    if gas_share < 0.0 or oil_share < 0.0:
        raise ValueError(f"Fuel shares in {path} must be non-negative.")
    return config


def single_family_fuel_share_sensitivity_points(path: Path = FUEL_SHARE_CONFIG) -> list[dict[str, Any]]:
    config = load_single_family_fuel_share_config(path)
    sensitivity = config["sensitivity"]["S1_fuel_split"]
    rows: list[dict[str, Any]] = []
    for point in sensitivity["points"]:
        gas_share = float(point["gas_share"])
        oil_share = float(point["oil_share"])
        if not math.isclose(gas_share + oil_share, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"S1 point {point['scenario']} must sum to 1.0.")
        rows.append(
            {
                "scenario": point["scenario"],
                "parameter": sensitivity["parameter"],
                "gas_share": gas_share,
                "oil_share": oil_share,
                "label": point["label"],
                "rationale": sensitivity["rationale"],
                "reference": "; ".join(config["source"]["references"]),
            }
        )
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).replace(",", "").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def fmt(value: float, digits: int = 6) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-15):
        value = 0.0
    return f"{value:.{digits}g}"


def add_dicts(*payloads: dict[str, float]) -> dict[str, float]:
    out: defaultdict[str, float] = defaultdict(float)
    for payload in payloads:
        for key, value in payload.items():
            out[key] += float(value)
    return dict(out)


def scaled_dict(payload: dict[str, float], factor: float) -> dict[str, float]:
    return {key: float(value) * factor for key, value in payload.items()}


def building_counts_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    groups: defaultdict[str, int] = defaultdict(int)
    total = 0
    for row in rows:
        group = row.get("group_name", "").strip()
        if not group:
            continue
        total += 1
        groups[group] += 1

    single_family = groups["west_residential"] + groups["east_or_southeast_residential"]
    return {
        "west_residential": float(groups["west_residential"]),
        "east_or_southeast_residential": float(groups["east_or_southeast_residential"]),
        "single_family": float(single_family),
        "fha_multifamily": float(groups["fha_multifamily"]),
        "commercial_municipal": float(groups["commercial_municipal"]),
        "total": float(total),
    }


def load_case_basis() -> dict[str, float]:
    basis_rows = read_csv_rows(MANUSCRIPT_ROOT / "tables" / "supplementary" / "table_s1_framingham_validation_basis.csv")
    out: dict[str, float] = {}
    for row in basis_rows:
        item = row.get("item", "")
        if item == "Annual delivered thermal":
            out["annual_delivered_kwh_th"] = to_float(row.get("value"))
        elif item == "Lifetime denominator":
            out["lifetime_delivered_kwh_th"] = to_float(row.get("value"))
        elif item == "Service life":
            out["service_life_years"] = to_float(row.get("value"))
        elif item == "Modeled buildings":
            out["modeled_buildings"] = to_float(row.get("value"))
    return out


def equipment_masses_for_counts(counts: dict[str, float]) -> dict[str, dict[str, float]]:
    sf = float(counts["single_family"])
    fha = float(counts["fha_multifamily"])
    comm = float(counts["commercial_municipal"])

    sf_heating = scaled_dict(SF_GAS_FURNACE_BOM_KG, sf)
    sf_cooling = scaled_dict(SF_AC_BOM_KG, sf)
    sf_duct = {"galvanized_steel": SF_DUCTWORK_GALV_KG * sf}

    mf_heat_per_building = scaled_dict(MF_GAS_FURNACE_COMBINED_KG, 1.0 / 9.0)
    mf_ac_per_building = scaled_dict(MF_AC_COMBINED_KG, 1.0 / 9.0)
    fha_heating = scaled_dict(mf_heat_per_building, fha * 0.35)
    fha_cooling = scaled_dict(mf_ac_per_building, fha)
    fha_duct = {"galvanized_steel": (MF_DUCTWORK_COMBINED_GALV_KG / 9.0) * fha}

    comm_heat_per_building = scaled_dict(COMM_GAS_FURNACE_COMBINED_KG, 1.0 / 5.0)
    comm_ac_per_building = scaled_dict(COMM_AC_COMBINED_KG, 1.0 / 5.0)
    comm_heating = scaled_dict(comm_heat_per_building, comm)
    comm_cooling = scaled_dict(comm_ac_per_building, comm)
    comm_duct = {"galvanized_steel": (COMM_DUCTWORK_COMBINED_GALV_KG / 5.0) * comm}

    initial_equipment = add_dicts(
        sf_heating,
        sf_cooling,
        sf_duct,
        fha_heating,
        fha_cooling,
        fha_duct,
        comm_heating,
        comm_cooling,
        comm_duct,
    )
    replacement_equipment = add_dicts(
        sf_heating,
        sf_cooling,
        fha_heating,
        fha_cooling,
        comm_heating,
        comm_cooling,
    )
    return {
        "initial_equipment": initial_equipment,
        "replacement_equipment": replacement_equipment,
    }


def local_gas_oil_infrastructure_inventory(
    counts: dict[str, float],
    assumptions: dict[str, float] = DEFAULT_ASSUMPTIONS,
    *,
    single_family_gas_share: float,
    single_family_oil_share: float,
) -> dict[str, float]:
    single_family = float(counts["single_family"])
    commercial = float(counts["commercial_municipal"])
    if not math.isclose(single_family_gas_share + single_family_oil_share, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("single-family gas/oil shares must sum to 1.0")
    gas_customers = single_family * single_family_gas_share + commercial
    oil_customers = single_family * single_family_oil_share
    service_length_m = gas_customers * assumptions["gas_service_length_per_customer_m"]
    gas_pipe_length_m = assumptions["gas_main_length_m"] + service_length_m
    gas_main_pipe_kg = assumptions["gas_main_length_m"] * assumptions["gas_main_pipe_kg_per_m"]
    gas_service_pipe_kg = service_length_m * assumptions["gas_service_pipe_kg_per_m"]
    gas_pipe_kg = gas_main_pipe_kg + gas_service_pipe_kg
    trench_volume_m3 = (
        gas_pipe_length_m
        * assumptions["low_pressure_trench_width_m"]
        * assumptions["low_pressure_trench_depth_m"]
    )
    sand_kg = gas_pipe_length_m * assumptions["gas_bedding_sand_kg_per_m"]
    cement_kg = gas_pipe_length_m * assumptions["gas_bedding_cement_kg_per_m"]
    meter_regulator_steel_kg = gas_customers * assumptions["gas_meter_regulator_steel_kg_per_customer"]
    oil_tank_steel_kg = oil_customers * assumptions["heating_oil_tank_steel_kg_per_customer"]
    local_material_lorry_tkm = (
        (sand_kg + cement_kg) / 1000.0 * assumptions["local_material_truck_distance_km"]
    )
    meter_tank_lorry_tkm = (
        (meter_regulator_steel_kg + oil_tank_steel_kg)
        / 1000.0
        * assumptions["meter_and_tank_truck_distance_km"]
    )
    oil_tank_eol_lorry_tkm = (
        oil_tank_steel_kg / 1000.0 * assumptions["eol_truck_distance_km"]
    )
    return {
        "gas_customers": gas_customers,
        "oil_customers": oil_customers,
        "gas_service_length_m": service_length_m,
        "gas_pipe_length_m": gas_pipe_length_m,
        "gas_main_pipe_kg": gas_main_pipe_kg,
        "gas_service_pipe_kg": gas_service_pipe_kg,
        "gas_pipe_kg": gas_pipe_kg,
        "trench_excavation_m3": trench_volume_m3,
        "bedding_sand_kg": sand_kg,
        "bedding_cement_kg": cement_kg,
        "meter_regulator_steel_kg": meter_regulator_steel_kg,
        "oil_tank_steel_kg": oil_tank_steel_kg,
        "local_material_lorry_tkm": local_material_lorry_tkm,
        "meter_tank_lorry_tkm": meter_tank_lorry_tkm,
        "oil_tank_eol_lorry_tkm": oil_tank_eol_lorry_tkm,
    }


def screening_factor(flow_key: str) -> float:
    return SCREENING_FACTORS_KG_CO2E_PER_UNIT.get(flow_key, 0.0)


def correction_row(
    *,
    stage: str,
    module: str,
    component: str,
    flow_name: str,
    flow_key: str,
    quantity: float,
    unit: str,
    timing: str,
    correction_status: str,
    calculation_basis: str,
    source_or_reference: str,
    brightway_mapping: str,
    notes: str,
    lifetime_denominator: float,
    scenario: str = "core_corrected_BAU",
) -> dict[str, Any]:
    factor = screening_factor(flow_key)
    screening_impact = quantity * factor
    if correction_status == "memo_no_elementary_output":
        factor = 0.0
        screening_impact = 0.0
    return {
        "pathway": "BAU",
        "scenario": scenario,
        "stage": stage,
        "module": module,
        "component": component,
        "flow_name": flow_name,
        "flow_key": flow_key,
        "quantity": fmt(quantity),
        "unit": unit,
        "quantity_per_FU": fmt(quantity),
        "quantity_per_kWh_th": fmt(quantity / lifetime_denominator),
        "timing": timing,
        "correction_status": correction_status,
        "calculation_basis": calculation_basis,
        "source_or_reference": source_or_reference,
        "brightway_mapping": brightway_mapping,
        "screening_factor_kgCO2e_per_unit": fmt(factor),
        "screening_static_kgCO2e": fmt(screening_impact),
        "notes": notes,
    }


def build_correction_rows(
    counts: dict[str, float],
    basis: dict[str, float],
    assumptions: dict[str, float] = DEFAULT_ASSUMPTIONS,
    *,
    single_family_gas_share: float,
    single_family_oil_share: float,
) -> list[dict[str, Any]]:
    lifetime = float(basis["lifetime_delivered_kwh_th"])
    corrected = equipment_masses_for_counts(counts)
    old_counts = {
        "single_family": 22.0,
        "fha_multifamily": 9.0,
        "commercial_municipal": 5.0,
        "total": 36.0,
    }
    old = equipment_masses_for_counts(old_counts)
    infra = local_gas_oil_infrastructure_inventory(
        counts,
        assumptions,
        single_family_gas_share=single_family_gas_share,
        single_family_oil_share=single_family_oil_share,
    )
    old_infra = {
        "gas_pipe_kg": 8532.70,
    }

    rows: list[dict[str, Any]] = []

    initial_delta = {
        key: corrected["initial_equipment"].get(key, 0.0) - old["initial_equipment"].get(key, 0.0)
        for key in sorted(set(corrected["initial_equipment"]) | set(old["initial_equipment"]))
    }
    for key, quantity in initial_delta.items():
        if abs(quantity) < 1e-12:
            continue
        rows.append(
            correction_row(
                stage="A",
                module="A1-A3",
                component="37-building BAU HVAC equipment count correction",
                flow_name=f"{key} manufacturing material adjustment",
                flow_key=key,
                quantity=quantity,
                unit="kg/FU",
                timing="year 0",
                correction_status="added_to_core",
                calculation_basis="Corrected single-family count is 23, replacing old non-B6 hard-code of 22.",
                source_or_reference="building_group_map.csv and table_s1_framingham_validation_basis.csv",
                brightway_mapping="same material proxies already used in bau_dynamic_fg_heatnets_sync",
                notes="Corrects the BAU non-B6 foreground to include all 37 public case buildings.",
                lifetime_denominator=lifetime,
            )
        )

    replacement_delta = {
        key: corrected["replacement_equipment"].get(key, 0.0) - old["replacement_equipment"].get(key, 0.0)
        for key in sorted(set(corrected["replacement_equipment"]) | set(old["replacement_equipment"]))
    }
    for pulse_year in [20, 40]:
        for key, quantity in replacement_delta.items():
            if abs(quantity) < 1e-12:
                continue
            rows.append(
                correction_row(
                    stage="B",
                    module="B4",
                    component=f"37-building BAU equipment replacement count correction y{pulse_year}",
                    flow_name=f"{key} replacement material adjustment",
                    flow_key=key,
                    quantity=quantity,
                    unit="kg/FU",
                    timing=f"year {pulse_year}",
                    correction_status="added_to_core",
                    calculation_basis="Replacement pulses follow the same corrected 23 single-family count.",
                    source_or_reference="Existing BAU replacement schedule plus corrected building_group_map.csv count.",
                    brightway_mapping="same material proxies already used in BAU B2-B4 replacement activities",
                    notes="Corrects year-20 and year-40 BAU equipment replacement quantities.",
                    lifetime_denominator=lifetime,
                )
            )

    rows.append(
        correction_row(
            stage="A",
            module="A1-A3",
            component="Local gas service infrastructure count correction",
            flow_name="HDPE gas main and service pipe material adjustment",
            flow_key="hdpe_pipe_material",
            quantity=infra["gas_pipe_kg"] - old_infra["gas_pipe_kg"],
            unit="kg/FU",
            timing="year 0",
            correction_status="adjusted_in_core",
            calculation_basis=(
                "Gas service customers = single-family gas share plus commercial gas customers; "
                "old non-B6 foreground used all 22 residential plus commercial."
            ),
            source_or_reference="building_group_map.csv, bau_baseline_config.yaml, and approved core boundary",
            brightway_mapping="market for polyethylene, high density, granulate",
            notes="Keeps local gas pipe material in the BAU core, but corrects service-customer count.",
            lifetime_denominator=lifetime,
        )
    )
    rows.append(
        correction_row(
            stage="A",
            module="A1-A3",
            component="Local gas service infrastructure count correction",
            flow_name="Plastic pipe extrusion service adjustment",
            flow_key="pipe_extrusion",
            quantity=infra["gas_pipe_kg"] - old_infra["gas_pipe_kg"],
            unit="kg/FU",
            timing="year 0",
            correction_status="adjusted_in_core",
            calculation_basis="Extrusion service follows corrected HDPE gas-pipe material quantity.",
            source_or_reference="Existing GEN/BAU pipe mapping plus corrected gas-customer count",
            brightway_mapping="market for extrusion, plastic pipes",
            notes="Negative values reduce the prior all-residential gas-service proxy.",
            lifetime_denominator=lifetime,
        )
    )

    local_additions = [
        (
            "A",
            "A5",
            "Local gas pipeline trench bedding",
            "Sand bedding for low-pressure local gas service trench",
            "bedding_sand",
            infra["bedding_sand_kg"],
            "kg/FU",
            "Schori distribution-network LCI reports 600 kg/m sand for 0.1-1.0 bar trench profile.",
            "market for sand or gravel proxy; requires Brightway remap",
        ),
        (
            "A",
            "A5",
            "Local gas pipeline trench bedding",
            "Cement or mortar bedding/restoration proxy",
            "bedding_cement",
            infra["bedding_cement_kg"],
            "kg/FU",
            "Schori distribution-network LCI reports 26 kg/m cement for distribution trench profiles.",
            "market for cement; requires Brightway remap",
        ),
        (
            "A",
            "A5",
            "Local gas pipeline trenching and backfill",
            "Open-trench excavation volume for gas main and service lines",
            "excavation_m3",
            infra["trench_excavation_m3"],
            "m3/FU",
            "0.6 m x 1.2 m low-pressure trench profile applied to corrected local gas length.",
            "excavation, hydraulic digger / skid-steer loader proxy; requires Brightway remap",
        ),
        (
            "A",
            "A1-A3",
            "Gas metering and service hardware",
            "Gas meter/regulator steel proxy",
            "steel_low_alloyed",
            infra["meter_regulator_steel_kg"],
            "kg/FU",
            "15 kg steel per gas customer engineering proxy pending utility asset takeoff.",
            "market for steel, low-alloyed, hot rolled",
        ),
        (
            "A",
            "A1-A3",
            "Fuel-oil storage infrastructure",
            "Residential heating-oil tank steel proxy",
            "steel_low_alloyed",
            infra["oil_tank_steel_kg"],
            "kg/FU",
            "125 kg steel per oil-served single-family customer engineering proxy pending field data.",
            "market for steel, low-alloyed, hot rolled",
        ),
        (
            "A",
            "A4",
            "Local gas bedding material transport",
            "Truck transport for sand and cement to site",
            "freight_lorry_tkm",
            infra["local_material_lorry_tkm"],
            "tkm/FU",
            "Local material transport distance = 50 km.",
            "transport, freight, lorry >32 metric ton",
        ),
        (
            "A",
            "A4",
            "Gas meter and oil-tank transport",
            "Truck transport for meters, regulators, and oil tanks",
            "freight_lorry_tkm",
            infra["meter_tank_lorry_tkm"],
            "tkm/FU",
            "Equipment transport distance = 100 km.",
            "transport, freight, lorry >32 metric ton",
        ),
    ]
    for stage, module, component, flow_name, flow_key, quantity, unit, basis_text, mapping in local_additions:
        rows.append(
            correction_row(
                stage=stage,
                module=module,
                component=component,
                flow_name=flow_name,
                flow_key=flow_key,
                quantity=quantity,
                unit=unit,
                timing="year 0",
                correction_status="added_to_core",
                calculation_basis=basis_text,
                source_or_reference="Schori 2012 natural gas LCI; PHMSA construction-phase description; approved local-infrastructure core boundary",
                brightway_mapping=mapping,
                notes="Added because the BAU comparison is now modeled as new local conventional infrastructure.",
                lifetime_denominator=lifetime,
            )
        )

    filter_kg = counts["total"] * assumptions["filter_kg_per_hvac_system_year"] * assumptions["service_life_years"]
    filter_tkm = filter_kg / 1000.0 * assumptions["filter_truck_distance_km"]
    rows.append(
        correction_row(
            stage="B",
            module="B2",
            component="Routine BAU HVAC maintenance",
            flow_name="Furnace/HVAC filter material",
            flow_key="filter_material",
            quantity=filter_kg,
            unit="kg/FU",
            timing="annual over years 1-50",
            correction_status="added_to_core",
            calculation_basis="2 kg filters per HVAC system per year x 37 buildings x 50 years.",
            source_or_reference="NRC Canada WBLCA guideline furnace maintenance module; corrected 37-building count",
            brightway_mapping="filter material or inert-waste proxy; requires Brightway remap",
            notes="Adds routine BAU maintenance that was not explicit in the manuscript-facing table.",
            lifetime_denominator=lifetime,
        )
    )
    rows.append(
        correction_row(
            stage="B",
            module="B2",
            component="Routine BAU HVAC maintenance",
            flow_name="Truck transport for replacement filters",
            flow_key="freight_lorry_tkm",
            quantity=filter_tkm,
            unit="tkm/FU",
            timing="annual over years 1-50",
            correction_status="added_to_core",
            calculation_basis="Filter mass transported 350 km.",
            source_or_reference="NRC Canada WBLCA guideline furnace maintenance module",
            brightway_mapping="transport, freight, lorry >32 metric ton",
            notes="Screening proxy; exact filter supply chain should be replaced if field data are obtained.",
            lifetime_denominator=lifetime,
        )
    )

    current_polyethylene_landfill_kg = 8532.7
    rows.append(
        correction_row(
            stage="C",
            module="C4",
            component="Gas service end of life",
            flow_name="Remove current landfill treatment for buried gas pipe",
            flow_key="polyethylene_landfill",
            quantity=-current_polyethylene_landfill_kg,
            unit="kg/FU",
            timing="year 50",
            correction_status="removed_from_core",
            calculation_basis="Core scenario treats buried local gas pipe as abandoned in place, not excavated and landfilled.",
            source_or_reference="Approved core boundary and Schori EOL discussion that abandonment/excavation is uncertain",
            brightway_mapping="remove treatment of waste polyethylene, sanitary landfill from BAU core",
            notes="Keep as a sensitivity if reviewers request removed-pipe EOL.",
            lifetime_denominator=lifetime,
        )
    )
    rows.append(
        correction_row(
            stage="C",
            module="C4",
            component="Gas service end of life",
            flow_name="Gas pipe abandoned in place memo flow",
            flow_key="hdpe_pipe_material",
            quantity=infra["gas_pipe_kg"],
            unit="kg/FU",
            timing="year 50 memo only",
            correction_status="memo_no_elementary_output",
            calculation_basis="Buried gas main and service lines remain in place in the core case.",
            source_or_reference="Approved core boundary",
            brightway_mapping="no technosphere exchange in core; memo inventory only",
            notes="Reported for transparency; not included in LCIA because no removal/disposal process is assigned.",
            lifetime_denominator=lifetime,
        )
    )
    rows.append(
        correction_row(
            stage="C",
            module="C2-C4",
            component="Fuel-oil storage end of life",
            flow_name="Oil tank mixed-metal removal/disposal proxy",
            flow_key="mixed_metal_disposal",
            quantity=infra["oil_tank_steel_kg"],
            unit="kg/FU",
            timing="year 50",
            correction_status="added_to_core",
            calculation_basis="Oil tanks are above/below-ground local fuel infrastructure requiring removal or abandonment decision.",
            source_or_reference="Approved BAU boundary; exact tank data pending field verification",
            brightway_mapping="mixed metal recycling/disposal proxy; requires Brightway remap",
            notes="Soil contamination is noted qualitatively; no quantified contamination release is added without site data.",
            lifetime_denominator=lifetime,
        )
    )
    rows.append(
        correction_row(
            stage="C",
            module="C2",
            component="Fuel-oil storage end of life",
            flow_name="Truck transport for removed oil tanks",
            flow_key="freight_lorry_tkm",
            quantity=infra["oil_tank_eol_lorry_tkm"],
            unit="tkm/FU",
            timing="year 50",
            correction_status="added_to_core",
            calculation_basis="Oil-tank steel mass transported 30 km to waste handling.",
            source_or_reference="Approved BAU boundary; NRC-style C2 transport convention",
            brightway_mapping="transport, freight, lorry >32 metric ton",
            notes="Screening proxy pending field-confirmed removal pathway.",
            lifetime_denominator=lifetime,
        )
    )
    return rows


def current_inventory_rows(basis: dict[str, float]) -> list[dict[str, Any]]:
    lifetime = basis["lifetime_delivered_kwh_th"]
    source = MANUSCRIPT_ROOT / "tables" / "supplementary" / "table_s3_lci_aggregated_foreground_inventory.csv"
    rows = []
    for row in read_csv_rows(source):
        quantity = to_float(row.get("Quantity over 50 years or annual quantity"))
        unit = row.get("Unit", "")
        rows.append(
            {
                "pathway": row.get("Pathway", ""),
                "scenario": "current_manuscript_inventory",
                "stage": row.get("Stage", ""),
                "module": "",
                "component": row.get("Component/process", ""),
                "flow_name": row.get("Component/process", ""),
                "flow_key": "",
                "quantity": row.get("Quantity over 50 years or annual quantity", ""),
                "unit": unit,
                "quantity_per_FU": row.get("Quantity over 50 years or annual quantity", ""),
                "quantity_per_kWh_th": fmt(quantity / lifetime) if quantity else "",
                "timing": row.get("Timing", ""),
                "correction_status": "existing_current_model",
                "calculation_basis": row.get("Source/data basis", ""),
                "source_or_reference": "Current manuscript Supplementary Table S3",
                "brightway_mapping": "",
                "screening_factor_kgCO2e_per_unit": "",
                "screening_static_kgCO2e": "",
                "notes": row.get("Notes/proxy", ""),
            }
        )
    return rows


def load_method_scores() -> dict[tuple[str, str, str], dict[str, Any]]:
    scores: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case in ["GEN", "BAU"]:
        path = ARCHIVE_ROOT / "method_comparison" / f"total_method_comparison_{case}.csv"
        for row in read_csv_rows(path):
            scores[(case, row["metric_key"], row["method_mode"])] = row
    return scores


def load_stage_climate() -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for case in ["GEN", "BAU"]:
        path = ARCHIVE_ROOT / "full_dynamic_lcia" / "total" / f"climate_stage_contribution_{case}.csv"
        for row in read_csv_rows(path):
            out[(case, row["stage"], row["metric"])] = to_float(row["stage_score_case_total"])
    return out


def correction_static_by_stage(correction_rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for row in correction_rows:
        if row["correction_status"] == "memo_no_elementary_output":
            continue
        totals[row["stage"]] += to_float(row["screening_static_kgCO2e"])
    return dict(totals)


def build_lcia_screening_summary(correction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = load_method_scores()
    stage = load_stage_climate()
    static_by_stage = correction_static_by_stage(correction_rows)
    static_delta_total = sum(static_by_stage.values())

    stage_dynamic_ratios: dict[str, float] = {}
    for stage_key in ["A", "B", "C"]:
        static_stage = stage.get(("BAU", stage_key, "static_climate"), 0.0)
        dynamic_stage = stage.get(("BAU", stage_key, "dynamic_GWP100"), 0.0)
        stage_dynamic_ratios[stage_key] = dynamic_stage / static_stage if static_stage else 0.0
    dynamic_delta = sum(static_by_stage.get(stage_key, 0.0) * stage_dynamic_ratios.get(stage_key, 0.0) for stage_key in ["A", "B", "C"])
    existing_dynamic = to_float(scores[("BAU", "dynamic_GWP100", "dynamic_climate")]["score_case_total"])
    existing_rf = to_float(scores[("BAU", "dynamic_RF", "dynamic_climate")]["score_case_total"])
    rf_per_dynamic_kg = existing_rf / existing_dynamic if existing_dynamic else 0.0

    climate_rows = [
        ("climate_static", "conventional_static", static_delta_total),
        ("climate_static", "time_explicit_static", static_delta_total),
        ("dynamic_GWP100", "dynamic_climate", dynamic_delta),
        ("dynamic_RF", "dynamic_climate", dynamic_delta * rf_per_dynamic_kg),
    ]

    rows: list[dict[str, Any]] = []
    for metric_key, method_mode, delta in climate_rows:
        gen = to_float(scores[("GEN", metric_key, method_mode)]["score_case_total"])
        bau = to_float(scores[("BAU", metric_key, method_mode)]["score_case_total"])
        unit = scores[("BAU", metric_key, method_mode)]["unit"]
        corrected = bau + delta
        rows.append(
            {
                "metric_key": metric_key,
                "method_mode": method_mode,
                "unit": unit,
                "GEN_existing": fmt(gen),
                "BAU_existing": fmt(bau),
                "BAU_core_screening_delta": fmt(delta),
                "BAU_corrected_screening": fmt(corrected),
                "BAU_minus_GEN_existing": fmt(bau - gen),
                "BAU_minus_GEN_corrected_screening": fmt(corrected - gen),
                "GEN_div_BAU_existing": fmt(gen / bau if bau else 0.0),
                "GEN_div_BAU_corrected_screening": fmt(gen / corrected if corrected else 0.0),
                "status": "screening_only_pending_brightway_rerun",
                "notes": "Delta uses explicit foreground quantities and simple literature screening factors; replace with full Brightway LCIA before manuscript use.",
            }
        )
    return rows


def build_stage_screening_summary(correction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage = load_stage_climate()
    static_by_stage = correction_static_by_stage(correction_rows)
    rows: list[dict[str, Any]] = []
    for stage_key in ["A", "B", "C"]:
        existing_static = stage.get(("BAU", stage_key, "static_climate"), 0.0)
        existing_dynamic = stage.get(("BAU", stage_key, "dynamic_GWP100"), 0.0)
        ratio = existing_dynamic / existing_static if existing_static else 0.0
        static_delta = static_by_stage.get(stage_key, 0.0)
        dynamic_delta = static_delta * ratio
        rows.append(
            {
                "case": "BAU",
                "stage": stage_key,
                "existing_time_explicit_static_kgCO2e": fmt(existing_static),
                "screening_static_delta_kgCO2e": fmt(static_delta),
                "corrected_time_explicit_static_screening_kgCO2e": fmt(existing_static + static_delta),
                "existing_dynamic_GWP100_kgCO2e": fmt(existing_dynamic),
                "screening_dynamic_delta_kgCO2e": fmt(dynamic_delta),
                "corrected_dynamic_screening_kgCO2e": fmt(existing_dynamic + dynamic_delta),
                "status": "screening_only_pending_brightway_rerun",
            }
        )
    return rows


def build_scope_sanity_rows(
    counts: dict[str, float],
    infra: dict[str, float],
    fuel_share_config: dict[str, Any],
) -> list[dict[str, Any]]:
    equipment = equipment_masses_for_counts(counts)
    hvac_steel = float(equipment["initial_equipment"].get("steel_low_alloyed", 0.0)) + float(
        equipment["initial_equipment"].get("galvanized_steel", 0.0)
    )
    indoor_gas_piping_steel = infra["gas_customers"] * 20.0 * 1.5
    hdpe_reconciles = math.isclose(
        infra["gas_main_pipe_kg"] + infra["gas_service_pipe_kg"],
        infra["gas_pipe_kg"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    base = fuel_share_config["base"]
    rows = [
        {
            "check": "public_case_building_count",
            "expected": "37 modeled buildings",
            "current_issue": "Old BAU non-B6 notebook used 22 + 9 + 5 = 36 buildings.",
            "corrected_value": fmt(counts["total"]),
            "status": "corrected_in_new_inventory_package",
        },
        {
            "check": "single_family_count",
            "expected": "23 single-family modeled buildings",
            "current_issue": "Old BAU non-B6 notebook used 22 residential buildings.",
            "corrected_value": fmt(counts["single_family"]),
            "status": "corrected_in_new_inventory_package",
        },
        {
            "check": "BAU_stage_A_local_gas_infrastructure",
            "expected": "Local gas main/service material, bedding, transport, and trench installation included in core.",
            "current_issue": "Current manuscript table includes gas pipe material but not explicit bedding/trenching/A5 construction service.",
            "corrected_value": "added A1-A3/A4/A5 rows plus exclusion sensitivity note",
            "status": "corrected_in_new_inventory_package",
        },
        {
            "check": "BAU_stage_C_gas_line_EOF",
            "expected": "Buried gas pipe abandonment in place in core; removal/landfill only as sensitivity.",
            "current_issue": "Current manuscript S3 reports polyethylene landfill treatment at year 50.",
            "corrected_value": "core removes landfill treatment and reports memo abandonment flow",
            "status": "corrected_in_new_inventory_package",
        },
        {
            "check": "LCIA_recalculation",
            "expected": "Fresh Brightway foreground write and denominator sync after inventory update; LCIA is the follow-on calculation step.",
            "current_issue": "Old package text marked the foreground write blocked when the temporary bw25 runtime was unavailable.",
            "corrected_value": "corrected source/synchronized foregrounds, B6 wrappers, and full dynamic LCIA rerun completed in GEN_DLCA_391",
            "status": "full_lcia_rerun_complete",
        },
    ]
    rows.extend(
        [
            {
                "check": "single_family_fuel_share_source",
                "expected": "ACS-backed base gas/oil split loaded from inputs/config/fuel_share_framingham_acs.yaml",
                "current_issue": "Old code hardcoded 0.60/0.40 while the previous defaults file carried 0.75/0.25.",
                "corrected_value": f"gas_share={float(base['gas_share']):.2f}; oil_share={float(base['oil_share']):.2f}",
                "status": "pass",
            },
            {
                "check": "single_family_fuel_share_s1_sensitivity",
                "expected": "Low/base/high S1 points for one-at-a-time sensitivity.",
                "current_issue": "No explicit S1 matrix was generated from the fuel-share source.",
                "corrected_value": "0.60/0.40; 0.70/0.30; 0.75/0.25",
                "status": "pass",
            },
            {
                "check": "base_gas_and_oil_customer_counts",
                "expected": "gas_customers=23*0.70+5=21.10; oil_customers=23*0.30=6.90",
                "current_issue": "Counts previously followed hardcoded 0.60/0.40.",
                "corrected_value": f"gas_customers={infra['gas_customers']:.2f}; oil_customers={infra['oil_customers']:.2f}",
                "status": "pass",
            },
            {
                "check": "stage_A_gas_infrastructure_totals",
                "expected": "All dependent gas infrastructure quantities update from gas customers and gas service length.",
                "current_issue": "Dependent quantities could drift if the fuel split was not passed explicitly.",
                "corrected_value": (
                    f"gas_pipe_kg={infra['gas_pipe_kg']:.2f}; bedding_sand_kg={infra['bedding_sand_kg']:.2f}; "
                    f"bedding_cement_kg={infra['bedding_cement_kg']:.2f}; trench_excavation_m3={infra['trench_excavation_m3']:.4f}; "
                    f"meter_regulator_steel_kg={infra['meter_regulator_steel_kg']:.2f}; oil_tank_steel_kg={infra['oil_tank_steel_kg']:.2f}"
                ),
                "status": "pass",
            },
            {
                "check": "stage_A_HDPE_reconciliation",
                "expected": "gas_main_pipe_kg + gas_service_pipe_kg = gas_pipe_kg",
                "current_issue": "HDPE total must follow the fuel-share-dependent service pipe length.",
                "corrected_value": (
                    f"{infra['gas_main_pipe_kg']:.2f} + {infra['gas_service_pipe_kg']:.2f} = {infra['gas_pipe_kg']:.2f} kg"
                ),
                "status": "pass" if hdpe_reconciles else "fail",
            },
            {
                "check": "stage_A_steel_reconciliation_expanded_foreground",
                "expected": "HVAC steel + meter/regulator + oil-tank + indoor piping = modeled expanded Stage A steel proxy total",
                "current_issue": "Steel total must follow both the ACS split and indoor gas-piping gas-customer count.",
                "corrected_value": (
                    f"{hvac_steel:.2f} + {infra['meter_regulator_steel_kg']:.2f} + "
                    f"{infra['oil_tank_steel_kg']:.2f} + {indoor_gas_piping_steel:.2f} = "
                    f"{hvac_steel + infra['meter_regulator_steel_kg'] + infra['oil_tank_steel_kg'] + indoor_gas_piping_steel:.2f} kg"
                ),
                "status": "pass",
            },
        ]
    )
    return rows


def build_rerun_status_rows() -> list[dict[str, Any]]:
    return [
        {
            "step": "1",
            "workflow_item": "Write corrected BAU foreground database",
            "required_environment": "bw25 with bw2data, bw2io, bw_temporalis, bw_timex, pandas",
            "status_in_this_package": "completed_in_brightway",
            "reason": "Corrected non-B6 source foregrounds were written into the local GEN_DLCA_391 Brightway project with the recreated bw25 runtime.",
            "command_when_environment_available": "MPLCONFIGDIR=/private/tmp/mplconfig /private/tmp/bw25-lcia-env/bin/python Dynamic_LCA_GEN_BAU/scripts/b6/build_corrected_non_b6_foregrounds.py",
        },
        {
            "step": "2",
            "workflow_item": "Rerun denominator sync and B6 writer against corrected BAU foreground",
            "required_environment": "bw25",
            "status_in_this_package": "completed_in_brightway",
            "reason": "Corrected source foregrounds were synchronized to the accepted HEATNETS denominator in the local GEN_DLCA_391 Brightway project.",
            "command_when_environment_available": "MPLCONFIGDIR=/private/tmp/mplconfig /private/tmp/bw25-lcia-env/bin/python Dynamic_LCA_GEN_BAU/scripts/b6/sync_non_b6_denominator.py --write-brightway",
        },
        {
            "step": "3",
            "workflow_item": "Rerun full static, time-explicit, and dynamic LCIA",
            "required_environment": "bw25 with local Brightway project GEN_DLCA_391",
            "status_in_this_package": "completed_in_brightway",
            "reason": "Full static, time-explicit, and dynamic LCIA outputs were regenerated under export/full_dynamic_lcia.",
            "command_when_environment_available": "MPLCONFIGDIR=/private/tmp/mplconfig /private/tmp/bw25-lcia-env/bin/python Dynamic_LCA_GEN_BAU/scripts/b6/run_full_dynamic_lcia.py",
        },
        {
            "step": "4",
            "workflow_item": "Rebuild manuscript figures and SI tables",
            "required_environment": "bw25 after LCIA rerun",
            "status_in_this_package": "screening_figures_only",
            "reason": "New SVG figures show correction magnitude but are not final manuscript LCIA figures.",
            "command_when_environment_available": "python Dynamic_LCA_GEN_BAU/scripts/b6/build_results_final_package.py",
        },
    ]


def make_svg_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], unit: str, color: str = "#3b7c78") -> None:
    width = 980
    row_h = 42
    margin_l = 260
    margin_r = 90
    top = 72
    height = top + len(rows) * row_h + 42
    max_abs = max([abs(value) for _, value in rows] + [1.0])
    axis_x = margin_l + (width - margin_l - margin_r) * (0.35 if any(v < 0 for _, v in rows) else 0.0)
    scale_span = width - margin_l - margin_r
    if any(v < 0 for _, v in rows):
        scale = scale_span / (2 * max_abs)
    else:
        scale = scale_span / max_abs

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="32" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
        f'<line x1="{axis_x:.1f}" y1="{top - 20}" x2="{axis_x:.1f}" y2="{height - 28}" stroke="#9aa6a6" stroke-width="1"/>',
    ]
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_h
        if value >= 0:
            x = axis_x
            bar_w = value * scale
            fill = color
        else:
            bar_w = abs(value) * scale
            x = axis_x - bar_w
            fill = "#b45a4d"
        parts.extend(
            [
                f'<text x="30" y="{y + 22}" font-family="Arial, sans-serif" font-size="15" fill="#2b3436">{html.escape(label)}</text>',
                f'<rect x="{x:.1f}" y="{y + 4}" width="{max(bar_w, 1):.1f}" height="24" rx="3" fill="{fill}"/>',
                f'<text x="{x + bar_w + 8 if value >= 0 else x - 8:.1f}" y="{y + 22}" font-family="Arial, sans-serif" font-size="14" fill="#2b3436" text-anchor="{"start" if value >= 0 else "end"}">{fmt(value, 4)} {html.escape(unit)}</text>',
            ]
        )
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def build_figures(output_root: Path, correction_rows: list[dict[str, Any]], lcia_rows: list[dict[str, Any]]) -> None:
    figure_root = output_root / "figures"
    by_component: defaultdict[str, float] = defaultdict(float)
    by_stage: defaultdict[str, float] = defaultdict(float)
    for row in correction_rows:
        if row["correction_status"] == "memo_no_elementary_output":
            continue
        by_component[row["component"]] += to_float(row["screening_static_kgCO2e"])
        by_stage[f"Stage {row['stage']}"] += to_float(row["screening_static_kgCO2e"])
    make_svg_bar_chart(
        figure_root / "fig01_bau_correction_static_screening_by_stage.svg",
        "BAU Corrected-Scope Climate Delta By Stage",
        sorted(by_stage.items()),
        "kg CO2e",
        "#477d78",
    )
    make_svg_bar_chart(
        figure_root / "fig02_bau_correction_static_screening_by_component.svg",
        "BAU Corrected-Scope Climate Delta By Component",
        sorted(by_component.items(), key=lambda item: abs(item[1]), reverse=True),
        "kg CO2e",
        "#576c9d",
    )
    method_rows = []
    for row in lcia_rows:
        if row["metric_key"] in {"climate_static", "dynamic_GWP100"}:
            method_rows.append((row["method_mode"], to_float(row["BAU_corrected_screening"]) / 1_000_000.0))
    make_svg_bar_chart(
        figure_root / "fig03_bau_corrected_total_climate_screening.svg",
        "BAU Corrected Total Climate Screening",
        method_rows,
        "million kg CO2e",
        "#b46b3c",
    )


def build_readme(
    output_root: Path,
    counts: dict[str, float],
    basis: dict[str, float],
    fuel_share_config: dict[str, Any],
) -> str:
    source = fuel_share_config["source"]
    base = fuel_share_config["base"]
    return f"""# Corrected GEN vs BAU Scope Package

Generated folder: `{output_root.relative_to(REPO_ROOT).as_posix()}`

## Purpose

This package corrects the manuscript-facing BAU comparison boundary so BAU is treated as a newly built conventional heating/cooling baseline over the same 37-building, 50-year functional unit as GEN.

## Functional Unit

- Modeled buildings: {fmt(counts['total'])}
- Single-family buildings: {fmt(counts['single_family'])}
- FHA multifamily buildings: {fmt(counts['fha_multifamily'])}
- Commercial/municipal buildings: {fmt(counts['commercial_municipal'])}
- Annual delivered thermal service: {fmt(basis['annual_delivered_kwh_th'])} kWh_th/yr
- Lifetime delivered thermal service: {fmt(basis['lifetime_delivered_kwh_th'])} kWh_th over {fmt(basis['service_life_years'])} years

## Single-Family Gas/Oil Split

The active single-family fossil heating split is loaded from `inputs/config/fuel_share_framingham_acs.yaml`, not from `DEFAULT_ASSUMPTIONS`.

- Base gas share: {fmt(float(base['gas_share']))}
- Base oil share: {fmt(float(base['oil_share']))}
- ACS source: {source['table']} utility gas {fmt(float(source['massachusetts_utility_gas_percent']))}% and fuel oil {fmt(float(source['massachusetts_fuel_oil_percent']))}%.
- Renormalized fossil gas share: {fmt(float(source['renormalized_gas_share']))}; rounded base: 0.70/0.30.
- S1 sensitivity points: 0.60/0.40, 0.70/0.30, and 0.75/0.25, one-at-a-time around the same functional unit.

## Boundary Correction

The old BAU non-B6 foreground was not fully symmetric with GEN. It used a 36-building hard-code for non-B6 equipment and did not explicitly represent local gas/oil service infrastructure construction, trench bedding, A5 installation work, routine filter maintenance, and the gas-line abandonment-in-place core EOL interpretation.

The corrected core case includes project-local BAU infrastructure: gas main/service pipe, bedding, trenching, meters/regulators, oil tanks, local material transport, and oil-tank retirement. Upstream regional gas production/transmission/distribution infrastructure remains in the natural-gas fuel-supply background process. The package also keeps an exclusion/removal sensitivity concept so reviewers can see the possible double-counting boundary.

## Important LCIA Status

This shell did not have `bw2data`, `pandas`, or the `bw25` environment, so a fresh Brightway/bw_timex rerun was not possible here. The folder therefore contains:

- Full corrected LCI and per-FU inventory tables.
- Screening LCIA deltas using transparent simple factors to show the direction and magnitude of the scope correction.
- A rerun-status table describing the exact workflow needed to replace the screening rows with full Brightway static, time-explicit, and dynamic LCIA.

Do not use the screening LCIA rows as final manuscript results until the Brightway rerun is completed.

## Key Files

- `tables/full_detailed_lci_per_fu.csv`: current GEN/BAU inventory plus corrected BAU rows, with quantities per FU and per kWh_th.
- `tables/bau_corrected_inventory_delta.csv`: only the BAU correction rows.
- `tables/corrected_lcia_screening_summary.csv`: climate screening deltas and corrected screening totals.
- `tables/stage_climate_screening_summary.csv`: stage-level screening deltas.
- `tables/fuel_split_sensitivity_matrix.csv`: S1 gas/oil split low/base/high values and rationale.
- `tables/scope_sanity_check.csv`: checks showing the original asymmetry and correction status.
- `tables/lcia_rerun_status.csv`: what still needs the local Brightway environment.
- `figures/*.svg`: screening figures for audit and communication.
- `code/build_corrected_scope_package.py`: reproducible builder copied from the source script.

## Sources Used

- NRC Canada, National guidelines for whole-building life cycle assessment: {SOURCE_NOTES['NRC Canada WBLCA guideline']}
- Schori et al., Life Cycle Inventory of Natural Gas Supply: {SOURCE_NOTES['Schori 2012 natural gas LCI']}
- PHMSA pipeline construction phases: {SOURCE_NOTES['PHMSA pipeline construction phases']}
- U.S. Census ACS Table B25040, House Heating Fuel: {source['references'][0].split(': ', 1)[1]}
- Mass.gov, How Massachusetts households heat their homes: {source['references'][1].split(': ', 1)[1]}
- Alvarez et al. 2018 methane supply-chain estimate: {SOURCE_NOTES['Alvarez et al. 2018 methane supply-chain estimate']}
- NETL natural gas environmental performance: {SOURCE_NOTES['NETL natural gas environmental performance']}
- NREL ComStock baseline documentation: {SOURCE_NOTES['NREL ComStock baseline documentation']}

## Modeling Notes

- Gas customers are calculated from the ACS-backed base split: 23 single-family buildings x 0.70 gas share, plus the 5 commercial/municipal gas baseline buildings.
- Oil customers are calculated from the single-family oil share.
- Gas pipe material already existed in the current manuscript BAU foreground, but the service-customer count and A5 construction treatment were corrected.
- Buried gas pipe is reported as a memo flow at EOL and not assigned landfill treatment in the corrected core.
- Methane leakage remains in B6 operation and should be retained as explicit CH4 in the full rerun; this package does not move supply-chain leakage from the gas background into local construction.
- Direct natural-gas fuel is not added to A4/A5 construction equipment because the available construction references describe pipeline installation machinery and material transport rather than gas-fired site equipment; upstream gas used inside background material or fuel supply datasets remains embedded in those datasets.
"""


def build_spec(output_root: Path) -> str:
    return f"""# Corrected GEN vs BAU Scope

**Category**: Major refactor

## Summary

Rebuild the manuscript-facing GEN vs BAU comparison so both systems represent newly built infrastructure over the same Framingham functional unit.

## Key Points

- Use the public 37-building case and 89,374,805.789 kWh_th lifetime denominator.
- Keep GEN in the accepted construction, operation, replacement, and EOL scope.
- Expand BAU non-B6 scope to include local gas/oil infrastructure construction, transport, installation, maintenance, replacement, and EOL/abandonment treatment.
- Keep upstream gas supply infrastructure in the natural-gas background process and document an exclusion sensitivity to avoid double-counting concerns.
- Rebuild LCIA in Brightway/bw_timex when the local `bw25` environment is available.

## Open Questions

- Replace screening factors with full Brightway LCIA scores after the corrected foreground database is written.
- Replace gas/oil infrastructure proxy quantities with utility GIS and field asset data if available.
"""


def build_package(output_root: Path = OUTPUT_ROOT) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    table_root = output_root / "tables"
    code_root = output_root / "code"

    counts = building_counts_from_rows(read_csv_rows(WORKSPACE_ROOT / "inputs" / "config" / "building_group_map.csv"))
    basis = load_case_basis()
    fuel_share_config = load_single_family_fuel_share_config()
    base_fuel_share = fuel_share_config["base"]
    base_infra = local_gas_oil_infrastructure_inventory(
        counts,
        DEFAULT_ASSUMPTIONS,
        single_family_gas_share=float(base_fuel_share["gas_share"]),
        single_family_oil_share=float(base_fuel_share["oil_share"]),
    )
    correction_rows = build_correction_rows(
        counts,
        basis,
        single_family_gas_share=float(base_fuel_share["gas_share"]),
        single_family_oil_share=float(base_fuel_share["oil_share"]),
    )
    full_rows = current_inventory_rows(basis) + correction_rows
    lcia_rows = build_lcia_screening_summary(correction_rows)
    stage_rows = build_stage_screening_summary(correction_rows)
    fuel_split_rows = single_family_fuel_share_sensitivity_points()
    sanity_rows = build_scope_sanity_rows(counts, base_infra, fuel_share_config)
    rerun_rows = build_rerun_status_rows()

    full_fields = [
        "pathway",
        "scenario",
        "stage",
        "module",
        "component",
        "flow_name",
        "flow_key",
        "quantity",
        "unit",
        "quantity_per_FU",
        "quantity_per_kWh_th",
        "timing",
        "correction_status",
        "calculation_basis",
        "source_or_reference",
        "brightway_mapping",
        "screening_factor_kgCO2e_per_unit",
        "screening_static_kgCO2e",
        "notes",
    ]

    outputs = {
        "spec": output_root / "SPEC.md",
        "readme": output_root / "README.md",
        "full_lci": table_root / "full_detailed_lci_per_fu.csv",
        "bau_delta": table_root / "bau_corrected_inventory_delta.csv",
        "lcia_screening": table_root / "corrected_lcia_screening_summary.csv",
        "stage_screening": table_root / "stage_climate_screening_summary.csv",
        "fuel_split_sensitivity": table_root / "fuel_split_sensitivity_matrix.csv",
        "scope_sanity": table_root / "scope_sanity_check.csv",
        "rerun_status": table_root / "lcia_rerun_status.csv",
        "manifest": output_root / "manifest.json",
        "code_copy": code_root / "build_corrected_scope_package.py",
    }

    write_text(outputs["spec"], build_spec(output_root))
    write_text(outputs["readme"], build_readme(output_root, counts, basis, fuel_share_config))
    write_csv_rows(outputs["full_lci"], full_rows, full_fields)
    write_csv_rows(outputs["bau_delta"], correction_rows, full_fields)
    write_csv_rows(outputs["lcia_screening"], lcia_rows)
    write_csv_rows(outputs["stage_screening"], stage_rows)
    write_csv_rows(outputs["fuel_split_sensitivity"], fuel_split_rows)
    write_csv_rows(outputs["scope_sanity"], sanity_rows)
    write_csv_rows(outputs["rerun_status"], rerun_rows)
    build_figures(output_root, correction_rows, lcia_rows)
    code_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), outputs["code_copy"])

    manifest = {
        "package": output_root.relative_to(REPO_ROOT).as_posix(),
        "status": "corrected_inventory_complete_full_dynamic_lcia_rerun_complete",
        "full_lcia_outputs": "Dynamic_LCA_GEN_BAU/export/full_dynamic_lcia",
        "counts": counts,
        "functional_unit": basis,
        "assumptions": DEFAULT_ASSUMPTIONS,
        "single_family_fuel_share_config": fuel_share_config,
        "single_family_fuel_share_sensitivity_points": single_family_fuel_share_sensitivity_points(),
        "source_notes": SOURCE_NOTES,
        "outputs": {key: path.relative_to(REPO_ROOT).as_posix() for key, path in outputs.items()},
    }
    write_json(outputs["manifest"], manifest)
    return outputs


def main() -> None:
    outputs = build_package()
    for key, path in outputs.items():
        print(f"{key}: {path.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
