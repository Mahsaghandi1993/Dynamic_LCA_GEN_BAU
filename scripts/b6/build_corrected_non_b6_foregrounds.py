from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.build_corrected_scope_package import (  # type: ignore
        DEFAULT_ASSUMPTIONS,
        building_counts_from_rows,
        equipment_masses_for_counts,
        load_case_basis,
        load_single_family_fuel_share_config,
        local_gas_oil_infrastructure_inventory,
        read_csv_rows,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        REPO_ROOT,
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        relpath,
        write_json,
    )
else:
    from .build_corrected_scope_package import (
        DEFAULT_ASSUMPTIONS,
        building_counts_from_rows,
        equipment_masses_for_counts,
        load_case_basis,
        load_single_family_fuel_share_config,
        local_gas_oil_infrastructure_inventory,
        read_csv_rows,
    )
    from .common import (
        REPO_ROOT,
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        relpath,
        write_json,
    )


BG_DB = "ecoinvent_2025_SSP2-PkBudg1000"
GEN_FG_DB = "gen_dynamic_fg"
BAU_FG_DB = "bau_dynamic_fg"
BIO_DB = "biosphere3"
NOTEBOOK_INPUTS = REPO_ROOT / "notebooks" / "inputs"
OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "manuscript_scope_corrected_2026_06_02"
TABLE_ROOT = OUTPUT_ROOT / "tables"
LCI_EXPORT_ROOT = OUTPUT_ROOT / "brightway_lci_exports" / "source_foregrounds"


GEN_A_PER_KWH = {
    "hdpe_granulate_market": 2.83e-4,
    "pipe_extrusion_market": 2.83e-4,
    "tap_water_market": 4.02e-4,
    "propylene_glycol_market": 1.39e-4,
    "heat_pump_brinewater_10kW": 1.03e-6,
    "pump_40W_market": 1.10e-5,
    "steel_low_alloyed_market_total": 8.46e-4 + 1.02e-7,
    "bentonite_market": 3.13e-3,
    "gravel_round_market": 1.57e-4,
    "diesel_building_machine_MJ": 2.25e-1,
    "freight_lorry_gt32t": 1.50e-4,
}

_GEN_LIFE = float(os.environ.get("GEN_LONGLIVED_LIFE_YEARS", "50"))
if _GEN_LIFE != 50.0:
    _gen_life_scale = 50.0 / _GEN_LIFE
    for _key in ("bentonite_market", "hdpe_granulate_market", "pipe_extrusion_market", "gravel_round_market", "diesel_building_machine_MJ"):
        GEN_A_PER_KWH[_key] = GEN_A_PER_KWH[_key] * _gen_life_scale

GEN_B_HP_PER_KWH = {
    "heat_pump_brinewater_10kW": 1.03e-6,
    "steel_low_alloyed_market": 1.02e-7,
}
GEN_B_PUMP_PER_KWH = {"pump_40W_market": 3.29e-5}
GEN_B_FLUID_PER_KWH = {
    "tap_water_market": 3.02e-4,
    "propylene_glycol_market": 1.04e-4,
}
GEN_B_MISC_PER_KWH = {
    "refrigerant_r134a_market": 5.00e-7,
    "chemical_inorganic_market": 1.00e-7,
    "wastewater_average_market": 1.00e-4,
}
GEN_C_PER_KWH = {
    "bentonite_market": 1.57e-4,
    "waste_polyethylene_sanitary_landfill": 1.42e-4,
    "scrap_steel_market_recycling_proxy": 8.46e-4,
    "fluid_disposal_hazard_incineration": 5.41e-4,
}

EXPANDED_INFRASTRUCTURE_ASSUMPTIONS = {
    "scenario_name": "expanded_project_infrastructure_2026_06_02",
    "gen_pumphouse_floor_area_m2": 60.0,
    "gen_pumphouse_building_proxy": "market for building, hall, steel construction",
    "bau_indoor_gas_piping_length_per_gas_customer_m": 20.0,
    "bau_indoor_gas_pipe_steel_kg_per_m": 1.5,
    "bau_indoor_gas_pipe_lorry_distance_km": 100.0,
    "bau_storage_compressor_station_floor_area_m2": 30.0,
    "bau_storage_compressor_station_building_proxy": "market for building, hall, steel construction",
    "bau_allocated_air_compressor_300kw_unit_fraction": 0.01,
    "bau_allocated_high_pressure_pipeline_interconnect_km": 0.05,
    "bau_storage_cavern_note": "No exact natural-gas storage cavern construction activity was available in the local ecoinvent/premise background; storage/compressor infrastructure is represented with a small steel-hall building proxy, allocated compressor-equipment proxy, and high-pressure pipeline interconnect proxy.",
}


HVAC_TRAIN_DISTANCE_KM = 2300.0
HVAC_TRUCK_DISTANCE_KM = 70.0
DUCT_TRUCK_DISTANCE_KM = 50.0
GAS_PIPE_TRUCK_DISTANCE_KM = 100.0
ANNUAL_REFRIG_LEAKAGE_RATE = 0.02
EOL_REFRIG_RECLAMATION_FRACTION = 0.80
EOL_REFRIG_LOSS_FRACTION = 0.20


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dedupe_exchanges(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[tuple[str, str], str], float] = defaultdict(float)
    kept: dict[tuple[tuple[str, str], str], dict[str, Any]] = {}
    for exc in exchanges:
        key = (tuple(exc["input"]), str(exc["type"]))
        grouped[key] += float(exc["amount"])
        kept.setdefault(key, dict(exc))
    out = []
    for key, exc in kept.items():
        exc["amount"] = grouped[key]
        out.append(exc)
    return out


def _add_dicts(*payloads: dict[str, float]) -> dict[str, float]:
    out: defaultdict[str, float] = defaultdict(float)
    for payload in payloads:
        for key, value in payload.items():
            out[key] += float(value)
    return dict(out)


def _per_kwh(payload: dict[str, float], denominator: float) -> dict[str, float]:
    return {key: float(value) / denominator for key, value in payload.items()}


class BackgroundResolver:
    def __init__(self, bd: Any, bg_db: str = BG_DB) -> None:
        self.bd = bd
        self.bg_db = bg_db
        self.bg = bd.Database(bg_db)
        self.bio = bd.Database(BIO_DB)
        self.bg_codes = {activity["code"] for activity in self.bg}
        self.gen_mapping = _load_json(NOTEBOOK_INPUTS / "gen_bg_mapping_2025_FULL.json")
        self.bau_mapping = _load_json(NOTEBOOK_INPUTS / "bau_bg_mapping_2025_FULL.json")
        self.resolved: dict[str, tuple[str, str]] = {}

    def mapped(self, mapping_name: str, stage_group: str, key: str) -> tuple[str, str]:
        mapping = self.gen_mapping if mapping_name == "gen" else self.bau_mapping
        rec = mapping[stage_group][key]
        value = (str(rec["db"]), str(rec["code"]))
        if value[0] == self.bg_db and value[1] not in self.bg_codes:
            value = self.fallback(stage_group, key, rec)
        self.resolved[f"{mapping_name}.{stage_group}.{key}"] = value
        return value

    def scan(self, *subs: str, limit: int = 100) -> list[Any]:
        terms = [term.lower() for term in subs]
        hits = []
        for activity in self.bg:
            text = " | ".join(
                [
                    activity.get("name", ""),
                    activity.get("reference product", ""),
                    activity.get("unit", ""),
                    activity.get("location", ""),
                ]
            ).lower()
            if all(term in text for term in terms):
                hits.append(activity)
                if len(hits) >= limit:
                    break
        return hits

    @staticmethod
    def _without(candidates: list[Any], *forbidden: str) -> list[Any]:
        terms = [term.lower() for term in forbidden]
        out = []
        for activity in candidates:
            text = " | ".join(
                [
                    activity.get("name", ""),
                    activity.get("reference product", ""),
                    activity.get("unit", ""),
                    activity.get("location", ""),
                ]
            ).lower()
            if not any(term in text for term in terms):
                out.append(activity)
        return out

    def pick(
        self,
        candidates: list[Any],
        *,
        label: str,
        must: list[str] | None = None,
        prefer: list[str] | None = None,
        unit: str | None = None,
    ) -> tuple[str, str]:
        must_terms = [item.lower() for item in (must or [])]
        prefer_terms = [item.lower() for item in (prefer or [])]
        scored = []
        for activity in candidates:
            text = " | ".join(
                [
                    activity.get("name", ""),
                    activity.get("reference product", ""),
                    activity.get("unit", ""),
                    activity.get("location", ""),
                ]
            ).lower()
            if any(term not in text for term in must_terms):
                continue
            if unit and activity.get("unit") != unit:
                continue
            score = sum(term in text for term in prefer_terms)
            scored.append((score, activity))
        if not scored:
            raise RuntimeError(f"No background activity matched `{label}`.")
        scored.sort(key=lambda item: (-item[0], item[1].get("location", ""), item[1].get("name", "")))
        chosen = scored[0][1]
        value = (chosen["database"], chosen["code"])
        self.resolved[label] = value
        return value

    def fallback(self, stage_group: str, key: str, rec: dict[str, Any]) -> tuple[str, str]:
        label = f"fallback.{stage_group}.{key}"
        if key in {"steel_low_alloyed_market", "steel_casing_market", "hx_steel_proxy_market"}:
            return self.pick(
                self.scan("market for steel", "low-alloyed", "hot rolled"),
                label=label,
                prefer=["glo", "market for"],
                unit="kilogram",
            )
        if key == "hdpe_granulate_market":
            candidates = self.scan("market for polyethylene", "high density", "granulate")
            virgin = self._without(candidates, "recycled")
            return self.pick(
                virgin or candidates,
                label=label,
                prefer=["glo", "market for"],
                unit="kilogram",
            )
        if key == "pipe_extrusion_market":
            return self.pick(
                self.scan("market for extrusion", "plastic pipes") or self.scan("extrusion", "plastic pipes"),
                label=label,
                prefer=["glo", "market for", "row"],
                unit="kilogram",
            )
        if key == "tap_water_market":
            return self.pick(
                self.scan("market for tap water"),
                label=label,
                prefer=["row", "glo", "rer"],
                unit="kilogram",
            )
        if key == "propylene_glycol_market":
            return self.pick(
                self.scan("market for propylene glycol", "liquid"),
                label=label,
                prefer=["row", "market for"],
                unit="kilogram",
            )
        if key in {"heat_pump_brinewater_10kW", "hp_proxy_for_AC"}:
            return self.pick(
                self.scan("heat pump production", "brine-water", "10kW"),
                label=label,
                prefer=["row", "production"],
                unit="unit",
            )
        if key == "pump_40W_market":
            return self.pick(
                self.scan("market for pump", "40W"),
                label=label,
                prefer=["glo", "market for"],
                unit="unit",
            )
        if key == "bentonite_market":
            return self.pick(
                self.scan("market for bentonite"),
                label=label,
                prefer=["glo", "world", "market for"],
                unit="kilogram",
            )
        if key == "gravel_round_market":
            return self.pick(
                self.scan("market for gravel", "round"),
                label=label,
                prefer=["row", "market for"],
                unit="kilogram",
            )
        if key == "diesel_building_machine":
            return self.pick(
                self.scan("market for diesel", "burned in building machine")
                or self.scan("diesel", "burned in building machine"),
                label=label,
                prefer=["glo", "market for"],
                unit="megajoule",
            )
        if key == "freight_lorry_gt32t":
            return self.pick(
                self.scan("transport, freight, lorry", ">32"),
                label=label,
                prefer=["euro6", "row", "market for"],
                unit="ton kilometer",
            )
        if key == "refrigerant_r134a_market":
            if "used refrigerant" in str(rec.get("name", "")).lower():
                return self.pick(
                    self.scan("market for used refrigerant", "R134a"),
                    label=label,
                    prefer=["glo", "market for"],
                    unit="kilogram",
                )
            return self.pick(
                self.scan("market for refrigerant", "R134a"),
                label=label,
                prefer=["glo", "market for"],
                unit="kilogram",
            )
        if key == "chemical_inorganic_market":
            return self.pick(
                self.scan("market for chemical", "inorganic"),
                label=label,
                prefer=["glo", "market for"],
                unit="kilogram",
            )
        if key == "wastewater_average_market":
            return self.pick(
                self.scan("market for wastewater", "average"),
                label=label,
                prefer=["row", "market for"],
            )
        if key in {"scrap_steel_market_recycling_proxy", "used_heat_pump_disposal_proxy", "used_motor_disposal_proxy", "used_hx_disposal_proxy"}:
            return self.pick(
                self.scan("market for scrap steel"),
                label=label,
                prefer=["row", "market for"],
                unit="kilogram",
            )
        if key == "scrap_steel_landfill_disposal_proxy":
            return self.pick(
                self.scan("treatment of scrap steel", "inert material landfill"),
                label=label,
                prefer=["row"],
                unit="kilogram",
            )
        if key == "waste_polyethylene_sanitary_landfill":
            candidates = self.scan("treatment of waste polyethylene", "landfill")
            non_pet = self._without(candidates, "terephthalate", "pet")
            return self.pick(
                non_pet or candidates,
                label=label,
                prefer=["row", "sanitary"],
                unit="kilogram",
            )
        if key == "fluid_disposal_hazard_incineration":
            return self.pick(
                self.scan("treatment of spent solvent mixture", "hazardous waste incineration"),
                label=label,
                prefer=["row", "with energy recovery"],
                unit="kilogram",
            )
        if key == "waste_mineral_oil_hazard_incineration":
            return self.pick(
                self.scan("treatment of waste mineral oil", "hazardous waste incineration"),
                label=label,
                prefer=["row", "with energy recovery"],
                unit="kilogram",
            )
        raise RuntimeError(
            f"Saved mapping code for `{stage_group}.{key}` is stale and no fallback search is defined."
        )

    def find_r134a_air_flow(self) -> tuple[str, str]:
        terms = [
            "r134a",
            "hfc-134a",
            "hfc 134a",
            "1,1,1,2-tetrafluoroethane",
            "tetrafluoroethane",
        ]
        candidates = []
        for flow in self.bio:
            name = (flow.get("name") or "").lower()
            categories = " | ".join(flow.get("categories", ())).lower()
            if any(term in name for term in terms) and "air" in categories:
                candidates.append(flow)
        if not candidates:
            raise RuntimeError("Could not find a biosphere air emission flow for R134a/HFC-134a.")
        candidates.sort(
            key=lambda flow: (
                "urban air close to ground" not in " | ".join(flow.get("categories", ())).lower(),
                "non-urban air or from high stacks" not in " | ".join(flow.get("categories", ())).lower(),
                flow.get("name", ""),
            )
        )
        chosen = candidates[0]
        value = (chosen["database"], chosen["code"])
        self.resolved["biosphere.refrigerant_r134a_emission_to_air"] = value
        return value

    def bau_keys(self) -> dict[str, tuple[str, str]]:
        keys = {
            "steel": self.mapped("bau", "A1_A3", "steel_low_alloyed_market"),
            "r134a": self.mapped("bau", "B2_B4", "refrigerant_r134a_market"),
            "hdpe": self.mapped("gen", "A1_A3", "hdpe_granulate_market"),
            "pipe_extrusion": self.mapped("gen", "A1_A3", "pipe_extrusion_market"),
            "lorry": self.mapped("gen", "A4_A5", "freight_lorry_gt32t"),
            "pe_landfill": self.mapped("bau", "C1_C4", "waste_polyethylene_sanitary_landfill"),
            "scrap_steel": self.mapped("bau", "C1_C4", "scrap_steel_market_recycling_proxy"),
            "scrap_steel_landfill": self.mapped("bau", "C1_C4", "scrap_steel_landfill_disposal_proxy"),
        }
        keys["aluminum"] = self.pick(
            self.scan("market for aluminium") or self.scan("market for aluminum"),
            label="bau.aluminum_market",
            prefer=["row", "ingot", "market for"],
            unit="kilogram",
        )
        keys["copper"] = self.pick(
            self.scan("market for copper"),
            label="bau.copper_market",
            prefer=["row", "cathode", "market for"],
            unit="kilogram",
        )
        keys["train"] = self.pick(
            self.scan("transport, freight train"),
            label="bau.freight_train",
            prefer=["market group", "diesel", "electric"],
            unit="ton kilometer",
        )
        keys["scrap_aluminum"] = self.pick(
            self.scan("market for scrap aluminium") or self.scan("market for scrap aluminum"),
            label="bau.scrap_aluminum_market",
            prefer=["row", "market for"],
            unit="kilogram",
        )
        keys["scrap_copper"] = self.pick(
            self.scan("market for scrap copper"),
            label="bau.scrap_copper_market",
            prefer=["row", "market for"],
            unit="kilogram",
        )
        keys["refrig_reclaim"] = self.pick(
            self.scan("used refrigerant", "r134a", "reclamation"),
            label="bau.used_refrigerant_r134a_reclamation",
            prefer=["market for", "glo"],
            unit="kilogram",
        )
        keys["r134a_air"] = self.find_r134a_air_flow()
        keys["sand"] = self.pick(
            self.scan("market for sand"),
            label="bau.bedding_sand_market",
            prefer=["row", "world"],
            unit="kilogram",
        )
        keys["cement"] = self.pick(
            self.scan("market for cement"),
            label="bau.bedding_cement_market",
            prefer=["row", "cem v/a", "portland", "unspecified"],
            unit="kilogram",
        )
        keys["excavation"] = self.pick(
            self.scan("market for excavation", "hydraulic digger")
            or self.scan("excavation", "hydraulic digger"),
            label="bau.gas_trench_excavation",
            prefer=["market for", "glo", "row"],
            unit="cubic meter",
        )
        keys["diesel_building_machine"] = self.pick(
            self.scan("market for diesel", "burned in building machine")
            or self.scan("diesel", "burned in building machine"),
            label="bau.directional_drilling_diesel_proxy",
            prefer=["glo", "market for"],
            unit="megajoule",
        )
        keys["filter"] = self.pick(
            self.scan("market for glass fibre"),
            label="bau.hvac_filter_glass_fibre_proxy",
            prefer=["glo", "market for"],
            unit="kilogram",
        )
        keys["building_hall_steel"] = self.pick(
            self.scan("market for building, hall", "steel construction")
            or self.scan("building construction, hall", "steel construction"),
            label="expanded.building_hall_steel_construction",
            prefer=["glo", "market for", "row"],
            unit="square meter",
        )
        keys["air_compressor_300kw"] = self.pick(
            self.scan("market for air compressor", "300kW")
            or self.scan("air compressor production", "300kW"),
            label="expanded.air_compressor_300kw_proxy",
            prefer=["glo", "market for", "row"],
            unit="unit",
        )
        keys["gas_pipeline_high_pressure"] = self.pick(
            self.scan("market for pipeline, natural gas, high pressure distribution network")
            or self.scan("pipeline construction, natural gas, high pressure distribution network"),
            label="expanded.natural_gas_high_pressure_pipeline_proxy",
            prefer=["glo", "market for", "row"],
            unit="kilometer",
        )
        return keys

    def gen_expanded_keys(self) -> dict[str, tuple[str, str]]:
        return {
            "building_hall_steel": self.pick(
                self.scan("market for building, hall", "steel construction")
                or self.scan("building construction, hall", "steel construction"),
                label="expanded.gen_pumphouse_building_hall_steel_construction",
                prefer=["glo", "market for", "row"],
                unit="square meter",
            )
        }


def _td_years(*years: int):
    from bw_temporalis import TemporalDistribution

    return TemporalDistribution(
        date=np.array(list(years), dtype="timedelta64[Y]"),
        amount=np.ones(len(years), dtype=float) / len(years),
    )


def _td_uniform_years(start: int, stop: int):
    from bw_temporalis import TemporalDistribution

    years = np.arange(start, stop + 1, dtype=int)
    return TemporalDistribution(
        date=years.astype("timedelta64[Y]"),
        amount=np.ones(len(years), dtype=float) / len(years),
    )


def _add_td(consumer_key: tuple[str, str], producer_key: tuple[str, str], td: Any) -> None:
    from bw_timex.utils import add_temporal_distribution_to_exchange

    add_temporal_distribution_to_exchange(
        temporal_distribution=td,
        input_database=producer_key[0],
        input_code=producer_key[1],
        output_database=consumer_key[0],
        output_code=consumer_key[1],
    )


def _write_database(bd: Any, db_name: str, data: dict[tuple[str, str], dict[str, Any]]) -> None:
    if db_name in bd.databases:
        del bd.databases[db_name]
    db = bd.Database(db_name)
    db.write(data, process=False)
    meta = dict(bd.databases[db_name])
    depends = set(meta.get("depends", []))
    depends.update([BG_DB, BIO_DB])
    meta["depends"] = sorted(depends)
    meta["source_denominator"] = "legacy_non_b6_project"
    bd.databases[db_name] = meta
    bd.Database(db_name).process()


def _build_gen_foreground(bd: Any, resolver: BackgroundResolver, legacy_denominator: float) -> dict[str, Any]:
    def m(stage_group: str, key: str) -> tuple[str, str]:
        return resolver.mapped("gen", stage_group, key)

    expanded_keys = resolver.gen_expanded_keys()
    gen_total = (GEN_FG_DB, "GEN_TOTAL_nonB6")
    gen_a = (GEN_FG_DB, "GEN_A1A5_construction")
    gen_b_hp = (GEN_FG_DB, "GEN_B_hp_replacement")
    gen_b_pump = (GEN_FG_DB, "GEN_B_pump_replacements")
    gen_b_fluid = (GEN_FG_DB, "GEN_B_fluid_topups")
    gen_b_misc = (GEN_FG_DB, "GEN_B_refrig_chem_wastewater")
    gen_c = (GEN_FG_DB, "GEN_C1C4_EOL")

    data = {
        gen_a: {
            "name": "GEN A1-A5 construction & installation (non-B6)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_a, "amount": 1.0, "type": "production"},
                    {"input": m("A1_A3", "hdpe_granulate_market"), "amount": GEN_A_PER_KWH["hdpe_granulate_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "pipe_extrusion_market"), "amount": GEN_A_PER_KWH["pipe_extrusion_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "tap_water_market"), "amount": GEN_A_PER_KWH["tap_water_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "propylene_glycol_market"), "amount": GEN_A_PER_KWH["propylene_glycol_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "heat_pump_brinewater_10kW"), "amount": GEN_A_PER_KWH["heat_pump_brinewater_10kW"], "type": "technosphere"},
                    {"input": m("A1_A3", "pump_40W_market"), "amount": GEN_A_PER_KWH["pump_40W_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "steel_low_alloyed_market"), "amount": GEN_A_PER_KWH["steel_low_alloyed_market_total"], "type": "technosphere"},
                    {"input": m("A4_A5", "bentonite_market"), "amount": GEN_A_PER_KWH["bentonite_market"], "type": "technosphere"},
                    {"input": m("A4_A5", "gravel_round_market"), "amount": GEN_A_PER_KWH["gravel_round_market"], "type": "technosphere"},
                    {"input": m("A4_A5", "diesel_building_machine"), "amount": GEN_A_PER_KWH["diesel_building_machine_MJ"], "type": "technosphere"},
                    {"input": m("A4_A5", "freight_lorry_gt32t"), "amount": GEN_A_PER_KWH["freight_lorry_gt32t"], "type": "technosphere"},
                    {
                        "input": expanded_keys["building_hall_steel"],
                        "amount": EXPANDED_INFRASTRUCTURE_ASSUMPTIONS["gen_pumphouse_floor_area_m2"] / legacy_denominator,
                        "type": "technosphere",
                    },
                ]
            ),
        },
        gen_b_hp: {
            "name": "GEN B2-B4 heat pump replacement (year 25)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_b_hp, "amount": 1.0, "type": "production"},
                    {"input": m("A1_A3", "heat_pump_brinewater_10kW"), "amount": GEN_B_HP_PER_KWH["heat_pump_brinewater_10kW"], "type": "technosphere"},
                    {"input": m("A1_A3", "steel_low_alloyed_market"), "amount": GEN_B_HP_PER_KWH["steel_low_alloyed_market"], "type": "technosphere"},
                ]
            ),
        },
        gen_b_pump: {
            "name": "GEN B2-B4 pump replacements (years 15/30/45)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_b_pump, "amount": 1.0, "type": "production"},
                    {"input": m("A1_A3", "pump_40W_market"), "amount": GEN_B_PUMP_PER_KWH["pump_40W_market"], "type": "technosphere"},
                ]
            ),
        },
        gen_b_fluid: {
            "name": "GEN B2-B4 working fluid top-ups (spread over life)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_b_fluid, "amount": 1.0, "type": "production"},
                    {"input": m("A1_A3", "tap_water_market"), "amount": GEN_B_FLUID_PER_KWH["tap_water_market"], "type": "technosphere"},
                    {"input": m("A1_A3", "propylene_glycol_market"), "amount": GEN_B_FLUID_PER_KWH["propylene_glycol_market"], "type": "technosphere"},
                ]
            ),
        },
        gen_b_misc: {
            "name": "GEN B2-B4 refrigerant + chemicals + wastewater (spread over life)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_b_misc, "amount": 1.0, "type": "production"},
                    {"input": m("B2_B4", "refrigerant_r134a_market"), "amount": GEN_B_MISC_PER_KWH["refrigerant_r134a_market"], "type": "technosphere"},
                    {"input": m("B2_B4", "chemical_inorganic_market"), "amount": GEN_B_MISC_PER_KWH["chemical_inorganic_market"], "type": "technosphere"},
                    {"input": m("B2_B4", "wastewater_average_market"), "amount": GEN_B_MISC_PER_KWH["wastewater_average_market"], "type": "technosphere"},
                ]
            ),
        },
        gen_c: {
            "name": "GEN C1-C4 end-of-life (year 50)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_c, "amount": 1.0, "type": "production"},
                    {"input": m("A4_A5", "bentonite_market"), "amount": GEN_C_PER_KWH["bentonite_market"], "type": "technosphere"},
                    {"input": m("C1_C4", "waste_polyethylene_sanitary_landfill"), "amount": GEN_C_PER_KWH["waste_polyethylene_sanitary_landfill"], "type": "technosphere"},
                    {"input": m("C1_C4", "scrap_steel_market_recycling_proxy"), "amount": GEN_C_PER_KWH["scrap_steel_market_recycling_proxy"], "type": "technosphere"},
                    {"input": m("C1_C4", "fluid_disposal_hazard_incineration"), "amount": GEN_C_PER_KWH["fluid_disposal_hazard_incineration"], "type": "technosphere"},
                ]
            ),
        },
        gen_total: {
            "name": "GEN TOTAL non-B6 (A1-A5, B2-B4, C1-C4)",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": gen_total, "amount": 1.0, "type": "production"},
                    {"input": gen_a, "amount": 1.0, "type": "technosphere"},
                    {"input": gen_b_hp, "amount": 1.0, "type": "technosphere"},
                    {"input": gen_b_pump, "amount": 1.0, "type": "technosphere"},
                    {"input": gen_b_fluid, "amount": 1.0, "type": "technosphere"},
                    {"input": gen_b_misc, "amount": 1.0, "type": "technosphere"},
                    {"input": gen_c, "amount": 1.0, "type": "technosphere"},
                ]
            ),
        },
    }
    _write_database(bd, GEN_FG_DB, data)
    _add_td(gen_total, gen_a, _td_years(0))
    _add_td(gen_total, gen_b_hp, _td_years(25))
    _add_td(gen_total, gen_b_pump, _td_years(15, 30, 45))
    _add_td(gen_total, gen_b_fluid, _td_uniform_years(1, 50))
    _add_td(gen_total, gen_b_misc, _td_uniform_years(1, 50))
    _add_td(gen_total, gen_c, _td_years(50))
    bd.Database(GEN_FG_DB).process()
    return {"db": GEN_FG_DB, "total": gen_total, "source_denominator": legacy_denominator}


def _split_eol_streams(stock_kg: dict[str, float], *, include_polyethylene_landfill: bool) -> dict[str, float]:
    steel = stock_kg.get("steel_low_alloyed", 0.0) + stock_kg.get("galvanized_steel", 0.0)
    aluminum = stock_kg.get("aluminum", 0.0)
    copper = stock_kg.get("copper", 0.0)
    refrigerant = stock_kg.get("refrigerant_r134a", 0.0)
    plastics = stock_kg.get("hdpe_pipe_material", 0.0) if include_polyethylene_landfill else 0.0
    return {
        "scrap_steel_recycling": steel * 0.617,
        "scrap_aluminum_recycling": aluminum * 0.90,
        "scrap_copper_recycling": copper * 0.41,
        "refrigerant_reclamation": refrigerant * EOL_REFRIG_RECLAMATION_FRACTION,
        "refrigerant_vented": refrigerant * EOL_REFRIG_LOSS_FRACTION,
        "polyethylene_landfill": plastics,
    }


def _bau_material_exchanges(
    activity_key: tuple[str, str],
    mass_dict: dict[str, float],
    keys: dict[str, tuple[str, str]],
    *,
    include_transport: bool,
    label: str,
) -> list[dict[str, Any]]:
    exchanges = [{"input": activity_key, "amount": 1.0, "type": "production"}]
    steel = mass_dict.get("steel_low_alloyed", 0.0) + mass_dict.get("galvanized_steel", 0.0)
    if steel > 0:
        exchanges.append({"input": keys["steel"], "amount": steel, "type": "technosphere"})
    if mass_dict.get("aluminum", 0.0) > 0:
        exchanges.append({"input": keys["aluminum"], "amount": mass_dict["aluminum"], "type": "technosphere"})
    if mass_dict.get("copper", 0.0) > 0:
        exchanges.append({"input": keys["copper"], "amount": mass_dict["copper"], "type": "technosphere"})
    if mass_dict.get("refrigerant_r134a", 0.0) > 0:
        exchanges.append({"input": keys["r134a"], "amount": mass_dict["refrigerant_r134a"], "type": "technosphere"})
    if mass_dict.get("hdpe_pipe_material", 0.0) > 0:
        exchanges.append({"input": keys["hdpe"], "amount": mass_dict["hdpe_pipe_material"], "type": "technosphere"})
        exchanges.append({"input": keys["pipe_extrusion"], "amount": mass_dict["hdpe_pipe_material"], "type": "technosphere"})
    if include_transport:
        equipment_mass = sum(value for key, value in mass_dict.items() if key != "hdpe_pipe_material")
        if equipment_mass > 0:
            exchanges.append({"input": keys["train"], "amount": (equipment_mass / 1000.0) * HVAC_TRAIN_DISTANCE_KM, "type": "technosphere"})
            exchanges.append({"input": keys["lorry"], "amount": (equipment_mass / 1000.0) * HVAC_TRUCK_DISTANCE_KM, "type": "technosphere"})
        duct_mass = mass_dict.get("galvanized_steel", 0.0) if "duct" in label.lower() else 0.0
        if duct_mass > 0:
            exchanges.append({"input": keys["lorry"], "amount": (duct_mass / 1000.0) * DUCT_TRUCK_DISTANCE_KM, "type": "technosphere"})
        pipe_mass = mass_dict.get("hdpe_pipe_material", 0.0)
        if pipe_mass > 0:
            exchanges.append({"input": keys["lorry"], "amount": (pipe_mass / 1000.0) * GAS_PIPE_TRUCK_DISTANCE_KM, "type": "technosphere"})
    return _dedupe_exchanges(exchanges)


def _bau_eol_exchanges(activity_key: tuple[str, str], eol_dict: dict[str, float], keys: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    exchanges = [{"input": activity_key, "amount": 1.0, "type": "production"}]
    if eol_dict.get("scrap_steel_recycling", 0.0) > 0:
        exchanges.append({"input": keys["scrap_steel"], "amount": eol_dict["scrap_steel_recycling"], "type": "technosphere"})
    if eol_dict.get("scrap_aluminum_recycling", 0.0) > 0:
        exchanges.append({"input": keys["scrap_aluminum"], "amount": eol_dict["scrap_aluminum_recycling"], "type": "technosphere"})
    if eol_dict.get("scrap_copper_recycling", 0.0) > 0:
        exchanges.append({"input": keys["scrap_copper"], "amount": eol_dict["scrap_copper_recycling"], "type": "technosphere"})
    if eol_dict.get("polyethylene_landfill", 0.0) > 0:
        exchanges.append({"input": keys["pe_landfill"], "amount": eol_dict["polyethylene_landfill"], "type": "technosphere"})
    if eol_dict.get("refrigerant_reclamation", 0.0) > 0:
        exchanges.append({"input": keys["refrig_reclaim"], "amount": eol_dict["refrigerant_reclamation"], "type": "technosphere"})
    if eol_dict.get("refrigerant_vented", 0.0) > 0:
        exchanges.append({"input": keys["r134a_air"], "amount": eol_dict["refrigerant_vented"], "type": "biosphere"})
    return _dedupe_exchanges(exchanges)


def _build_bau_foreground(
    bd: Any,
    resolver: BackgroundResolver,
    legacy_denominator: float,
    accepted_denominator: float,
) -> dict[str, Any]:
    counts = building_counts_from_rows(read_csv_rows(WORKSPACE_ROOT / "inputs" / "config" / "building_group_map.csv"))
    equipment = equipment_masses_for_counts(counts)
    fuel_share_config = load_single_family_fuel_share_config()
    base_fuel_share = fuel_share_config["base"]
    infra = local_gas_oil_infrastructure_inventory(
        counts,
        DEFAULT_ASSUMPTIONS,
        single_family_gas_share=float(base_fuel_share["gas_share"]),
        single_family_oil_share=float(base_fuel_share["oil_share"]),
    )
    keys = resolver.bau_keys()
    expanded = dict(EXPANDED_INFRASTRUCTURE_ASSUMPTIONS)
    expanded["bau_gas_customers"] = infra["gas_customers"]
    expanded["bau_indoor_gas_piping_length_m"] = (
        infra["gas_customers"] * expanded["bau_indoor_gas_piping_length_per_gas_customer_m"]
    )
    expanded["bau_indoor_gas_pipe_steel_kg"] = (
        expanded["bau_indoor_gas_piping_length_m"] * expanded["bau_indoor_gas_pipe_steel_kg_per_m"]
    )
    expanded["bau_indoor_gas_pipe_lorry_tkm"] = (
        expanded["bau_indoor_gas_pipe_steel_kg"]
        / 1000.0
        * expanded["bau_indoor_gas_pipe_lorry_distance_km"]
    )

    initial_total = _add_dicts(equipment["initial_equipment"], {"hdpe_pipe_material": infra["gas_pipe_kg"]})
    replacement = dict(equipment["replacement_equipment"])
    duct_stock = {"galvanized_steel": float(equipment["initial_equipment"].get("galvanized_steel", 0.0) - replacement.get("galvanized_steel", 0.0))}
    final_equipment_stock = _add_dicts(replacement, duct_stock)

    a_per_kwh = _per_kwh(initial_total, legacy_denominator)
    b20_per_kwh = _per_kwh(replacement, legacy_denominator)
    b40_per_kwh = _per_kwh(replacement, legacy_denominator)
    c20_eol = _per_kwh(_split_eol_streams(replacement, include_polyethylene_landfill=False), legacy_denominator)
    c40_eol = _per_kwh(_split_eol_streams(replacement, include_polyethylene_landfill=False), legacy_denominator)
    c50_eol = _per_kwh(_split_eol_streams(final_equipment_stock, include_polyethylene_landfill=False), legacy_denominator)

    filter_kg = counts["total"] * DEFAULT_ASSUMPTIONS["filter_kg_per_hvac_system_year"] * DEFAULT_ASSUMPTIONS["service_life_years"]
    filter_tkm = filter_kg / 1000.0 * DEFAULT_ASSUMPTIONS["filter_truck_distance_km"]

    bau_a = (BAU_FG_DB, "BAU_A1A5_construction")
    bau_b20 = (BAU_FG_DB, "BAU_B2B4_replacement_y20")
    bau_b40 = (BAU_FG_DB, "BAU_B2B4_replacement_y40")
    bau_b_cons = (BAU_FG_DB, "BAU_B2B4_consumables_spread")
    bau_c20 = (BAU_FG_DB, "BAU_C1C4_EOL_removed_y20")
    bau_c40 = (BAU_FG_DB, "BAU_C1C4_EOL_removed_y40")
    bau_c50 = (BAU_FG_DB, "BAU_C1C4_EOL_final_y50")
    bau_total = (BAU_FG_DB, "BAU_TOTAL_nonB6")

    a_exchanges = _bau_material_exchanges(bau_a, a_per_kwh, keys, include_transport=True, label="A_initial")
    a_exchanges.extend(
        [
            {"input": keys["sand"], "amount": infra["bedding_sand_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["cement"], "amount": infra["bedding_cement_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["excavation"], "amount": infra["trench_excavation_m3"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["diesel_building_machine"], "amount": 23045.75 / legacy_denominator, "type": "technosphere"},
            {"input": keys["excavation"], "amount": 2.693 / legacy_denominator, "type": "technosphere"},
            {"input": keys["steel"], "amount": infra["meter_regulator_steel_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["steel"], "amount": infra["oil_tank_steel_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["lorry"], "amount": infra["local_material_lorry_tkm"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["lorry"], "amount": infra["meter_tank_lorry_tkm"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["steel"], "amount": expanded["bau_indoor_gas_pipe_steel_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["lorry"], "amount": expanded["bau_indoor_gas_pipe_lorry_tkm"] / legacy_denominator, "type": "technosphere"},
            {
                "input": keys["building_hall_steel"],
                "amount": expanded["bau_storage_compressor_station_floor_area_m2"] / legacy_denominator,
                "type": "technosphere",
            },
            {
                "input": keys["air_compressor_300kw"],
                "amount": expanded["bau_allocated_air_compressor_300kw_unit_fraction"] / legacy_denominator,
                "type": "technosphere",
            },
            {
                "input": keys["gas_pipeline_high_pressure"],
                "amount": expanded["bau_allocated_high_pressure_pipeline_interconnect_km"] / legacy_denominator,
                "type": "technosphere",
            },
        ]
    )
    annual_refrig_topup = float(initial_total.get("refrigerant_r134a", 0.0)) * ANNUAL_REFRIG_LEAKAGE_RATE
    cons_exchanges = [
        {"input": bau_b_cons, "amount": 1.0, "type": "production"},
        {"input": keys["r134a"], "amount": annual_refrig_topup * 50.0 / legacy_denominator, "type": "technosphere"},
        {"input": keys["r134a_air"], "amount": annual_refrig_topup * 50.0 / legacy_denominator, "type": "biosphere"},
        {"input": keys["filter"], "amount": filter_kg / legacy_denominator, "type": "technosphere"},
        {"input": keys["lorry"], "amount": filter_tkm / legacy_denominator, "type": "technosphere"},
    ]
    c50_exchanges = _bau_eol_exchanges(bau_c50, c50_eol, keys)
    c50_exchanges.extend(
        [
            {"input": keys["scrap_steel_landfill"], "amount": infra["oil_tank_steel_kg"] / legacy_denominator, "type": "technosphere"},
            {"input": keys["lorry"], "amount": infra["oil_tank_eol_lorry_tkm"] / legacy_denominator, "type": "technosphere"},
        ]
    )

    data = {
        bau_a: {
            "name": "BAU A1-A5 corrected new conventional equipment, local gas/oil infrastructure, and installation",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(a_exchanges),
        },
        bau_b20: {
            "name": "BAU B2-B4 corrected heating/cooling equipment replacement at year 20",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _bau_material_exchanges(bau_b20, b20_per_kwh, keys, include_transport=True, label="repl20"),
        },
        bau_b40: {
            "name": "BAU B2-B4 corrected heating/cooling equipment replacement at year 40",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _bau_material_exchanges(bau_b40, b40_per_kwh, keys, include_transport=True, label="repl40"),
        },
        bau_b_cons: {
            "name": "BAU B2-B4 corrected annual refrigerant top-up and filter maintenance",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(cons_exchanges),
        },
        bau_c20: {
            "name": "BAU C1-C4 EOL of removed equipment at year 20",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _bau_eol_exchanges(bau_c20, c20_eol, keys),
        },
        bau_c40: {
            "name": "BAU C1-C4 EOL of removed equipment at year 40",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _bau_eol_exchanges(bau_c40, c40_eol, keys),
        },
        bau_c50: {
            "name": "BAU C1-C4 corrected final EOL with gas line abandonment and oil tank disposal",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(c50_exchanges),
        },
        bau_total: {
            "name": "BAU TOTAL non-B6 corrected-scope conventional new construction baseline",
            "location": "US",
            "unit": "kilowatt hour",
            "reference product": "thermal energy delivered",
            "exchanges": _dedupe_exchanges(
                [
                    {"input": bau_total, "amount": 1.0, "type": "production"},
                    {"input": bau_a, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_b20, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_b40, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_b_cons, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_c20, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_c40, "amount": 1.0, "type": "technosphere"},
                    {"input": bau_c50, "amount": 1.0, "type": "technosphere"},
                ]
            ),
        },
    }

    _write_database(bd, BAU_FG_DB, data)
    _add_td(bau_total, bau_a, _td_years(0))
    _add_td(bau_total, bau_b20, _td_years(20))
    _add_td(bau_total, bau_b40, _td_years(40))
    _add_td(bau_total, bau_b_cons, _td_uniform_years(1, 50))
    _add_td(bau_total, bau_c20, _td_years(20))
    _add_td(bau_total, bau_c40, _td_years(40))
    _add_td(bau_total, bau_c50, _td_years(50))
    bd.Database(BAU_FG_DB).process()

    calculation_rows = []
    for stage_label, payload in [
        ("A_initial_corrected", initial_total),
        ("B20_replacement_corrected", replacement),
        ("B40_replacement_corrected", replacement),
        ("B_filters_total", {"filter_material": filter_kg, "filter_freight_lorry_tkm": filter_tkm}),
        ("C50_local_fuel_infrastructure", {"oil_tank_steel_disposal": infra["oil_tank_steel_kg"], "oil_tank_eol_lorry_tkm": infra["oil_tank_eol_lorry_tkm"], "gas_pipe_abandoned_memo_kg": infra["gas_pipe_kg"]}),
        ("A_local_gas_oil_infrastructure", infra),
        (
            "A_expanded_gas_storage_compressor_indoor_piping",
            {
                "directional_drilling_diesel_MJ": 23045.75,
                "storage_cavern_excavation_m3": 2.693,
                "indoor_gas_piping_length_m": expanded["bau_indoor_gas_piping_length_m"],
                "indoor_gas_pipe_steel_kg": expanded["bau_indoor_gas_pipe_steel_kg"],
                "indoor_gas_pipe_lorry_tkm": expanded["bau_indoor_gas_pipe_lorry_tkm"],
                "storage_compressor_station_floor_area_m2": expanded["bau_storage_compressor_station_floor_area_m2"],
                "allocated_air_compressor_300kw_unit_fraction": expanded["bau_allocated_air_compressor_300kw_unit_fraction"],
                "allocated_high_pressure_pipeline_interconnect_km": expanded["bau_allocated_high_pressure_pipeline_interconnect_km"],
            },
        ),
    ]:
        for flow_key, quantity in payload.items():
            calculation_rows.append(
                {
                    "case": "BAU",
                    "stage_block": stage_label,
                    "flow_key": flow_key,
                    "quantity_per_FU": quantity,
                    "source_amount_per_legacy_kWh_th": quantity / legacy_denominator,
                    "synced_amount_per_accepted_kWh_th": quantity / accepted_denominator,
                }
            )
    pd.DataFrame(calculation_rows).to_csv(TABLE_ROOT / "corrected_bau_brightway_foreground_calculation_table.csv", index=False)
    return {
        "db": BAU_FG_DB,
        "total": bau_total,
        "source_denominator": legacy_denominator,
        "counts": counts,
        "infrastructure": infra,
        "expanded_infrastructure": expanded,
    }


def _export_foreground_tables(
    bd: Any,
    *,
    legacy_denominator: float,
    accepted_denominator: float,
    resolver: BackgroundResolver,
) -> dict[str, Path]:
    from bw2io.export.csv import write_lci_csv

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    LCI_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for db_name, case in [(GEN_FG_DB, "GEN"), (BAU_FG_DB, "BAU")]:
        db = bd.Database(db_name)
        write_lci_csv(database_name=db_name, objs=[obj for obj in db], dirpath=LCI_EXPORT_ROOT.as_posix())
        for activity in db:
            for exc in activity.exchanges():
                input_activity = exc.input
                is_internal = input_activity.get("database") == db_name
                td = exc.get("temporal_distribution")
                rows.append(
                    {
                        "case": case,
                        "source_db": db_name,
                        "activity_code": activity["code"],
                        "activity_name": activity.get("name", ""),
                        "exchange_type": exc["type"],
                        "input_database": input_activity.get("database", ""),
                        "input_code": input_activity.get("code", ""),
                        "input_name": input_activity.get("name", ""),
                        "input_unit": input_activity.get("unit", ""),
                        "amount_per_legacy_kWh_th": float(exc["amount"]),
                        "quantity_per_FU": "" if is_internal or exc["type"] == "production" else float(exc["amount"]) * legacy_denominator,
                        "quantity_per_accepted_kWh_th_after_sync": "" if is_internal or exc["type"] == "production" else (float(exc["amount"]) * legacy_denominator / accepted_denominator),
                        "internal_link": bool(is_internal),
                        "has_temporal_distribution": td is not None,
                    }
                )
    summary_rows = []
    for db_name, case in [(GEN_FG_DB, "GEN"), (BAU_FG_DB, "BAU")]:
        db = bd.Database(db_name)
        summary_rows.append(
            {
                "case": case,
                "db": db_name,
                "activity_count": len(list(db)),
                "source_denominator_kwh_th": legacy_denominator,
                "accepted_denominator_kwh_th": accepted_denominator,
                "status": "written_and_processed",
            }
        )
    mapping_rows = []
    for label, key in sorted(resolver.resolved.items()):
        act = bd.get_activity(key)
        mapping_rows.append(
            {
                "mapping_label": label,
                "database": key[0],
                "code": key[1],
                "name": act.get("name", ""),
                "reference_product": act.get("reference product", ""),
                "unit": act.get("unit", ""),
                "location": act.get("location", ""),
            }
        )

    outputs = {
        "exchange_summary": TABLE_ROOT / "corrected_non_b6_brightway_exchange_summary.csv",
        "write_summary": TABLE_ROOT / "corrected_non_b6_brightway_write_summary.csv",
        "mapping": TABLE_ROOT / "corrected_non_b6_background_mapping.csv",
    }
    pd.DataFrame(rows).to_csv(outputs["exchange_summary"], index=False)
    pd.DataFrame(summary_rows).to_csv(outputs["write_summary"], index=False)
    pd.DataFrame(mapping_rows).to_csv(outputs["mapping"], index=False)
    return outputs


def build(write_exports: bool = True) -> dict[str, Path]:
    ensure_workspace_tree()
    config = load_b6_case_config()
    project = str(config["project"]["brightway_project"])
    legacy_denominator = float(config["bw_timex"]["normalization"]["existing_non_b6_lifetime_delivered_kwh_th"])
    accepted_denominator = float(load_case_basis()["lifetime_delivered_kwh_th"])

    try:
        import bw2data as bd
    except ImportError as exc:
        raise RuntimeError("Run this script from the bw25 environment with bw2data installed.") from exc

    bd.projects.set_current(project)
    missing = [db for db in [BIO_DB, BG_DB] if db not in bd.databases]
    if missing:
        raise RuntimeError(f"Missing required Brightway databases in `{project}`: {missing}")

    resolver = BackgroundResolver(bd, bg_db=BG_DB)
    gen = _build_gen_foreground(bd, resolver, legacy_denominator)
    bau = _build_bau_foreground(bd, resolver, legacy_denominator, accepted_denominator)

    outputs: dict[str, Path] = {}
    if write_exports:
        outputs.update(
            _export_foreground_tables(
                bd,
                legacy_denominator=legacy_denominator,
                accepted_denominator=accepted_denominator,
                resolver=resolver,
            )
        )

    manifest = {
        "project": project,
        "background_database": BG_DB,
        "source_foregrounds": {
            "GEN": gen,
            "BAU": {k: v for k, v in bau.items() if k not in {"counts", "infrastructure"}},
        },
        "counts": bau["counts"],
        "bau_local_infrastructure": bau["infrastructure"],
        "expanded_project_infrastructure_assumptions": EXPANDED_INFRASTRUCTURE_ASSUMPTIONS,
        "bau_expanded_project_infrastructure": bau.get("expanded_infrastructure", {}),
        "legacy_source_denominator_kwh_th": legacy_denominator,
        "accepted_sync_denominator_kwh_th": accepted_denominator,
        "status": "corrected_source_foregrounds_written",
        "notes": [
            "Source foregrounds are normalized to the legacy non-B6 denominator so the existing sync script preserves physical totals while moving to the accepted HEATNETS denominator.",
            "BAU Stage A now includes corrected 37-building HVAC equipment, local gas/oil service infrastructure, trench bedding, excavation, and transport.",
            "BAU Stage B now includes corrected replacements and routine filter maintenance in addition to refrigerant top-up.",
            "BAU Stage C treats buried local gas pipe as abandonment-in-place memo inventory and includes oil-tank disposal/transport.",
            "Expanded-infrastructure scenario adds GEN pumphouse building proxy, BAU indoor gas piping, and BAU project-allocated storage/compressor station proxy exchanges as Stage A year-0 pulses.",
        ],
    }
    manifest_path = OUTPUT_ROOT / "corrected_non_b6_foreground_manifest.json"
    write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Write corrected GEN and BAU non-B6 source foregrounds into the Brightway project.")
    parser.add_argument("--no-export", action="store_true", help="Skip CSV LCI exports and QA tables.")
    args = parser.parse_args()
    outputs = build(write_exports=not args.no_export)
    print("Corrected non-B6 foreground outputs:")
    for label, path in outputs.items():
        print(f" - {label}: {relpath(path)}")


if __name__ == "__main__":
    main()
