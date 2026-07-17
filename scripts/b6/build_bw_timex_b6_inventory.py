from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        EXPORT_ROOT,
        dump_yaml,
        ensure_workspace_tree,
        load_b6_case_config,
        relpath,
        validate_exists,
        write_json,
        write_markdown,
    )
else:
    from .common import (
        EXPORT_ROOT,
        dump_yaml,
        ensure_workspace_tree,
        load_b6_case_config,
        relpath,
        validate_exists,
        write_json,
        write_markdown,
    )


def _load_operational_totals() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gen_total = pd.read_csv(
        validate_exists(EXPORT_ROOT / "gen" / "gen_b6_hourly_total.csv", "GEN Stage B6 hourly total export"),
        parse_dates=["timestamp"],
    )
    bau_group = pd.read_csv(
        validate_exists(EXPORT_ROOT / "bau" / "bau_b6_hourly_group.csv", "BAU Stage B6 hourly group export"),
        parse_dates=["timestamp"],
    )
    common_service = pd.read_csv(
        validate_exists(EXPORT_ROOT / "bau" / "common_service_loads_hourly_building.csv", "common service-load export"),
        parse_dates=["timestamp"],
    )
    bau_total = (
        bau_group.groupby("timestamp", as_index=False)[
            [
                "BAU_heating_gas_kWh_fuel",
                "BAU_heating_oil_kWh_fuel",
                "BAU_heating_electric_kWh",
                "BAU_cooling_electric_kWh",
                "BAU_DHW_gas_kWh_fuel",
                "BAU_DHW_oil_kWh_fuel",
                "BAU_DHW_electric_kWh",
                "BAU_methane_leakage_mass_CH4",
            ]
        ]
        .sum()
        .sort_values("timestamp")
    )
    return gen_total, bau_total, common_service


def _annual_direct_combustion(bau_total: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    gas_cfg = config["bau"]["direct_combustion"]["natural_gas"]
    oil_cfg = config["bau"]["direct_combustion"]["fuel_oil"]
    gas = bau_total["BAU_heating_gas_kWh_fuel"].sum() + bau_total["BAU_DHW_gas_kWh_fuel"].sum()
    oil = bau_total["BAU_heating_oil_kWh_fuel"].sum() + bau_total["BAU_DHW_oil_kWh_fuel"].sum()
    return {
        "co2_kg": float(
            gas * float(gas_cfg["co2_kg_per_kwh_fuel"])
            + oil * float(oil_cfg["co2_kg_per_kwh_fuel"])
        ),
        "ch4_kg": float(
            gas * float(gas_cfg["ch4_kg_per_kwh_fuel"])
            + oil * float(oil_cfg["ch4_kg_per_kwh_fuel"])
            + bau_total["BAU_methane_leakage_mass_CH4"].sum()
        ),
        "n2o_kg": float(
            gas * float(gas_cfg["n2o_kg_per_kwh_fuel"])
            + oil * float(oil_cfg["n2o_kg_per_kwh_fuel"])
        ),
    }


def _denominator_summary(config: dict[str, Any], common_service: pd.DataFrame) -> pd.DataFrame:
    service_life = int(config["bau"]["service_life_years"])
    annual_delivered = float(common_service["delivered_total_kwh_th"].sum())
    heatnets_life = annual_delivered * service_life
    existing_non_b6 = float(
        config["bw_timex"]["normalization"]["existing_non_b6_lifetime_delivered_kwh_th"]
    )
    return pd.DataFrame(
        [
            {
                "mode": "heatnets_case_service_loads",
                "annual_delivered_kwh_th": annual_delivered,
                "lifetime_delivered_kwh_th": heatnets_life,
                "service_life_years": service_life,
                "note": "Audited 37-building HEATNETS-aligned service-load denominator.",
            },
            {
                "mode": "existing_non_b6_project",
                "annual_delivered_kwh_th": annual_delivered,
                "lifetime_delivered_kwh_th": existing_non_b6,
                "service_life_years": service_life,
                "note": "Current non-B6 notebook denominator retained for immediate Brightway compatibility.",
            },
        ]
    )


def _exchange_rows_for_modes(
    config: dict[str, Any],
    gen_total: pd.DataFrame,
    bau_total: pd.DataFrame,
    denominator_summary: pd.DataFrame,
) -> pd.DataFrame:
    service_life = int(config["bau"]["service_life_years"])
    gas_kwh_per_m3 = float(config["bau"]["natural_gas_kwh_per_m3"])
    oil_kwh_per_kg = float(config["bau"]["fuel_oil_kwh_per_kg"])

    annual_gen_elec = float(gen_total["GEN_total_B6_kWh_el_hourly"].sum())
    annual_gen_network = float(gen_total["GEN_HVAC_network_allocated_kWh_el_hourly"].sum())
    annual_gen_dhw = float(gen_total["GEN_DHW_kWh_el_hourly"].sum())

    annual_bau_gas = float(
        bau_total["BAU_heating_gas_kWh_fuel"].sum() + bau_total["BAU_DHW_gas_kWh_fuel"].sum()
    )
    annual_bau_oil = float(
        bau_total["BAU_heating_oil_kWh_fuel"].sum() + bau_total["BAU_DHW_oil_kWh_fuel"].sum()
    )
    annual_bau_elec = float(
        bau_total["BAU_heating_electric_kWh"].sum()
        + bau_total["BAU_cooling_electric_kWh"].sum()
        + bau_total["BAU_DHW_electric_kWh"].sum()
    )
    annual_direct = _annual_direct_combustion(bau_total, config)

    rows: list[dict[str, Any]] = []
    for denom in denominator_summary.to_dict(orient="records"):
        denominator = float(denom["lifetime_delivered_kwh_th"])
        mode = str(denom["mode"])

        gen_life_total = annual_gen_elec * service_life
        rows.extend(
            [
                {
                    "case": "GEN",
                    "normalization_mode": mode,
                    "exchange_key": "electricity_medium_voltage",
                    "exchange_kind": "technosphere",
                    "input_amount_total_over_life": gen_life_total,
                    "input_amount_per_kwh_th": gen_life_total / denominator,
                    "unit": "kilowatt hour",
                    "notes": f"Includes network ({annual_gen_network:.3f} kWh/y) and DHW ({annual_gen_dhw:.3f} kWh/y).",
                }
            ]
        )

        rows.extend(
            [
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "electricity_medium_voltage",
                    "exchange_kind": "technosphere",
                    "input_amount_total_over_life": annual_bau_elec * service_life,
                    "input_amount_per_kwh_th": (annual_bau_elec * service_life) / denominator,
                    "unit": "kilowatt hour",
                    "notes": "BAU electric heating, cooling, and DHW site energy.",
                },
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "natural_gas_high_pressure",
                    "exchange_kind": "technosphere",
                    "input_amount_total_over_life": (annual_bau_gas / gas_kwh_per_m3) * service_life,
                    "input_amount_per_kwh_th": ((annual_bau_gas / gas_kwh_per_m3) * service_life) / denominator,
                    "unit": "cubic meter",
                    "notes": "Natural gas supply only; direct combustion and leakage stay in biosphere flows.",
                },
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "light_fuel_oil",
                    "exchange_kind": "technosphere",
                    "input_amount_total_over_life": (annual_bau_oil / oil_kwh_per_kg) * service_life,
                    "input_amount_per_kwh_th": ((annual_bau_oil / oil_kwh_per_kg) * service_life) / denominator,
                    "unit": "kilogram",
                    "notes": "Fuel-oil supply only; direct combustion stays in biosphere flows.",
                },
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "carbon_dioxide_fossil",
                    "exchange_kind": "biosphere",
                    "input_amount_total_over_life": annual_direct["co2_kg"] * service_life,
                    "input_amount_per_kwh_th": (annual_direct["co2_kg"] * service_life) / denominator,
                    "unit": "kilogram",
                    "notes": "Direct operational CO2 from gas plus oil combustion.",
                },
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "methane_fossil",
                    "exchange_kind": "biosphere",
                    "input_amount_total_over_life": annual_direct["ch4_kg"] * service_life,
                    "input_amount_per_kwh_th": (annual_direct["ch4_kg"] * service_life) / denominator,
                    "unit": "kilogram",
                    "notes": "Direct operational CH4 including gas combustion and methane leakage.",
                },
                {
                    "case": "BAU",
                    "normalization_mode": mode,
                    "exchange_key": "dinitrogen_monoxide",
                    "exchange_kind": "biosphere",
                    "input_amount_total_over_life": annual_direct["n2o_kg"] * service_life,
                    "input_amount_per_kwh_th": (annual_direct["n2o_kg"] * service_life) / denominator,
                    "unit": "kilogram",
                    "notes": "Direct operational N2O from gas plus oil combustion.",
                },
            ]
        )
    return pd.DataFrame(rows)


def _scan_bg(bg, *subs: str, limit: int = 50):
    terms = [term.lower() for term in subs]
    hits = []
    for activity in bg:
        text = " | ".join(
            [
                activity.get("name", ""),
                activity.get("location", ""),
                activity.get("unit", ""),
                activity.get("reference product", ""),
            ]
        ).lower()
        if all(term in text for term in terms):
            hits.append(activity)
            if len(hits) >= limit:
                break
    return hits


def _pick_best(candidates, must: list[str] | None = None, prefer: list[str] | None = None):
    must = [item.lower() for item in (must or [])]
    prefer = [item.lower() for item in (prefer or [])]
    scored = []
    for activity in candidates:
        text = " | ".join(
            [
                activity.get("name", ""),
                activity.get("location", ""),
                activity.get("unit", ""),
                activity.get("reference product", ""),
            ]
        ).lower()
        if any(item not in text for item in must):
            continue
        score = sum(item in text for item in prefer)
        scored.append((score, activity))
    if not scored:
        raise RuntimeError(
            f"No background candidate satisfied must={must} from {len(candidates)} candidates."
        )
    scored.sort(key=lambda item: (-item[0], item[1].get("location", "")))
    return scored[0][1]


def _find_biosphere_flow(bd, name: str, prefer_categories: list[str]) -> tuple[str, str]:
    hits = []
    for flow in bd.Database("biosphere3"):
        if flow.get("name") != name:
            continue
        cats = " | ".join(flow.get("categories", ())).lower()
        score = sum(term.lower() in cats for term in prefer_categories)
        hits.append((score, flow))
    if not hits:
        raise KeyError(f"Could not find biosphere flow `{name}` in biosphere3.")
    hits.sort(key=lambda item: (-item[0], item[1].get("code", "")))
    chosen = hits[0][1]
    return (chosen["database"], chosen["code"])


def _write_brightway_foreground(
    config: dict[str, Any],
    exchange_table: pd.DataFrame,
    normalization_mode: str,
    foreground_set: str,
) -> tuple[dict[str, Any], list[Path]]:
    try:
        import bw2data as bd
        import numpy as np
        from bw_temporalis import TemporalDistribution
        from bw_timex.utils import add_temporal_distribution_to_exchange
    except ImportError as exc:
        raise RuntimeError(
            "Brightway/TimeX writing requires a Python environment with bw2data, bw_temporalis, and bw_timex installed. "
            "Run this script from your `bw25` environment."
        ) from exc

    project = config["project"]["brightway_project"]
    bd.projects.set_current(project)

    foreground_sets = config["bw_timex"]["foreground_sets"]
    if foreground_set not in foreground_sets:
        raise RuntimeError(
            f"Unsupported foreground set `{foreground_set}`. Supported: {sorted(foreground_sets)}"
        )
    gen_db_name = str(foreground_sets[foreground_set]["gen_fg_db"])
    bau_db_name = str(foreground_sets[foreground_set]["bau_fg_db"])
    bg_db_name = config["project"]["background_dbs"][0]
    fg_codes = config["bw_timex"]["foreground_activity_codes"]

    missing = [db_name for db_name in [gen_db_name, bau_db_name, bg_db_name] if db_name not in bd.databases]
    if missing:
        raise RuntimeError(f"Missing Brightway databases in project `{project}`: {missing}")

    bg = bd.Database(bg_db_name)
    electricity_market = _pick_best(
        _scan_bg(bg, "market for electricity", "medium voltage"),
        must=["market for electricity", "medium voltage"],
        prefer=["us-npcc", "us-rfc", "us", "npcc"],
    )
    natural_gas_market = _pick_best(
        _scan_bg(bg, "market for natural gas", "high pressure"),
        must=["market for natural gas", "high pressure"],
        prefer=["usa", " us ", "row"],
    )
    light_fuel_oil_market = _pick_best(
        _scan_bg(bg, "market for light fuel oil"),
        must=["market for light fuel oil"],
        prefer=["usa", "us", "row"],
    )
    mapping = {
        "electricity_medium_voltage": (
            electricity_market["database"],
            electricity_market["code"],
        ),
        "natural_gas_high_pressure": (
            natural_gas_market["database"],
            natural_gas_market["code"],
        ),
        "light_fuel_oil": (
            light_fuel_oil_market["database"],
            light_fuel_oil_market["code"],
        ),
        "carbon_dioxide_fossil": _find_biosphere_flow(
            bd, "Carbon dioxide, fossil", ["air", "low population density"]
        ),
        "methane_fossil": _find_biosphere_flow(
            bd, "Methane, fossil", ["air", "non-urban air", "high stacks"]
        ),
        "dinitrogen_monoxide": _find_biosphere_flow(
            bd, "Dinitrogen monoxide", ["air", "lower stratosphere", "upper troposphere"]
        ),
    }

    def upsert_activity(db_name: str, code: str, name: str):
        db = bd.Database(db_name)
        for activity in db:
            if activity["code"] == code:
                act = activity
                break
        else:
            act = db.new_activity(code=code)
        act["name"] = name
        act["unit"] = "kilowatt hour"
        act["location"] = "US"
        act["reference product"] = "thermal energy delivered"
        act.save()
        for exc in list(act.exchanges()):
            exc.delete()
        act.new_exchange(input=act.key, amount=1.0, type="production").save()
        return act

    def add_exchanges(activity, rows: pd.DataFrame) -> None:
        for row in rows.to_dict(orient="records"):
            input_key = mapping[str(row["exchange_key"])]
            activity.new_exchange(
                input=input_key,
                amount=float(row["input_amount_per_kwh_th"]),
                type=str(row["exchange_kind"]),
            ).save()

    gen_rows = exchange_table[
        (exchange_table["case"] == "GEN")
        & (exchange_table["normalization_mode"] == normalization_mode)
    ]
    bau_rows = exchange_table[
        (exchange_table["case"] == "BAU")
        & (exchange_table["normalization_mode"] == normalization_mode)
    ]

    gen_b6 = upsert_activity(
        gen_db_name,
        str(fg_codes["gen_b6_operation"]),
        f"GEN B6 operational energy ({normalization_mode})",
    )
    add_exchanges(gen_b6, gen_rows)

    gen_b6_dynamic_only = upsert_activity(
        gen_db_name,
        str(fg_codes["gen_b6_dynamic_only"]),
        f"GEN B6-only dynamic wrapper ({normalization_mode})",
    )
    gen_b6_dynamic_only.new_exchange(input=gen_b6.key, amount=1.0, type="technosphere").save()

    bau_b6 = upsert_activity(
        bau_db_name,
        str(fg_codes["bau_b6_operation"]),
        f"BAU B6 operational energy ({normalization_mode})",
    )
    add_exchanges(bau_b6, bau_rows)

    bau_b6_dynamic_only = upsert_activity(
        bau_db_name,
        str(fg_codes["bau_b6_dynamic_only"]),
        f"BAU B6-only dynamic wrapper ({normalization_mode})",
    )
    bau_b6_dynamic_only.new_exchange(input=bau_b6.key, amount=1.0, type="technosphere").save()

    gen_total_with_b6 = upsert_activity(
        gen_db_name,
        str(fg_codes["gen_total_with_b6"]),
        f"GEN total with B6 ({normalization_mode})",
    )
    gen_total_with_b6.new_exchange(
        input=(gen_db_name, str(fg_codes["gen_non_b6_total"])),
        amount=1.0,
        type="technosphere",
    ).save()
    gen_total_with_b6.new_exchange(input=gen_b6.key, amount=1.0, type="technosphere").save()

    bau_total_with_b6 = upsert_activity(
        bau_db_name,
        str(fg_codes["bau_total_with_b6"]),
        f"BAU total with B6 ({normalization_mode})",
    )
    bau_total_with_b6.new_exchange(
        input=(bau_db_name, str(fg_codes["bau_non_b6_total"])),
        amount=1.0,
        type="technosphere",
    ).save()
    bau_total_with_b6.new_exchange(input=bau_b6.key, amount=1.0, type="technosphere").save()

    offsets = list(
        range(
            int(config["bw_timex"]["operation_start_year_offset"]),
            int(config["bw_timex"]["operation_end_year_offset"]) + 1,
        )
    )
    td_b6 = TemporalDistribution(
        date=pd.Series(offsets).to_numpy().astype("timedelta64[Y]"),
        amount=(pd.Series(1.0, index=range(len(offsets))) / len(offsets)).to_numpy(),
    )
    add_temporal_distribution_to_exchange(
        td_b6,
        input_database=gen_b6.key[0],
        input_code=gen_b6.key[1],
        output_database=gen_b6_dynamic_only.key[0],
        output_code=gen_b6_dynamic_only.key[1],
    )
    add_temporal_distribution_to_exchange(
        td_b6,
        input_database=gen_b6.key[0],
        input_code=gen_b6.key[1],
        output_database=gen_total_with_b6.key[0],
        output_code=gen_total_with_b6.key[1],
    )
    add_temporal_distribution_to_exchange(
        td_b6,
        input_database=bau_b6.key[0],
        input_code=bau_b6.key[1],
        output_database=bau_b6_dynamic_only.key[0],
        output_code=bau_b6_dynamic_only.key[1],
    )
    add_temporal_distribution_to_exchange(
        td_b6,
        input_database=bau_b6.key[0],
        input_code=bau_b6.key[1],
        output_database=bau_total_with_b6.key[0],
        output_code=bau_total_with_b6.key[1],
    )

    bd.Database(gen_db_name).process()
    bd.Database(bau_db_name).process()

    exports: list[Path] = []
    try:
        from bw2io.export.csv import write_lci_csv
    except ImportError:
        write_lci_csv = None
    if write_lci_csv is not None:
        export_dir = EXPORT_ROOT / "bw_timex_inputs"
        write_lci_csv(
            database_name=gen_db_name,
            objs=[gen_b6, gen_b6_dynamic_only, gen_total_with_b6],
            dirpath=export_dir.as_posix(),
        )
        write_lci_csv(
            database_name=bau_db_name,
            objs=[bau_b6, bau_b6_dynamic_only, bau_total_with_b6],
            dirpath=export_dir.as_posix(),
        )
        exports.extend(
            [
                export_dir / f"lci-{gen_db_name}.csv",
                export_dir / f"lci-{bau_db_name}.csv",
            ]
        )

    manifest = {
        "project": project,
        "normalization_mode": normalization_mode,
        "foreground_set": foreground_set,
        "background_database": bg_db_name,
        "gen_b6_key": gen_b6.key,
        "gen_b6_dynamic_only_key": gen_b6_dynamic_only.key,
        "gen_total_with_b6_key": gen_total_with_b6.key,
        "bau_b6_key": bau_b6.key,
        "bau_b6_dynamic_only_key": bau_b6_dynamic_only.key,
        "bau_total_with_b6_key": bau_total_with_b6.key,
        "mapping": mapping,
    }
    return manifest, exports


def build(
    normalization_mode: str | None = None,
    write_brightway: bool = False,
    foreground_set: str | None = None,
) -> dict[str, Path]:
    ensure_workspace_tree()
    config = load_b6_case_config()
    gen_total, bau_total, common_service = _load_operational_totals()

    default_mode = str(config["bw_timex"]["normalization"]["default_mode"])
    normalization_mode = normalization_mode or default_mode
    supported = set(config["bw_timex"]["normalization"]["supported_modes"])
    if normalization_mode not in supported:
        raise ValueError(
            f"Unsupported normalization mode `{normalization_mode}`. Supported: {sorted(supported)}"
        )
    if foreground_set is None:
        foreground_set = "heatnets_authoritative_sync" if normalization_mode == "heatnets_case_service_loads" else "existing_non_b6_project"
    foreground_sets = set(config["bw_timex"]["foreground_sets"])
    if foreground_set not in foreground_sets:
        raise ValueError(
            f"Unsupported foreground set `{foreground_set}`. Supported: {sorted(foreground_sets)}"
        )

    denominator_summary = _denominator_summary(config, common_service)
    exchange_table = _exchange_rows_for_modes(config, gen_total, bau_total, denominator_summary)

    out_root = EXPORT_ROOT / "bw_timex_inputs"
    outputs = {
        "denominator_summary": out_root / "b6_denominator_summary.csv",
        "exchange_table": out_root / "b6_operational_exchange_table.csv",
        "selected_mode_table": out_root / f"b6_operational_exchange_table_{normalization_mode}_{foreground_set}.csv",
        "manifest": out_root / "b6_inventory_manifest.json",
        "pathway_note": out_root / "b6_bw_timex_design_note.md",
        "config_snapshot": out_root / "b6_case_config_snapshot.yaml",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    denominator_summary.to_csv(outputs["denominator_summary"], index=False)
    exchange_table.to_csv(outputs["exchange_table"], index=False)
    exchange_table.loc[exchange_table["normalization_mode"] == normalization_mode].to_csv(
        outputs["selected_mode_table"], index=False
    )
    dump_yaml(outputs["config_snapshot"], config)

    note = f"""# Integrated B6 Brightway/TimeX Pathway

- This pathway converts the rebuilt hourly GEN and BAU Stage B6 results into lifetime-normalized foreground exchanges.
- Direct combustion and methane leakage remain biosphere exchanges in the BAU B6 activity.
- Electricity, natural gas, and light fuel oil remain technosphere links.
- Selected normalization mode: `{normalization_mode}`.
- Target foreground set: `{foreground_set}`.
- `existing_non_b6_project` preserves immediate compatibility with the current non-B6 foreground databases.
- `heatnets_case_service_loads` is the audited 37-building denominator and is the authoritative mode for the synchronized foreground set.
"""
    write_markdown(outputs["pathway_note"], note)

    manifest: dict[str, Any] = {
        "selected_normalization_mode": normalization_mode,
        "selected_foreground_set": foreground_set,
        "write_brightway": write_brightway,
    }

    if write_brightway:
        bw_manifest, exported_csvs = _write_brightway_foreground(config, exchange_table, normalization_mode, foreground_set)
        manifest.update(bw_manifest)
        for idx, path in enumerate(exported_csvs, start=1):
            outputs[f"brightway_csv_{idx}"] = path

    write_json(outputs["manifest"], manifest)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build integrated Stage B6 Brightway/TimeX inventory inputs and optionally write B6 activities into the existing foreground databases."
    )
    parser.add_argument(
        "--normalization-mode",
        default=None,
        help="Normalization mode to use when selecting the operational B6 per-kWh_th inventory.",
    )
    parser.add_argument(
        "--write-brightway",
        action="store_true",
        help="Write/update B6 foreground activities in the existing Brightway project and export CSV snapshots.",
    )
    parser.add_argument(
        "--foreground-set",
        default=None,
        help="Foreground database set to target: existing_non_b6_project or heatnets_authoritative_sync.",
    )
    args = parser.parse_args()
    outputs = build(
        normalization_mode=args.normalization_mode,
        write_brightway=args.write_brightway,
        foreground_set=args.foreground_set,
    )
    for label, path in outputs.items():
        print(f"{label}: {relpath(path)}")


if __name__ == "__main__":
    main()
