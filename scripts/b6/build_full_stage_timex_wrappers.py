from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Dynamic_LCA_GEN_BAU.scripts.b6.common import (  # type: ignore
        DOCS_ROOT,
        EXPORT_ROOT,
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
    )
else:
    from .common import (
        DOCS_ROOT,
        EXPORT_ROOT,
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
    )


OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "full_dynamic_lcia"
QA_ROOT = OUTPUT_ROOT / "qa"
B6_MANIFEST_PATH = EXPORT_ROOT / "bw_timex_inputs" / "b6_inventory_manifest.json"
QA_REPORT = DOCS_ROOT / "b6_qa_report.md"


def _ensure_output_tree() -> None:
    ensure_workspace_tree()
    for folder in [
        OUTPUT_ROOT,
        OUTPUT_ROOT / "stage_A",
        OUTPUT_ROOT / "stage_B",
        OUTPUT_ROOT / "stage_C",
        OUTPUT_ROOT / "total",
        OUTPUT_ROOT / "figures",
        OUTPUT_ROOT / "summaries",
        QA_ROOT,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def _require_authoritative_mode() -> dict[str, Any]:
    manifest = read_json(B6_MANIFEST_PATH)
    normalization_mode = str(
        manifest.get("selected_normalization_mode", manifest.get("normalization_mode", ""))
    )
    foreground_set = str(manifest.get("selected_foreground_set", manifest.get("foreground_set", "")))
    if normalization_mode != "heatnets_case_service_loads":
        raise RuntimeError(
            "Full synchronized LCIA requires the authoritative HEATNETS denominator. "
            f"Manifest selected `{normalization_mode}` instead."
        )
    if foreground_set != "heatnets_authoritative_sync":
        raise RuntimeError(
            "Full synchronized LCIA requires the synchronized authoritative foreground set. "
            f"Manifest selected `{foreground_set}` instead."
        )
    return manifest


def _find_activity(db, code: str):
    for activity in db:
        if activity["code"] == code:
            return activity
    return None


def _upsert_activity(db, code: str, name: str):
    activity = _find_activity(db, code)
    if activity is None:
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


def _classify_stage(activity) -> str | None:
    text = " | ".join([activity.get("name", ""), activity.get("code", "")]).upper()
    if "A1-A5" in text or "_A1A5_" in text:
        return "A"
    if "B2-B4" in text or "_B2B4_" in text or "_B_" in text:
        return "B"
    if "C1-C4" in text or "_C1C4_" in text or "_C" in text:
        return "C"
    return None


def _temporal_summary(td: Any) -> dict[str, Any]:
    raw_dates = np.array(td.date)
    raw_amount = np.array(td.amount, dtype=float)
    return {
        "td_point_count": int(len(raw_dates)),
        "td_amount_sum": float(raw_amount.sum()),
        "td_date_dtype": str(raw_dates.dtype),
        "td_first_date": str(raw_dates[0]) if len(raw_dates) else "",
        "td_last_date": str(raw_dates[-1]) if len(raw_dates) else "",
    }


def _extract_stage_links(total_activity, stage_code: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for exc in total_activity.technosphere():
        input_act = exc.input
        if input_act["database"] != total_activity["database"]:
            continue
        if _classify_stage(input_act) != stage_code:
            continue
        td = exc.get("temporal_distribution")
        if td is None:
            raise RuntimeError(
                f"Missing temporal_distribution on {total_activity.key} -> {input_act.key}. "
                "Stage wrappers must preserve the audited timing."
            )
        row = {
            "source_total_key": str(total_activity.key),
            "input_key": input_act.key,
            "input_name": input_act.get("name", input_act["code"]),
            "amount": float(exc["amount"]),
            "temporal_distribution": td,
            "included_stage": stage_code,
        }
        row.update(_temporal_summary(td))
        links.append(row)
    if not links:
        raise RuntimeError(
            f"No stage `{stage_code}` links were found in {total_activity.key}. "
            "The synchronized total activities do not match the expected stage structure."
        )
    return links


def _extract_b6_link(total_with_b6, expected_code: str) -> dict[str, Any]:
    for exc in total_with_b6.technosphere():
        input_act = exc.input
        if input_act["database"] != total_with_b6["database"]:
            continue
        if input_act["code"] != expected_code:
            continue
        td = exc.get("temporal_distribution")
        if td is None:
            raise RuntimeError(
                f"Missing temporal_distribution on {total_with_b6.key} -> {input_act.key}. "
                "The corrected B6 link must stay explicitly timed."
            )
        row = {
            "source_total_key": str(total_with_b6.key),
            "input_key": input_act.key,
            "input_name": input_act.get("name", input_act["code"]),
            "amount": float(exc["amount"]),
            "temporal_distribution": td,
            "included_stage": "B",
        }
        row.update(_temporal_summary(td))
        return row
    raise RuntimeError(
        f"Could not find B6 link `{expected_code}` inside {total_with_b6.key}. "
        "Rebuild the authoritative B6 inventory first."
    )


def _add_internal_links(activity, links: list[dict[str, Any]]) -> None:
    from bw_timex.utils import add_temporal_distribution_to_exchange

    for link in links:
        input_key = tuple(link["input_key"])
        activity.new_exchange(input=input_key, amount=float(link["amount"]), type="technosphere").save()
        add_temporal_distribution_to_exchange(
            temporal_distribution=link["temporal_distribution"],
            input_database=input_key[0],
            input_code=input_key[1],
            output_database=activity.key[0],
            output_code=activity.key[1],
        )


def _wrapper_name(system: str, stage: str) -> str:
    names = {
        "A": f"{system} Stage A dynamic wrapper (authoritative sync)",
        "B": f"{system} Stage B full dynamic wrapper incl. B6 (authoritative sync)",
        "C": f"{system} Stage C dynamic wrapper (authoritative sync)",
    }
    return names[stage]


def build(export_csv: bool = True) -> dict[str, Any]:
    _ensure_output_tree()
    config = load_b6_case_config()
    b6_manifest = _require_authoritative_mode()

    try:
        import bw2data as bd
        from bw2io.export.csv import write_lci_csv
    except ImportError as exc:
        raise RuntimeError(
            "Brightway wrapper writing requires the `bw25` environment with bw2data and bw2io."
        ) from exc

    bd.projects.set_current(str(config["project"]["brightway_project"]))
    codes = config["bw_timex"]["foreground_activity_codes"]
    fg_set = config["bw_timex"]["foreground_sets"]["heatnets_authoritative_sync"]

    systems = {
        "GEN": {
            "fg_db": str(fg_set["gen_fg_db"]),
            "total_non_b6": str(codes["gen_non_b6_total"]),
            "total_with_b6": str(codes["gen_total_with_b6"]),
            "b6_operation": str(codes["gen_b6_operation"]),
            "stage_codes": {
                "A": str(codes["gen_stage_a_dynamic"]),
                "B": str(codes["gen_stage_b_full_dynamic"]),
                "C": str(codes["gen_stage_c_dynamic"]),
            },
        },
        "BAU": {
            "fg_db": str(fg_set["bau_fg_db"]),
            "total_non_b6": str(codes["bau_non_b6_total"]),
            "total_with_b6": str(codes["bau_total_with_b6"]),
            "b6_operation": str(codes["bau_b6_operation"]),
            "stage_codes": {
                "A": str(codes["bau_stage_a_dynamic"]),
                "B": str(codes["bau_stage_b_full_dynamic"]),
                "C": str(codes["bau_stage_c_dynamic"]),
            },
        },
    }

    manifest: dict[str, Any] = {
        "project": config["project"]["brightway_project"],
        "normalization_mode": b6_manifest["selected_normalization_mode"],
        "foreground_set": b6_manifest["selected_foreground_set"],
        "stage_wrapper_keys": {},
        "source_manifest": relpath(B6_MANIFEST_PATH),
    }
    scope_rows: list[dict[str, Any]] = []
    export_objs = []

    for system, payload in systems.items():
        db = bd.Database(payload["fg_db"])
        total_non_b6 = bd.get_activity((payload["fg_db"], payload["total_non_b6"]))
        total_with_b6 = bd.get_activity((payload["fg_db"], payload["total_with_b6"]))

        stage_links = {
            "A": _extract_stage_links(total_non_b6, "A"),
            "B": _extract_stage_links(total_non_b6, "B")
            + [_extract_b6_link(total_with_b6, payload["b6_operation"])],
            "C": _extract_stage_links(total_non_b6, "C"),
        }

        manifest["stage_wrapper_keys"][system] = {}
        for stage_code, wrapper_code in payload["stage_codes"].items():
            wrapper = _upsert_activity(db, wrapper_code, _wrapper_name(system, stage_code))
            _add_internal_links(wrapper, stage_links[stage_code])
            export_objs.append(wrapper)
            manifest["stage_wrapper_keys"][system][stage_code] = wrapper.key

            for link in stage_links[stage_code]:
                scope_rows.append(
                    {
                        "system": system,
                        "stage_block": stage_code,
                        "wrapper_key": str(wrapper.key),
                        "wrapper_code": wrapper_code,
                        "source_total_key": link["source_total_key"],
                        "input_key": str(tuple(link["input_key"])),
                        "input_name": link["input_name"],
                        "amount": link["amount"],
                        "td_point_count": link["td_point_count"],
                        "td_amount_sum": link["td_amount_sum"],
                        "td_date_dtype": link["td_date_dtype"],
                        "td_first_date": link["td_first_date"],
                        "td_last_date": link["td_last_date"],
                    }
                )

        db.process()

    exports: list[str] = []
    if export_csv:
        for system, payload in systems.items():
            db_name = payload["fg_db"]
            objs = [obj for obj in export_objs if obj["database"] == db_name]
            if objs:
                write_lci_csv(
                    database_name=db_name,
                    objs=objs,
                    dirpath=QA_ROOT.as_posix(),
                )
                exports.append(relpath(QA_ROOT / f"lci-{db_name}.csv"))

    scope_path = QA_ROOT / "stage_scope_check.csv"
    pd.DataFrame(scope_rows).to_csv(scope_path, index=False)
    manifest["stage_scope_check"] = relpath(scope_path)
    manifest["generated_lci_csvs"] = exports

    manifest_path = QA_ROOT / "stage_wrapper_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "stage_scope_check": scope_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build synchronized Stage A/B/C dynamic wrapper activities from the authoritative HEATNETS-aligned foregrounds."
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Skip CSV snapshot export of the new wrapper activities.",
    )
    args = parser.parse_args()

    outputs = build(export_csv=not args.no_export_csv)
    print("Built synchronized stage wrappers:")
    for key, path in outputs.items():
        print(f" - {key}: {relpath(path)}")


if __name__ == "__main__":
    main()
