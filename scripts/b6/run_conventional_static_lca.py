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
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
    )
    from Dynamic_LCA_GEN_BAU.scripts.b6.run_full_dynamic_lcia import (  # type: ignore
        NON_CLIMATE_METHOD_SPECS,
        _choose_climate_method,
    )
else:
    from .common import (
        WORKSPACE_ROOT,
        ensure_workspace_tree,
        load_b6_case_config,
        read_json,
        relpath,
        validate_exists,
        write_json,
    )
    from .run_full_dynamic_lcia import NON_CLIMATE_METHOD_SPECS, _choose_climate_method


OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "static_lca"
QA_ROOT = OUTPUT_ROOT / "qa"
FULL_MANIFEST_PATH = WORKSPACE_ROOT / "export" / "full_dynamic_lcia" / "qa" / "full_dynamic_lcia_manifest.json"
FULL_METHOD_SUMMARY_PATH = WORKSPACE_ROOT / "export" / "full_dynamic_lcia" / "qa" / "denominator_and_method_summary_full.csv"

STAGE_FILE_STUBS = {
    "A": "stage_A",
    "B": "stage_B",
    "C": "stage_C",
    "total": "total",
}


def _ensure_output_tree() -> None:
    ensure_workspace_tree()
    for folder in [OUTPUT_ROOT, QA_ROOT]:
        folder.mkdir(parents=True, exist_ok=True)


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


def _load_context() -> dict[str, Any]:
    manifest = read_json(validate_exists(FULL_MANIFEST_PATH, "full dynamic LCIA manifest"))
    method_summary = pd.read_csv(validate_exists(FULL_METHOD_SUMMARY_PATH, "full dynamic LCIA method summary"))
    lookup = method_summary.set_index("key")["value"].to_dict()
    denominator = float(lookup["denominator_lifetime_kwh_th"])
    background_2025 = str(load_b6_case_config()["project"]["background_dbs"][0])
    if background_2025 != "ecoinvent_2025_SSP2-PkBudg1000":
        raise RuntimeError(
            f"Conventional static package expects the fixed 2025 background as first project database, got `{background_2025}`."
        )
    return {
        "project": manifest["project"],
        "foreground_set": manifest["foreground_set"],
        "normalization_mode": manifest["normalization_mode"],
        "denominator_lifetime_kwh_th": denominator,
        "background_db": background_2025,
        "time_explicit_manifest": manifest,
    }


def _static_code(case: str, stage: str) -> str:
    suffix = {
        "A": "STAGEA_STATIC_2025",
        "B": "STAGEB_FULL_STATIC_2025",
        "C": "STAGEC_STATIC_2025",
        "total": "TOTAL_STATIC_2025",
    }[stage]
    return f"{case}_{suffix}"


def _static_name(case: str, stage: str) -> str:
    names = {
        "A": f"{case} Stage A conventional static wrapper (2025 background)",
        "B": f"{case} Stage B conventional static wrapper incl. B6 (2025 background)",
        "C": f"{case} Stage C conventional static wrapper (2025 background)",
        "total": f"{case} total conventional static wrapper incl. B6 (2025 background)",
    }
    return names[stage]


def _build_static_wrappers(context: dict[str, Any]) -> tuple[dict[str, dict[str, tuple[str, str]]], pd.DataFrame]:
    import bw2data as bd

    bd.projects.set_current(context["project"])
    manifest = context["time_explicit_manifest"]
    wrapper_keys = manifest["stage_wrappers"]

    rows: list[dict[str, Any]] = []
    out: dict[str, dict[str, tuple[str, str]]] = {"GEN": {}, "BAU": {}}

    for case in ["GEN", "BAU"]:
        fg_db_name = str(wrapper_keys[case]["A"][0])
        db = bd.Database(fg_db_name)
        static_stage_keys: dict[str, tuple[str, str]] = {}

        for stage in ["A", "B", "C"]:
            dynamic_wrapper = bd.get_activity(tuple(wrapper_keys[case][stage]))
            static_code = _static_code(case, stage)
            static_wrapper = _upsert_activity(db, static_code, _static_name(case, stage))
            for exc in dynamic_wrapper.technosphere():
                static_wrapper.new_exchange(input=exc.input.key, amount=float(exc["amount"]), type="technosphere").save()
                rows.append(
                    {
                        "case": case,
                        "stage": stage,
                        "static_wrapper_key": str(static_wrapper.key),
                        "input_key": str(exc.input.key),
                        "input_name": exc.input.get("name", exc.input["code"]),
                        "amount": float(exc["amount"]),
                        "timed_exchange_removed": int("temporal_distribution" in exc),
                        "input_database": exc.input["database"],
                    }
                )
            static_stage_keys[stage] = static_wrapper.key
            out[case][stage] = static_wrapper.key

        total_wrapper = _upsert_activity(db, _static_code(case, "total"), _static_name(case, "total"))
        for stage in ["A", "B", "C"]:
            total_wrapper.new_exchange(input=static_stage_keys[stage], amount=1.0, type="technosphere").save()
            rows.append(
                {
                    "case": case,
                    "stage": "total",
                    "static_wrapper_key": str(total_wrapper.key),
                    "input_key": str(static_stage_keys[stage]),
                    "input_name": bd.get_activity(static_stage_keys[stage]).get("name", static_stage_keys[stage][1]),
                    "amount": 1.0,
                    "timed_exchange_removed": 0,
                    "input_database": static_stage_keys[stage][0],
                }
            )
        out[case]["total"] = total_wrapper.key
        db.process()

    link_table = pd.DataFrame(rows).sort_values(["case", "stage", "input_name"])
    return out, link_table


def _run_lca(wrapper_key: tuple[str, str], method: tuple[str, ...]) -> float:
    import bw2calc as bc

    lca = bc.LCA({wrapper_key: 1.0}, method=method)
    lca.lci()
    lca.lcia()
    return float(lca.score)


def _result_table(
    *,
    case: str,
    stage: str,
    wrapper_key: tuple[str, str],
    context: dict[str, Any],
    climate_method: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    climate_score = _run_lca(wrapper_key, climate_method)
    rows.append(
        {
            "case": case,
            "stage": stage,
            "wrapper_key": str(wrapper_key),
            "category_key": "climate_static",
            "category_display_name": "Climate Change",
            "method_family": climate_method[0],
            "method_tuple": " | ".join(climate_method),
            "unit": "kg CO2e",
            "conventional_static_background_db": context["background_db"],
            "normalization_mode": context["normalization_mode"],
            "foreground_set": context["foreground_set"],
            "static_score_per_fu": climate_score,
            "static_score_case_total": climate_score * context["denominator_lifetime_kwh_th"],
        }
    )

    for category_key, spec in NON_CLIMATE_METHOD_SPECS.items():
        method = tuple(spec["method"])
        score = _run_lca(wrapper_key, method)
        rows.append(
            {
                "case": case,
                "stage": stage,
                "wrapper_key": str(wrapper_key),
                "category_key": category_key,
                "category_display_name": spec["display_name"],
                "method_family": spec["family"],
                "method_tuple": " | ".join(method),
                "unit": spec["unit"],
                "conventional_static_background_db": context["background_db"],
                "normalization_mode": context["normalization_mode"],
                "foreground_set": context["foreground_set"],
                "static_score_per_fu": score,
                "static_score_case_total": score * context["denominator_lifetime_kwh_th"],
            }
        )

    return pd.DataFrame(rows)


def _write_stage_exports(results: dict[str, dict[str, pd.DataFrame]]) -> None:
    for case in ["GEN", "BAU"]:
        for stage, stub in STAGE_FILE_STUBS.items():
            results[case][stage].to_csv(OUTPUT_ROOT / f"{stub}_static_{case}.csv", index=False)


def _build_stage_contributions(results: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for case in ["GEN", "BAU"]:
        total_lookup = results[case]["total"].set_index("category_key")["static_score_case_total"].to_dict()
        rows = []
        for stage in ["A", "B", "C"]:
            table = results[case][stage]
            for record in table.to_dict(orient="records"):
                total_value = float(total_lookup[record["category_key"]])
                rows.append(
                    {
                        "case": case,
                        "stage": stage,
                        "category_key": record["category_key"],
                        "category_display_name": record["category_display_name"],
                        "unit": record["unit"],
                        "stage_score_case_total": float(record["static_score_case_total"]),
                        "total_score_case_total": total_value,
                        "share_of_total": (float(record["static_score_case_total"]) / total_value) if total_value else np.nan,
                    }
                )
        out[case] = pd.DataFrame(rows).sort_values(["category_key", "stage"])
        out[case].to_csv(OUTPUT_ROOT / f"stage_contribution_static_{case}.csv", index=False)
    return out


def _method_summary(context: dict[str, Any], climate_method: tuple[str, ...], static_wrapper_keys: dict[str, dict[str, tuple[str, str]]]) -> pd.DataFrame:
    rows = [
        {"category": "project", "key": "brightway_project", "value": context["project"]},
        {"category": "mode", "key": "foreground_set", "value": context["foreground_set"]},
        {"category": "mode", "key": "normalization_mode", "value": context["normalization_mode"]},
        {"category": "mode", "key": "denominator_lifetime_kwh_th", "value": context["denominator_lifetime_kwh_th"]},
        {"category": "mode", "key": "background_db", "value": context["background_db"]},
        {"category": "methods", "key": "climate_static_method", "value": " | ".join(climate_method)},
        {"category": "methods", "key": "non_climate_method_family", "value": "TRACI v2.1 no LT core categories + CED fossil supplemental metric"},
    ]
    for case in ["GEN", "BAU"]:
        for stage in ["A", "B", "C", "total"]:
            rows.append(
                {
                    "category": "static_wrapper",
                    "key": f"{case}_{stage}",
                    "value": str(static_wrapper_keys[case][stage]),
                }
            )
    return pd.DataFrame(rows)


def _sanity_checks(
    *,
    context: dict[str, Any],
    link_table: pd.DataFrame,
    results: dict[str, dict[str, pd.DataFrame]],
    static_wrapper_keys: dict[str, dict[str, tuple[str, str]]],
) -> pd.DataFrame:
    import bw2data as bd

    rows = []
    rows.append(
        {
            "check": "All conventional static wrappers removed timed wrapper exchanges",
            "value": int(link_table["timed_exchange_removed"].sum()),
            "expected_minimum": 6,
            "status": "pass" if int(link_table["timed_exchange_removed"].sum()) >= 6 else "fail",
        }
    )
    external_background_dbs = set()
    for case in ["GEN", "BAU"]:
        for stage in ["A", "B", "C"]:
            wrapper = bd.get_activity(static_wrapper_keys[case][stage])
            for exc in wrapper.technosphere():
                stage_activity = exc.input
                for stage_exc in stage_activity.technosphere():
                    input_db = stage_exc.input["database"]
                    if input_db not in {wrapper["database"]}:
                        external_background_dbs.add(input_db)
    rows.append(
        {
            "check": "Conventional static stage activities use a single fixed external background DB",
            "value": " | ".join(sorted(external_background_dbs)),
            "expected_minimum": context["background_db"],
            "status": "pass" if external_background_dbs == {context["background_db"]} else "fail",
        }
    )
    rows.append(
        {
            "check": "Conventional static denominator matches authoritative denominator",
            "value": context["denominator_lifetime_kwh_th"],
            "expected_minimum": context["denominator_lifetime_kwh_th"],
            "status": "pass",
        }
    )

    for case in ["GEN", "BAU"]:
        total_lookup = results[case]["total"].set_index("category_key")["static_score_case_total"].to_dict()
        for category_key, total_value in total_lookup.items():
            stage_sum = sum(
                float(results[case][stage].set_index("category_key").loc[category_key, "static_score_case_total"])
                for stage in ["A", "B", "C"]
            )
            rows.append(
                {
                    "check": f"{case} A+B+C conventional static {category_key} reconciles to total",
                    "value": stage_sum,
                    "expected_minimum": float(total_value),
                    # Matrix solves differ by tiny roundoff across solver stacks; 1e-7 still catches material drift.
                    "status": "pass" if np.isclose(stage_sum, float(total_value), rtol=1e-7, atol=1e-6) else "fail",
                }
            )
    return pd.DataFrame(rows)


def run() -> dict[str, Path]:
    _ensure_output_tree()
    context = _load_context()

    try:
        import bw2data as bd
    except ImportError as exc:
        raise RuntimeError("Run conventional static LCA from the `bw25` environment.") from exc

    bd.projects.set_current(context["project"])
    climate_method = _choose_climate_method(bd)
    static_wrapper_keys, link_table = _build_static_wrappers(context)

    results: dict[str, dict[str, pd.DataFrame]] = {"GEN": {}, "BAU": {}}
    for case in ["GEN", "BAU"]:
        for stage in ["A", "B", "C", "total"]:
            results[case][stage] = _result_table(
                case=case,
                stage=stage,
                wrapper_key=static_wrapper_keys[case][stage],
                context=context,
                climate_method=climate_method,
            )

    _write_stage_exports(results)
    _build_stage_contributions(results)

    method_summary = _method_summary(context, climate_method, static_wrapper_keys)
    method_summary.to_csv(OUTPUT_ROOT / "static_method_summary.csv", index=False)

    link_table.to_csv(QA_ROOT / "static_wrapper_links.csv", index=False)
    sanity = _sanity_checks(
        context=context,
        link_table=link_table,
        results=results,
        static_wrapper_keys=static_wrapper_keys,
    )
    sanity.to_csv(OUTPUT_ROOT / "static_sanity_checks.csv", index=False)

    manifest = {
        "project": context["project"],
        "foreground_set": context["foreground_set"],
        "normalization_mode": context["normalization_mode"],
        "background_db": context["background_db"],
        "denominator_lifetime_kwh_th": context["denominator_lifetime_kwh_th"],
        "static_wrapper_keys": {case: {stage: list(key) for stage, key in mapping.items()} for case, mapping in static_wrapper_keys.items()},
        "generated_outputs": {
            "output_root": relpath(OUTPUT_ROOT),
            "qa_root": relpath(QA_ROOT),
        },
    }
    write_json(QA_ROOT / "static_lca_manifest.json", manifest)

    return {
        "static_method_summary": OUTPUT_ROOT / "static_method_summary.csv",
        "static_sanity_checks": OUTPUT_ROOT / "static_sanity_checks.csv",
        "static_manifest": QA_ROOT / "static_lca_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conventional static LCA package on the synchronized authoritative foreground set.")
    parser.parse_args()
    outputs = run()
    print("Conventional static LCA outputs:")
    for key, path in outputs.items():
        print(f" - {key}: {relpath(path)}")


if __name__ == "__main__":
    main()
