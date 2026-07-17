from __future__ import annotations

import argparse
import csv
import json
import sys
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
OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "manuscript_scope_corrected_2026_06_02"
STATUS_PATH = OUTPUT_ROOT / "tables" / "fresh_brightway_setup_status.csv"
MANIFEST_PATH = OUTPUT_ROOT / "fresh_brightway_setup_manifest.json"

DEFAULT_ECOINVENT_DATASETS = (
    REPO_ROOT / "inputs" / "Resource&Inventory" / "ecoinvent 3.9.1_cutoff_ecoSpold02" / "datasets"
)

REQUIRED_FUTURE_DBS = [
    "ecoinvent_2025_SSP2-PkBudg1000",
    "ecoinvent_2040_SSP2-PkBudg1000",
    "ecoinvent_2045_SSP2-PkBudg1000",
    "ecoinvent_2050_SSP2-PkBudg1000",
    "ecoinvent_2055_SSP2-PkBudg1000",
]


def write_status(rows: list[dict[str, Any]]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "status", "detail"]
    with STATUS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(payload: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_status(rows: list[dict[str, Any]], step: str, status: str, detail: str) -> None:
    rows.append({"step": step, "status": status, "detail": detail})
    write_status(rows)
    print(f"[{status}] {step}: {detail}", flush=True)


def ensure_biosphere_and_methods(bd, bi, rows: list[dict[str, Any]]) -> None:
    if "biosphere3" in bd.databases:
        append_status(rows, "biosphere3", "present", "biosphere3 already exists.")
        return
    append_status(rows, "biosphere3", "running", "Creating default biosphere3 and LCIA methods with bw2setup().")
    bi.bw2setup()
    append_status(
        rows,
        "biosphere3",
        "complete",
        f"Databases now include biosphere3={('biosphere3' in bd.databases)}; method count={len(list(bd.methods))}.",
    )


def import_local_ecoinvent(
    bd,
    bi,
    rows: list[dict[str, Any]],
    datasets: Path,
    db_name: str,
    use_mp: bool,
) -> None:
    if db_name in bd.databases:
        append_status(rows, db_name, "present", "Source ecoinvent database already exists.")
        return
    if not datasets.exists():
        raise FileNotFoundError(f"Missing local ecoinvent datasets folder: {datasets}")
    spold_count = len(list(datasets.glob("*.spold")))
    append_status(
        rows,
        db_name,
        "running",
        f"Importing {spold_count} local ecoSpold files from {datasets} with biosphere3 linking.",
    )
    importer = bi.SingleOutputEcospold2Importer(
        dirpath=str(datasets),
        db_name=db_name,
        biosphere_database_name="biosphere3",
        use_mp=use_mp,
    )
    importer.apply_strategies()
    stats = importer.statistics()
    importer.write_database()
    bd.Database(db_name).process()
    append_status(rows, db_name, "complete", f"Imported and processed {db_name}; importer statistics={stats!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the local Brightway project required for the GEN/BAU LCIA rerun.")
    parser.add_argument("--project", default="GEN_DLCA_391")
    parser.add_argument("--source-db", default="ecoinvent-3.9.1-cutoff")
    parser.add_argument("--datasets", type=Path, default=DEFAULT_ECOINVENT_DATASETS)
    parser.add_argument("--use-mp", action="store_true", help="Use multiprocessing during local ecoSpold import.")
    parser.add_argument("--skip-premise", action="store_true", help="Only import biosphere/source DB; do not run premise.")
    args = parser.parse_args()

    try:
        import bw2data as bd
        import bw2io as bi
    except ImportError as exc:
        raise RuntimeError("This script must be run in the Brightway environment.") from exc

    rows: list[dict[str, Any]] = []
    bd.projects.set_current(args.project)
    append_status(rows, "project", "selected", f"Current Brightway project is {bd.projects.current}.")
    ensure_biosphere_and_methods(bd, bi, rows)
    import_local_ecoinvent(
        bd=bd,
        bi=bi,
        rows=rows,
        datasets=args.datasets,
        db_name=args.source_db,
        use_mp=args.use_mp,
    )

    if args.skip_premise:
        append_status(rows, "premise_backgrounds", "skipped", "Premise generation was skipped by CLI flag.")
    else:
        missing = [name for name in REQUIRED_FUTURE_DBS if name not in bd.databases]
        if not missing:
            append_status(rows, "premise_backgrounds", "present", "All required future background databases exist.")
        else:
            append_status(rows, "premise_backgrounds", "blocked", f"Missing future DBs: {missing}. Run scripts/01_ensure_premise_backgrounds.py after confirming premise data access.")

    write_manifest(
        {
            "project": args.project,
            "source_db": args.source_db,
            "datasets": str(args.datasets),
            "python": sys.version,
            "databases": sorted(str(name) for name in bd.databases),
            "method_count": len(list(bd.methods)),
            "required_future_databases": REQUIRED_FUTURE_DBS,
            "missing_future_databases": [name for name in REQUIRED_FUTURE_DBS if name not in bd.databases],
            "status_csv": str(STATUS_PATH),
        }
    )


if __name__ == "__main__":
    main()
