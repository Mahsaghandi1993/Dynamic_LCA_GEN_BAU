import argparse
import sys
import importlib.metadata as md
import os
import shutil
from pathlib import Path

import bw2data as bd
import premise
from premise import NewDatabase
from premise.filesystem_constants import IAM_OUTPUT_DIR

def db_name(year: int, pathway: str) -> str:
    return f"ecoinvent_{year}_{pathway}"

def default_iam_file(model: str, pathway: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "inputs"
        / "Resource&Inventory"
        / "iam_scenarios"
        / f"{model}_{pathway}.csv"
    )

def looks_encrypted(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return handle.read(8).startswith(b"gAAAA")

def stage_iam_file(model: str, pathway: str, source: Path | None) -> Path | None:
    if source is None or not source.exists():
        return None
    target = Path(IAM_OUTPUT_DIR) / f"{model}_{pathway}{source.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or source.stat().st_size != target.stat().st_size:
        shutil.copy2(source, target)
    return target

def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--project", default="GEN_DLCA_391")
    p.add_argument("--source-db", default="ecoinvent-3.9.1-cutoff")
    p.add_argument("--source-version", default="3.9.1")
    p.add_argument("--model", default="remind")
    p.add_argument("--pathway", default="SSP2-PkBudg1000")
    p.add_argument("--years", nargs="+", type=int, default=[2025, 2040, 2045, 2050, 2055])
    p.add_argument("--iam-file", type=Path, default=None)
    p.add_argument("--key", default=None, help="premise IAM decryption key. Prefer PREMIS/IAM env vars for shell history safety.")
    p.add_argument("--key-env", default="PREMISE_IAM_KEY")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args, unknown = p.parse_known_args(argv)

    if unknown:
        print("Ignoring unknown args:", unknown)

    bd.projects.set_current(args.project)

    if args.source_db not in bd.databases:
        raise SystemExit(f"Missing source DB in project: {args.source_db}")

    desired = [db_name(y, args.pathway) for y in args.years]
    existing = [name for name in desired if name in bd.databases]
    missing = [name for name in desired if name not in bd.databases]
    iam_file = args.iam_file or default_iam_file(args.model, args.pathway)
    staged_iam = stage_iam_file(args.model, args.pathway, iam_file)
    key = args.key or os.environ.get(args.key_env) or os.environ.get("IAM_DECRYPTION_KEY")

    print("Project:", bd.projects.current)
    print("Python:", sys.version.split()[0])
    print("bw2data:", md.version("bw2data"))
    print("premise:", premise.__version__)
    print("Source DB:", args.source_db, "(version:", args.source_version + ")")
    print("Desired DBs:", desired)
    print("Existing DBs:", existing)
    print("Missing DBs:", missing)
    print("IAM file:", staged_iam or "not found")
    print("IAM file encrypted:", bool(staged_iam and looks_encrypted(staged_iam)))
    print("IAM key provided:", bool(key))

    if not missing:
        print("Nothing to do. Skipping.")
        return

    if args.dry_run:
        print("Dry run only. Exiting.")
        return

    missing_years = [int(name.split("_")[1]) for name in missing]
    scenarios = [{"model": args.model, "pathway": args.pathway, "year": y} for y in missing_years]

    if staged_iam and looks_encrypted(staged_iam) and not key:
        raise SystemExit(
            f"IAM file `{staged_iam}` is encrypted. Provide the premise decryption key via "
            f"`{args.key_env}` or `--key` before generating future databases."
        )

    ndb = NewDatabase(
        scenarios=scenarios,
        source_db=args.source_db,
        source_version=args.source_version,
        key=key,
    )
    ndb.update()
    ndb.write_db_to_brightway(name=missing)

    print("Done. Wrote:", missing)

if __name__ == "__main__":
    main()
