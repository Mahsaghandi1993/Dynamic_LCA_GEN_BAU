from __future__ import annotations

import csv
import json
import pickle
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT
INPUTS_ROOT = WORKSPACE_ROOT / "inputs"
RAW_ROOT = INPUTS_ROOT / "raw"
CONFIG_ROOT = INPUTS_ROOT / "config"
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts" / "b6"
NOTEBOOKS_ROOT = WORKSPACE_ROOT / "notebooks"
EXPORT_ROOT = WORKSPACE_ROOT / "export" / "b6"
DOCS_ROOT = WORKSPACE_ROOT / "docs"


@dataclass(frozen=True)
class WorkspacePaths:
    repo_root: Path = REPO_ROOT
    workspace_root: Path = WORKSPACE_ROOT
    inputs_root: Path = INPUTS_ROOT
    raw_root: Path = RAW_ROOT
    config_root: Path = CONFIG_ROOT
    scripts_root: Path = SCRIPTS_ROOT
    notebooks_root: Path = NOTEBOOKS_ROOT
    export_root: Path = EXPORT_ROOT
    docs_root: Path = DOCS_ROOT


def workspace_paths() -> WorkspacePaths:
    return WorkspacePaths()


def ensure_workspace_tree() -> WorkspacePaths:
    paths = workspace_paths()
    required = [
        paths.raw_root,
        paths.raw_root / "cambium",
        paths.config_root,
        paths.scripts_root,
        paths.notebooks_root,
        paths.export_root / "audit",
        paths.export_root / "gen",
        paths.export_root / "bau",
        paths.export_root / "comparison",
        paths.export_root / "cambium_ghg",
        paths.export_root / "bw_timex_inputs",
        paths.docs_root,
    ]
    for folder in required:
        folder.mkdir(parents=True, exist_ok=True)
    return paths


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {relpath(path)}. "
            f"Run the organization step first or update the config paths."
        )
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    validate_exists(path, "YAML config")
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(payload, handle, sort_keys=False)
        else:
            json.dump(payload, handle, indent=2)


def read_json(path: Path) -> Any:
    validate_exists(path, "JSON file")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copytree_preserve(src: Path, dst: Path) -> None:
    validate_exists(src, "source directory")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file_preserve(src: Path, dst: Path) -> None:
    validate_exists(src, "source file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_subprocess(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _col_to_idx(col: str) -> int:
    value = 0
    for char in col:
        if char.isalpha():
            value = value * 26 + (ord(char.upper()) - 64)
    return value - 1


def _xlsx_metadata(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    with ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for node in root.findall("a:si", XLSX_NS):
                shared.append("".join(t.text or "" for t in node.iterfind(".//a:t", XLSX_NS)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("rel:Relationship", XLSX_NS)
        }
        sheets = []
        for sheet in workbook.find("a:sheets", XLSX_NS):
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            sheets.append((sheet.attrib["name"], "xl/" + rel_map[rel_id]))
    return shared, sheets


def _xlsx_rows(path: Path, sheet_name: str | int = 0) -> list[list[str]]:
    shared, sheets = _xlsx_metadata(path)
    if isinstance(sheet_name, int):
        _, target = sheets[sheet_name]
    else:
        target = dict(sheets)[sheet_name]

    rows: list[list[str]] = []
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read(target))
        for row in root.iterfind(".//a:sheetData/a:row", XLSX_NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                column = "".join(ch for ch in ref if ch.isalpha())
                idx = _col_to_idx(column)
                cell_type = cell.attrib.get("t")
                node = cell.find("a:v", XLSX_NS)
                if node is None:
                    value = ""
                else:
                    value = node.text or ""
                    if cell_type == "s":
                        value = shared[int(value)]
                values[idx] = value
            if values:
                width = max(values) + 1
                row_values = [""] * width
                for idx, value in values.items():
                    row_values[idx] = value
                rows.append(row_values)
    return rows


def read_excel_any(path: Path, sheet_name: str | int = 0, skiprows: int = 0) -> pd.DataFrame:
    validate_exists(path, "Excel workbook")
    try:
        return pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows)
    except Exception:
        rows = _xlsx_rows(path, sheet_name=sheet_name)
        trimmed = rows[skiprows:]
        if not trimmed:
            return pd.DataFrame()
        header = trimmed[0]
        width = max(len(header), *(len(row) for row in trimmed[1:]))
        header = header + [f"unnamed_{i}" for i in range(len(header), width)]
        body = [row + [""] * (width - len(row)) for row in trimmed[1:]]
        frame = pd.DataFrame(body, columns=header)
        for col in frame.columns:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except Exception:
                continue
        return frame


def load_b6_case_config() -> dict[str, Any]:
    return load_yaml(CONFIG_ROOT / "b6_case_config.yaml")


def load_bau_baseline_config() -> dict[str, Any]:
    return load_yaml(CONFIG_ROOT / "bau_baseline_config.yaml")


def load_fuel_share_defaults() -> dict[str, Any]:
    return load_yaml(CONFIG_ROOT / "fuel_share_framingham_acs.yaml")


def load_group_map() -> pd.DataFrame:
    return pd.read_csv(validate_exists(CONFIG_ROOT / "building_group_map.csv", "building group map"))


def load_archetype_map() -> pd.DataFrame:
    return pd.read_csv(validate_exists(CONFIG_ROOT / "archetype_mapping.csv", "archetype mapping"))


def heatnets_source_root(raw: bool = False) -> Path:
    config = load_b6_case_config()
    rel = Path(config["paths"]["heatnets_source"])
    return (RAW_ROOT if raw else REPO_ROOT) / rel.name if raw else REPO_ROOT / rel


def legacy_operation_root(raw: bool = False) -> Path:
    config = load_b6_case_config()
    rel = Path(config["paths"]["legacy_operation_phase_dir"])
    return (RAW_ROOT if raw else REPO_ROOT) / rel.name if raw else REPO_ROOT / rel


def heatnets_weather_folder() -> Path:
    config = load_b6_case_config()
    return heatnets_source_root(raw=True) / "URBANopt_results" / config["heatnets"]["preferred_weather_folder"]


def load_pickle_compat(path: Path, extra_sys_path: Path | None = None) -> Any:
    validate_exists(path, "pickle file")
    mod = types.ModuleType("pandas.core.indexes.numeric")
    for name in ["Int64Index", "UInt64Index", "Float64Index", "NumericIndex"]:
        if hasattr(pd, name):
            setattr(mod, name, getattr(pd, name))
        else:
            setattr(mod, name, pd.Index)
    sys.modules["pandas.core.indexes.numeric"] = mod
    if extra_sys_path is not None and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
    with path.open("rb") as handle:
        return pickle.load(handle)


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"
