from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[3] / "Dynamic_LCA_GEN_BAU" / ".mplconfig").resolve()),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str((Path(__file__).resolve().parents[3] / "Dynamic_LCA_GEN_BAU" / ".cache").resolve()),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
        write_markdown,
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
        write_markdown,
    )


OUTPUT_ROOT = WORKSPACE_ROOT / "export" / "b6_dynamic_lcia"
DETAIL_ROOT = OUTPUT_ROOT / "details"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
CASE_DISPLAY_LABELS = {"GEN": "GEN", "BAU": "REF"}

QA_REPORT = DOCS_ROOT / "b6_qa_report.md"
MANIFEST_PATH = EXPORT_ROOT / "bw_timex_inputs" / "b6_inventory_manifest.json"
DENOMINATOR_PATH = EXPORT_ROOT / "bw_timex_inputs" / "b6_denominator_summary.csv"
ANNUAL_TOTAL_PATH = EXPORT_ROOT / "comparison" / "gen_vs_bau_b6_annual_total.csv"
EXCHANGE_TABLE_PATH = (
    EXPORT_ROOT
    / "bw_timex_inputs"
    / "b6_operational_exchange_table_heatnets_case_service_loads_heatnets_authoritative_sync.csv"
)
HEATNETS_VALIDATION_PATH = EXPORT_ROOT / "gen" / "heatnets_rerun_validation.csv"
GEN_HOURLY_PATH = EXPORT_ROOT / "gen" / "gen_b6_hourly_total.csv"
BAU_HOURLY_GROUP_PATH = EXPORT_ROOT / "bau" / "bau_b6_hourly_group.csv"

DEFAULT_T0 = dt.datetime(2025, 1, 1)
DEFAULT_HORIZON_YEARS = 100


@dataclass(frozen=True)
class ReadyStatus:
    gen_accepted: bool
    bau_accepted: bool
    dhw_accepted: bool
    denominator_sync_complete: bool
    ready_for_b6_dynamic_lcia: bool


@dataclass(frozen=True)
class InventoryContext:
    project: str
    normalization_mode: str
    foreground_set: str
    denominator_lifetime_kwh_th: float
    denominator_annual_kwh_th: float
    service_life_years: int
    gen_wrapper_key: tuple[str, str]
    bau_wrapper_key: tuple[str, str]
    gen_b6_key: tuple[str, str]
    bau_b6_key: tuple[str, str]
    gen_fg_db: str
    bau_fg_db: str
    background_dbs: list[str]


@dataclass
class CaseResult:
    label: str
    fg_db: str
    wrapper_key: tuple[str, str]
    tlca: Any
    stage_key: tuple[str, str]
    inv_df: pd.DataFrame
    static_annual: pd.DataFrame
    gwp_df: pd.DataFrame
    rf_df: pd.DataFrame
    annual_ghg: pd.DataFrame
    annual_rf: pd.DataFrame
    top_gwp_flows: pd.DataFrame
    top_rf_flows: pd.DataFrame
    top_gwp_sources: pd.DataFrame
    climate_summary: pd.DataFrame


def _require_ready() -> ReadyStatus:
    text = validate_exists(QA_REPORT, "latest Stage B6 QA report").read_text(encoding="utf-8")

    def extract(label: str) -> bool:
        pattern = rf"{re.escape(label)}:\s*`?(True|False)`?"
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(
                f"Could not find `{label}` in {relpath(QA_REPORT)}. "
                "Run the Stage B6 QA pass before dynamic LCIA."
            )
        return match.group(1) == "True"

    status = ReadyStatus(
        gen_accepted=extract("GEN accepted"),
        bau_accepted=extract("BAU accepted"),
        dhw_accepted=extract("DHW surrogate COP accepted as provisional"),
        denominator_sync_complete=extract("Denominator synchronization complete"),
        ready_for_b6_dynamic_lcia=extract("Ready for B6-only dynamic LCIA (GHG and RF)"),
    )
    failed = [
        label
        for label, ok in {
            "GEN accepted": status.gen_accepted,
            "BAU accepted": status.bau_accepted,
            "DHW surrogate COP accepted as provisional": status.dhw_accepted,
            "Denominator synchronization complete": status.denominator_sync_complete,
            "Ready for B6-only dynamic LCIA (GHG and RF)": status.ready_for_b6_dynamic_lcia,
        }.items()
        if not ok
    ]
    if failed:
        raise RuntimeError(
            "Stage B6 dynamic LCIA is blocked because the latest QA report is not fully accepted. "
            f"Failing flags: {failed}"
        )
    return status


def _load_context() -> InventoryContext:
    config = load_b6_case_config()
    manifest = read_json(MANIFEST_PATH)
    denominator = pd.read_csv(validate_exists(DENOMINATOR_PATH, "B6 denominator summary"))

    normalization_mode = str(
        manifest.get("selected_normalization_mode", manifest.get("normalization_mode", ""))
    )
    foreground_set = str(manifest.get("selected_foreground_set", manifest.get("foreground_set", "")))
    if normalization_mode != "heatnets_case_service_loads":
        raise RuntimeError(
            f"Dynamic LCIA must use `heatnets_case_service_loads`, but manifest selected `{normalization_mode}`."
        )
    if foreground_set != "heatnets_authoritative_sync":
        raise RuntimeError(
            f"Dynamic LCIA must use `heatnets_authoritative_sync`, but manifest selected `{foreground_set}`."
        )

    denom_row = denominator.loc[denominator["mode"] == normalization_mode]
    if denom_row.empty:
        raise RuntimeError(
            f"Normalization mode `{normalization_mode}` was not found in {relpath(DENOMINATOR_PATH)}."
        )
    denom = denom_row.iloc[0]

    def as_key(name: str) -> tuple[str, str]:
        if name not in manifest:
            raise RuntimeError(
                f"Manifest is missing `{name}`. Re-run {relpath(Path(__file__).resolve().parents[0] / 'build_bw_timex_b6_inventory.py')} "
                "with `--write-brightway --normalization-mode heatnets_case_service_loads --foreground-set heatnets_authoritative_sync`."
            )
        payload = manifest[name]
        if not isinstance(payload, list) or len(payload) != 2:
            raise RuntimeError(f"Manifest entry `{name}` must be a two-item list, got: {payload!r}")
        return (str(payload[0]), str(payload[1]))

    return InventoryContext(
        project=str(config["project"]["brightway_project"]),
        normalization_mode=normalization_mode,
        foreground_set=foreground_set,
        denominator_lifetime_kwh_th=float(denom["lifetime_delivered_kwh_th"]),
        denominator_annual_kwh_th=float(denom["annual_delivered_kwh_th"]),
        service_life_years=int(denom["service_life_years"]),
        gen_wrapper_key=as_key("gen_b6_dynamic_only_key"),
        bau_wrapper_key=as_key("bau_b6_dynamic_only_key"),
        gen_b6_key=as_key("gen_b6_key"),
        bau_b6_key=as_key("bau_b6_key"),
        gen_fg_db=str(manifest["gen_b6_key"][0]),
        bau_fg_db=str(manifest["bau_b6_key"][0]),
        background_dbs=[str(db) for db in config["project"]["background_dbs"]],
    )


def _choose_climate_method(bd) -> tuple[str, ...]:
    for method in bd.methods:
        text = " | ".join(method).lower()
        if ("climate" in text or "global warming" in text or "gwp" in text) and (
            "100" in text or "gwp100" in text
        ):
            try:
                mm = bd.Method(method)
                mm.process()
                if len(list(mm.load())) > 0:
                    return tuple(method)
            except Exception:
                continue
    raise RuntimeError("No usable static climate GWP100 method was found in the current Brightway project.")


def _build_database_dates(dynamic_fg_db: str, bg_dbs: list[str], bio_db: str) -> dict[str, Any]:
    out: dict[str, Any] = {dynamic_fg_db: "dynamic", bio_db: dt.datetime(1900, 1, 1)}
    for db_name in bg_dbs:
        year = int(re.search(r"(20\d{2})", db_name).group(1))
        out[db_name] = dt.datetime(year, 1, 1)
    return out


def _build_linear_bg_points(bg_dbs: list[str]) -> list[tuple[dt.datetime, str]]:
    points: list[tuple[dt.datetime, str]] = []
    for db_name in bg_dbs:
        year = int(re.search(r"(20\d{2})", db_name).group(1))
        points.append((dt.datetime(year, 1, 1), db_name))
    points.sort()
    return points


def _get_node_from_id(bd, node_id: int):
    try:
        return bd.get_node(id=int(node_id))
    except Exception:
        return bd.get_activity(id=int(node_id))


def _linear_shares(ts: Any, bg_points: list[tuple[dt.datetime, str]]) -> dict[str, float]:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts <= bg_points[0][0]:
        return {bg_points[0][1]: 1.0}
    if ts >= bg_points[-1][0]:
        return {bg_points[-1][1]: 1.0}
    for (t0, db0), (t1, db1) in zip(bg_points[:-1], bg_points[1:]):
        if t0 <= ts <= t1:
            span = (t1 - t0).total_seconds()
            w1 = (ts - t0).total_seconds() / span if span else 0.0
            return {db0: float(1.0 - w1), db1: float(w1)}
    return {bg_points[0][1]: 1.0}


def _patch_temporal_market_shares(tlca, bd, bg_dbs: list[str]) -> None:
    timeline = tlca.timeline.copy()
    if "temporal_market_shares" not in timeline.columns:
        tlca.timeline = timeline
        return

    bg_points = _build_linear_bg_points(bg_dbs)
    patched_producers: dict[int, tuple[str | None, dict[str, float]]] = {}

    for idx, row in timeline.iterrows():
        if not pd.isna(row.get("temporal_market_shares")):
            continue
        producer_id = int(row["producer"])
        try:
            node = _get_node_from_id(bd, producer_id)
        except Exception:
            continue
        producer_db = node.get("database")
        if producer_db not in bg_dbs:
            continue
        shares = _linear_shares(row["date_producer"], bg_points)
        timeline.at[idx, "temporal_market_shares"] = shares
        patched_producers[producer_id] = (node.get("code"), shares)

    tlca.timeline = timeline
    interdb = tlca.interdatabase_activity_mapping
    for producer_id, (code, shares) in patched_producers.items():
        if not code:
            continue
        if producer_id not in interdb:
            interdb[producer_id] = {}
        for db_name in shares:
            try:
                interdb[producer_id][db_name] = bd.get_activity((db_name, code)).id
            except Exception:
                continue
    try:
        interdb.make_reciprocal()
    except Exception:
        pass


def _install_safe_interdatabase_mapping(tlca) -> None:
    def safe_add_interdatabase_activity_mapping_from_timeline(self) -> None:
        if not hasattr(self, "timeline"):
            raise AttributeError("Timeline not yet built. Call TimexLCA.build_timeline() first.")

        filtered_timeline = self.timeline.loc[self.timeline.temporal_market_shares.notnull()]
        unique_producers = [producer for producer in filtered_timeline.producer.unique() if producer in self.nodes]

        self.interdatabase_activity_mapping.update({producer: {} for producer in unique_producers})

        producer_tuples: dict[tuple[str, Any, str], Any] = {}
        for producer in unique_producers:
            producer_node = self.nodes[producer]
            try:
                key = (
                    producer_node["name"],
                    producer_node.get("reference product"),
                    producer_node["location"],
                )
            except Exception:
                continue
            producer_tuples[key] = producer

        unique_tuples = set(producer_tuples)
        for node in self.nodes.values():
            try:
                node_tuple = (node["name"], node.get("reference product"), node["location"])
            except Exception:
                continue
            if node_tuple in unique_tuples:
                producer_id = producer_tuples[node_tuple]
                self.interdatabase_activity_mapping[producer_id][node["database"]] = node.id

        self.interdatabase_activity_mapping.make_reciprocal()

    tlca.add_interdatabase_activity_mapping_from_timeline = types.MethodType(
        safe_add_interdatabase_activity_mapping_from_timeline,
        tlca,
    )


def _install_safe_dynamic_biosphere_builder() -> None:
    from bw_temporalis import TemporalDistribution
    import bw2data as bd
    import scipy.sparse as sp
    from bw_timex.dynamic_biosphere_builder import (
        DynamicBiosphereBuilder,
        convert_date_string_to_datetime,
    )

    if getattr(DynamicBiosphereBuilder, "_gen_safe_patch_applied", False):
        return

    def safe_build_dynamic_biosphere_matrix(self, expand_technosphere: bool = True):
        lci_dict = {}
        temporal_market_lcis = {}

        for row in self.timeline.itertuples():
            idx = row.time_mapped_producer

            if expand_technosphere:
                process_col_index = self.activity_dict[idx]
            else:
                process_col_index = row.Index

            ((original_db, original_code), time) = self.activity_time_mapping.reversed[idx]

            if idx in self.node_collections["temporalized_processes"]:
                time_in_datetime = convert_date_string_to_datetime(self.temporal_grouping, str(time))
                td_producer = TemporalDistribution(
                    date=np.array([time_in_datetime], dtype=self.time_res),
                    amount=np.array([1]),
                ).date
                date = td_producer[0]

                if original_db == "temporalized":
                    try:
                        act = bd.get_node(id=int(row.producer))
                    except Exception:
                        act = bd.get_node(code=original_code)
                else:
                    act = bd.get_node(database=original_db, code=original_code)

                for exc in act.biosphere():
                    if exc.get("temporal_distribution"):
                        td_dates = exc["temporal_distribution"].date
                        td_values = exc["temporal_distribution"].amount
                        if isinstance(td_dates[0], np.datetime64):
                            dates = td_producer
                            values = [
                                exc["amount"]
                                * td_values[
                                    np.argmin(
                                        np.abs(
                                            td_dates.astype(self.time_res)
                                            - td_producer.astype(self.time_res)
                                        )
                                    )
                                ]
                            ]
                        else:
                            dates = td_producer + td_dates
                            values = exc["amount"] * td_values
                    else:
                        dates = td_producer
                        values = [exc["amount"]]

                    for date, amount in zip(dates, values):
                        time_mapped_matrix_idx = self.biosphere_time_mapping.add((exc.input.id, date))
                        self.add_matrix_entry_for_biosphere_flows(
                            row=time_mapped_matrix_idx,
                            col=process_col_index,
                            amount=amount,
                        )

            elif idx in self.node_collections["temporal_markets"]:
                self.temporal_market_cols.append(process_col_index)
                ((original_db, original_code), time) = self.activity_time_mapping.reversed[idx]

                if expand_technosphere:
                    demand = self.demand_from_technosphere(idx, process_col_index)
                else:
                    demand = self.demand_from_timeline(row)

                if demand:
                    for act, amount in demand.items():
                        if act not in lci_dict:
                            self.lca_obj.redo_lci({act: 1})
                            lci_dict[act] = self.lca_obj.inventory
                        if idx not in temporal_market_lcis:
                            temporal_market_lcis[idx] = lci_dict[act] * amount
                        else:
                            temporal_market_lcis[idx] += lci_dict[act] * amount

                    aggregated_inventory = temporal_market_lcis[idx].sum(axis=1)
                    temporal_market_lcis[idx] *= self.dynamic_supply_array[process_col_index]

                    for row_idx, amount in enumerate(aggregated_inventory.A1):
                        bioflow = self.lca_obj.dicts.biosphere.reversed[row_idx]
                        ((_, _), time) = self.activity_time_mapping.reversed[idx]

                        time_in_datetime = convert_date_string_to_datetime(self.temporal_grouping, str(time))
                        td_producer = TemporalDistribution(
                            date=np.array([str(time_in_datetime)], dtype=self.time_res),
                            amount=np.array([1]),
                        ).date
                        date = td_producer[0]

                        time_mapped_matrix_idx = self.biosphere_time_mapping.add((bioflow, date))
                        self.add_matrix_entry_for_biosphere_flows(
                            row=time_mapped_matrix_idx,
                            col=process_col_index,
                            amount=amount,
                        )

        if expand_technosphere:
            ncols = len(self.activity_time_mapping)
        else:
            ncols = len(self.timeline)

        if not self._matrix_entries:
            dynamic_biosphere_matrix = sp.csr_matrix((0, ncols))
            return dynamic_biosphere_matrix, temporal_market_lcis
        rows = []
        cols = []
        values = []
        for (r, c), amount in self._matrix_entries.items():
            rows.append(r)
            cols.append(c)
            values.append(amount)
        shape = (max(rows) + 1, ncols)
        dynamic_biosphere_matrix = sp.coo_matrix((values, (rows, cols)), shape).tocsr()
        return dynamic_biosphere_matrix, temporal_market_lcis

    DynamicBiosphereBuilder.build_dynamic_biosphere_matrix = safe_build_dynamic_biosphere_matrix
    DynamicBiosphereBuilder._gen_safe_patch_applied = True


def _install_safe_temporalis_init() -> None:
    import bw2data as bd
    from bw_temporalis.lca import AD, NewNodeEachVisitGraphTraversal, TemporalDistribution, TemporalisLCA

    if getattr(TemporalisLCA, "_gen_safe_patch_applied", False):
        return

    def safe_init(
        self,
        lca_object,
        starting_datetime: dt.datetime | str = "now",
        cutoff: float | None = 5e-4,
        biosphere_cutoff: float | None = 1e-6,
        max_calc: int | None = 2000,
        static_activity_indices: set[int] | None = None,
        skip_coproducts: bool | None = False,
        functional_unit_unique_id: int | None = -1,
        graph_traversal=NewNodeEachVisitGraphTraversal,
    ):
        self.lca_object = lca_object
        self.unique_id = functional_unit_unique_id
        self.t0 = TemporalDistribution(
            np.array([np.datetime64(starting_datetime)]),
            np.array([1]),
        )

        if static_activity_indices is None:
            static_activity_indices = set()

        for db_name in bd.databases:
            if bd.databases[db_name].get("static"):
                static_activity_indices.update(
                    obj[0] for obj in AD.select(AD.id).where(AD.database == db_name).tuples()
                )

        translated_static_indices = {
            self.lca_object.dicts.activity[idx]
            for idx in static_activity_indices
            if idx in self.lca_object.dicts.activity
        }

        print("Starting graph traversal")
        gt = graph_traversal.calculate(
            lca_object=lca_object,
            static_activity_indices=translated_static_indices,
            max_calc=max_calc,
            cutoff=cutoff,
            biosphere_cutoff=biosphere_cutoff,
            separate_biosphere_flows=True,
            skip_coproducts=skip_coproducts,
            functional_unit_unique_id=functional_unit_unique_id,
        )
        print("Calculation count:", gt["calculation_count"])
        self.nodes = gt["nodes"]
        self.edges = gt["edges"]
        self.edge_mapping = defaultdict(list)
        for edge in self.edges:
            self.edge_mapping[edge.consumer_unique_id].append(edge)

        self.flows = gt["flows"]
        self.flow_mapping = defaultdict(list)
        for flow in self.flows:
            self.flow_mapping[flow.activity_unique_id].append(flow)

    TemporalisLCA.__init__ = safe_init
    TemporalisLCA._gen_safe_patch_applied = True


def _build_characterization_functions(dynamic_inventory_df: pd.DataFrame, bd) -> dict[int, Any]:
    mapping: dict[int, Any] = {}
    from dynamic_characterization.ipcc_ar6 import (
        characterize_ch4,
        characterize_co2,
        characterize_co2_uptake,
        characterize_n2o,
    )

    for flow_id in dynamic_inventory_df["flow"].dropna().unique():
        try:
            flow = bd.get_node(id=int(flow_id))
        except Exception:
            continue
        name = str(flow.get("name", "")).lower()
        if "carbon dioxide" in name:
            mapping[int(flow_id)] = (
                characterize_co2_uptake if ("uptake" in name or "sequestration" in name) else characterize_co2
            )
        elif name.startswith("methane"):
            mapping[int(flow_id)] = characterize_ch4
        elif "dinitrogen monoxide" in name or "nitrous oxide" in name:
            mapping[int(flow_id)] = characterize_n2o
    if not mapping:
        raise RuntimeError("No CO2, CH4, or N2O flows were found in the dynamic inventory for AR6 characterization.")
    return mapping


def _annual_impacts_from_method(dynamic_inventory_df: pd.DataFrame, method: tuple[str, ...], bd) -> pd.DataFrame:
    method_obj = bd.Method(method)
    method_obj.process()
    cf_pairs = list(method_obj.load())
    cf_by_key = {
        tuple(flow_key_like) if isinstance(flow_key_like, list) else flow_key_like: float(cf)
        for flow_key_like, cf in cf_pairs
    }

    df = dynamic_inventory_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    cf_by_flow_id: dict[int, float] = {}
    for flow_id in df["flow"].dropna().unique():
        node = bd.get_node(id=int(flow_id))
        cf_by_flow_id[int(flow_id)] = cf_by_key.get(node.key, 0.0)

    df["static_cf"] = df["flow"].astype(int).map(cf_by_flow_id)
    df["impact"] = df["amount"] * df["static_cf"]
    return df.groupby("year", as_index=False)["impact"].sum().sort_values("year").reset_index(drop=True)


def _build_activity_label_map_from_timeline(tlca) -> dict[int, str]:
    label_map: dict[int, str] = {}
    timeline = tlca.timeline.copy()
    for id_col, name_col in [("producer", "producer_name"), ("consumer", "consumer_name")]:
        if id_col not in timeline.columns or name_col not in timeline.columns:
            continue
        subset = timeline[[id_col, name_col]].dropna().drop_duplicates()
        for _, row in subset.iterrows():
            try:
                label_map[int(row[id_col])] = str(row[name_col])
            except Exception:
                continue
    return label_map


def _attach_labels(df: pd.DataFrame, tlca, bd) -> pd.DataFrame:
    out = df.copy()
    if "flow" in out.columns:
        flow_map: dict[int, str] = {}
        for flow_id in pd.Series(out["flow"]).dropna().unique():
            try:
                flow_map[int(flow_id)] = str(bd.get_node(id=int(flow_id)).get("name", f"flow_{flow_id}"))
            except Exception:
                flow_map[int(flow_id)] = f"flow_{flow_id}"
        out["flow_label"] = out["flow"].map(
            lambda value: flow_map.get(int(value), f"flow_{value}") if pd.notna(value) else "flow_nan"
        )

    if "activity" in out.columns:
        activity_map = _build_activity_label_map_from_timeline(tlca)
        out["activity_label"] = out["activity"].map(
            lambda value: activity_map.get(int(value), f"activity_{int(value)}")
            if pd.notna(value)
            else "activity_nan"
        )
    return out


def _source_bucket(activity_label: str) -> str:
    text = activity_label.lower()
    if "market for electricity" in text or "electricity" in text:
        return "Electricity supply chain"
    if "natural gas" in text:
        return "Natural gas supply chain"
    if "fuel oil" in text or "light fuel oil" in text:
        return "Fuel oil supply chain"
    if "b6 operational energy" in text or "b6-only dynamic wrapper" in text:
        return "Direct operational emissions"
    return "Other upstream source"


def _top_flow_table(
    characterized_df: pd.DataFrame,
    tlca,
    bd,
    denominator: float,
    metric_label: str,
    top_n: int = 10,
) -> pd.DataFrame:
    labeled = _attach_labels(characterized_df, tlca=tlca, bd=bd)
    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled["year"] = labeled["date"].dt.year
    totals = labeled.groupby("flow_label")["amount"].sum().abs().sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    rank_map = {label: rank + 1 for rank, label in enumerate(keep)}

    out = (
        labeled.loc[labeled["flow_label"].isin(keep)]
        .groupby(["year", "flow_label"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": f"{metric_label}_per_fu"})
    )
    out[f"{metric_label}_case_total"] = out[f"{metric_label}_per_fu"] * denominator
    out["rank"] = out["flow_label"].map(rank_map)
    out["total_abs_contribution_per_fu"] = out["flow_label"].map(totals.to_dict())
    return out.sort_values(["rank", "year"]).reset_index(drop=True)


def _top_source_table(
    characterized_df: pd.DataFrame,
    tlca,
    bd,
    denominator: float,
    metric_label: str,
    top_n: int = 7,
) -> pd.DataFrame:
    labeled = _attach_labels(characterized_df, tlca=tlca, bd=bd)
    labeled["date"] = pd.to_datetime(labeled["date"])
    labeled["year"] = labeled["date"].dt.year
    if "activity_label" not in labeled.columns:
        labeled["activity_label"] = "activity_nan"
    labeled["source_bucket"] = labeled["activity_label"].fillna("activity_nan").map(_source_bucket)
    labeled["flow_source_label"] = labeled["flow_label"] + " | " + labeled["source_bucket"]

    totals = labeled.groupby("flow_source_label")["amount"].sum().abs().sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    rank_map = {label: rank + 1 for rank, label in enumerate(keep)}

    out = (
        labeled.loc[labeled["flow_source_label"].isin(keep)]
        .groupby(["year", "flow_source_label"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": f"{metric_label}_per_fu"})
    )
    out[f"{metric_label}_case_total"] = out[f"{metric_label}_per_fu"] * denominator
    out["rank"] = out["flow_source_label"].map(rank_map)
    out["total_abs_contribution_per_fu"] = out["flow_source_label"].map(totals.to_dict())
    return out.sort_values(["rank", "year"]).reset_index(drop=True)


def _full_year_frame(start_year: int, horizon_years: int) -> pd.DataFrame:
    return pd.DataFrame({"year": list(range(start_year, start_year + horizon_years))})


def _build_annual_ghg_table(
    static_annual: pd.DataFrame,
    dynamic_annual: pd.DataFrame,
    denominator: float,
    start_year: int,
    horizon_years: int,
    normalization_mode: str,
    foreground_set: str,
) -> pd.DataFrame:
    out = _full_year_frame(start_year, horizon_years)
    out = out.merge(
        static_annual.rename(columns={"impact": "annual_time_explicit_static_climate_kgCO2e_per_fu"}),
        on="year",
        how="left",
    )
    out = out.merge(
        dynamic_annual.rename(columns={"amount": "annual_dynamic_GWP100_kgCO2e_per_fu"}),
        on="year",
        how="left",
    )
    for col in [
        "annual_time_explicit_static_climate_kgCO2e_per_fu",
        "annual_dynamic_GWP100_kgCO2e_per_fu",
    ]:
        out[col] = out[col].fillna(0.0)
    out["annual_time_explicit_static_climate_kgCO2e_case_total"] = (
        out["annual_time_explicit_static_climate_kgCO2e_per_fu"] * denominator
    )
    out["annual_dynamic_GWP100_kgCO2e_case_total"] = out["annual_dynamic_GWP100_kgCO2e_per_fu"] * denominator
    out["cumulative_time_explicit_static_climate_kgCO2e_per_fu"] = (
        out["annual_time_explicit_static_climate_kgCO2e_per_fu"].cumsum()
    )
    out["cumulative_dynamic_GWP100_kgCO2e_per_fu"] = out["annual_dynamic_GWP100_kgCO2e_per_fu"].cumsum()
    out["cumulative_time_explicit_static_climate_kgCO2e_case_total"] = (
        out["annual_time_explicit_static_climate_kgCO2e_case_total"].cumsum()
    )
    out["cumulative_dynamic_GWP100_kgCO2e_case_total"] = out["annual_dynamic_GWP100_kgCO2e_case_total"].cumsum()
    out["normalization_mode"] = normalization_mode
    out["foreground_set"] = foreground_set
    out["denominator_lifetime_kwh_th"] = denominator
    return out


def _build_annual_rf_table(
    annual_rf: pd.DataFrame,
    denominator: float,
    start_year: int,
    horizon_years: int,
    normalization_mode: str,
    foreground_set: str,
) -> pd.DataFrame:
    out = _full_year_frame(start_year, horizon_years)
    out = out.merge(
        annual_rf.rename(columns={"amount": "annual_radiative_forcing_W_per_m2_per_fu"}),
        on="year",
        how="left",
    )
    out["annual_radiative_forcing_W_per_m2_per_fu"] = out["annual_radiative_forcing_W_per_m2_per_fu"].fillna(0.0)
    out["annual_radiative_forcing_W_per_m2_case_total"] = out["annual_radiative_forcing_W_per_m2_per_fu"] * denominator
    out["cumulative_discrete_annual_rf_Wyr_per_m2_per_fu"] = out["annual_radiative_forcing_W_per_m2_per_fu"].cumsum()
    out["cumulative_discrete_annual_rf_Wyr_per_m2_case_total"] = (
        out["annual_radiative_forcing_W_per_m2_case_total"].cumsum()
    )
    out["normalization_mode"] = normalization_mode
    out["foreground_set"] = foreground_set
    out["denominator_lifetime_kwh_th"] = denominator
    return out


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save_figure(fig: plt.Figure, basepath: Path) -> None:
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(basepath.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _title_suffix(context: InventoryContext) -> str:
    denom_million = context.denominator_lifetime_kwh_th / 1_000_000.0
    return f"Normalization: audited HEATNETS denominator ({denom_million:.2f} million kWh_th lifetime)"


def _case_label(case: str) -> str:
    return CASE_DISPLAY_LABELS.get(case, case)


def _plot_compare_lines(
    compare_df: pd.DataFrame,
    gen_col: str,
    bau_col: str,
    ylabel: str,
    title: str,
    basepath: Path,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(compare_df["year"], compare_df[gen_col], linewidth=2.2, color="#1b9e77", label=_case_label("GEN"))
    ax.plot(compare_df["year"], compare_df[bau_col], linewidth=2.2, color="#d95f02", label=_case_label("BAU"))
    ax.set_xlabel("Calendar year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=True, loc="best")
    _save_figure(fig, basepath)


def _plot_stacked_bars(
    table: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    basepath: Path,
    top_n: int = 6,
) -> None:
    _set_plot_style()
    totals = table.groupby(category_col)[value_col].sum().abs().sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    data = table.copy()
    data["plot_category"] = data[category_col].where(data[category_col].isin(keep), "Other")
    pivot = (
        data.groupby(["year", "plot_category"], as_index=False)[value_col]
        .sum()
        .pivot(index="year", columns="plot_category", values=value_col)
        .fillna(0.0)
        .sort_index()
    )

    ordered_cols = [col for col in keep if col in pivot.columns] + [col for col in ["Other"] if col in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    bottom = np.zeros(len(pivot.index))
    cmap = plt.get_cmap("tab20")
    for idx, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy()
        ax.bar(
            pivot.index.to_numpy(),
            values,
            bottom=bottom,
            width=0.85,
            color=cmap(idx),
            label=column,
            linewidth=0.0,
        )
        bottom = bottom + values

    ax.set_xlabel("Calendar year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save_figure(fig, basepath)


def _plot_top_flow_lines(
    table: pd.DataFrame,
    flow_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    basepath: Path,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    cmap = plt.get_cmap("tab10")
    for idx, (label, group) in enumerate(table.groupby(flow_col)):
        group = group.sort_values("year")
        ax.plot(
            group["year"],
            group[value_col],
            linewidth=2.0,
            color=cmap(idx % 10),
            label=label,
        )
    ax.set_xlabel("Calendar year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save_figure(fig, basepath)


def _run_case(
    label: str,
    fg_db: str,
    wrapper_key: tuple[str, str],
    context: InventoryContext,
    method: tuple[str, ...],
    t0: dt.datetime,
    horizon_years: int,
) -> CaseResult:
    import bw2data as bd
    from bw_timex import TimexLCA
    import dynamic_characterization.dynamic_characterization as dc

    bio_db = "biosphere3" if "biosphere3" in bd.databases else "biosphere"
    stage_key = wrapper_key

    _install_safe_temporalis_init()
    tlca = TimexLCA({stage_key: 1.0}, method, _build_database_dates(fg_db, context.background_dbs, bio_db))
    _install_safe_interdatabase_mapping(tlca)
    _install_safe_dynamic_biosphere_builder()
    tlca.build_timeline(
        starting_datetime=t0.strftime("%Y-%m-%d"),
        temporal_grouping="year",
        interpolation_type="linear",
        cutoff=1e-12,
        max_calc=400000,
    )
    _patch_temporal_market_shares(tlca, bd=bd, bg_dbs=context.background_dbs)

    tlca.lci()
    tlca.static_lcia()

    inv_df = tlca.dynamic_inventory_df.copy()
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    static_annual = _annual_impacts_from_method(inv_df, method=method, bd=bd)

    characterization_functions = _build_characterization_functions(inv_df, bd=bd)
    tlca.dynamic_lcia(
        metric="GWP",
        time_horizon=horizon_years,
        characterization_functions=characterization_functions,
    )

    gwp_df = dc.characterize(
        dynamic_inventory_df=inv_df[["date", "flow", "activity", "amount"]].copy(),
        metric="GWP",
        characterization_functions=characterization_functions,
        time_horizon=horizon_years,
        fixed_time_horizon=True,
        time_horizon_start=t0,
    )
    rf_df = dc.characterize(
        dynamic_inventory_df=inv_df[["date", "flow", "activity", "amount"]].copy(),
        metric="radiative_forcing",
        characterization_functions=characterization_functions,
        time_horizon=horizon_years,
        fixed_time_horizon=True,
        time_horizon_start=t0,
    )

    for frame in [gwp_df, rf_df]:
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year

    annual_gwp = gwp_df.groupby("year", as_index=False)["amount"].sum().sort_values("year").reset_index(drop=True)
    annual_rf = rf_df.groupby("year", as_index=False)["amount"].sum().sort_values("year").reset_index(drop=True)

    ghg_table = _build_annual_ghg_table(
        static_annual=static_annual,
        dynamic_annual=annual_gwp,
        denominator=context.denominator_lifetime_kwh_th,
        start_year=t0.year,
        horizon_years=horizon_years,
        normalization_mode=context.normalization_mode,
        foreground_set=context.foreground_set,
    )
    rf_table = _build_annual_rf_table(
        annual_rf=annual_rf,
        denominator=context.denominator_lifetime_kwh_th,
        start_year=t0.year,
        horizon_years=horizon_years,
        normalization_mode=context.normalization_mode,
        foreground_set=context.foreground_set,
    )

    top_gwp_flows = _top_flow_table(
        characterized_df=gwp_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="dynamic_GWP100_kgCO2e",
    )
    top_rf_flows = _top_flow_table(
        characterized_df=rf_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="radiative_forcing_W_per_m2",
    )
    top_gwp_sources = _top_source_table(
        characterized_df=gwp_df,
        tlca=tlca,
        bd=bd,
        denominator=context.denominator_lifetime_kwh_th,
        metric_label="dynamic_GWP100_kgCO2e",
    )

    climate_summary = pd.DataFrame(
        [
            {
                "case": label,
                "stage": "B6-only",
                "project": context.project,
                "fg_db": fg_db,
                "wrapper_key": str(stage_key),
                "normalization_mode": context.normalization_mode,
                "foreground_set": context.foreground_set,
                "denominator_lifetime_kwh_th": context.denominator_lifetime_kwh_th,
                "base_score_static_climate_per_fu": float(tlca.base_score),
                "static_score_timex_climate_per_fu": float(tlca.static_score),
                "dynamic_score_gwp_per_fu": float(tlca.dynamic_score),
                "dynamic_GWP100_total_kgCO2e_per_fu": float(ghg_table["annual_dynamic_GWP100_kgCO2e_per_fu"].sum()),
                "dynamic_GWP100_total_kgCO2e_case_total": float(
                    ghg_table["annual_dynamic_GWP100_kgCO2e_case_total"].sum()
                ),
                "peak_annual_rf_W_per_m2_per_fu": float(rf_table["annual_radiative_forcing_W_per_m2_per_fu"].max()),
                "peak_annual_rf_W_per_m2_case_total": float(
                    rf_table["annual_radiative_forcing_W_per_m2_case_total"].max()
                ),
                "peak_annual_rf_year": int(
                    rf_table.loc[
                        rf_table["annual_radiative_forcing_W_per_m2_per_fu"].idxmax(),
                        "year",
                    ]
                ),
                "time_horizon_years": horizon_years,
                "time_horizon_start": t0.strftime("%Y-%m-%d"),
            }
        ]
    )

    return CaseResult(
        label=label,
        fg_db=fg_db,
        wrapper_key=wrapper_key,
        tlca=tlca,
        stage_key=stage_key,
        inv_df=inv_df,
        static_annual=static_annual,
        gwp_df=gwp_df,
        rf_df=rf_df,
        annual_ghg=ghg_table,
        annual_rf=rf_table,
        top_gwp_flows=top_gwp_flows,
        top_rf_flows=top_rf_flows,
        top_gwp_sources=top_gwp_sources,
        climate_summary=climate_summary,
    )


def _sanity_checks(context: InventoryContext) -> pd.DataFrame:
    exchange = pd.read_csv(validate_exists(EXCHANGE_TABLE_PATH, "authoritative B6 exchange table"))
    annual = pd.read_csv(validate_exists(ANNUAL_TOTAL_PATH, "annual B6 comparison table"))
    heatnets_validation = pd.read_csv(validate_exists(HEATNETS_VALIDATION_PATH, "HEATNETS rerun validation"))
    annual_row = annual.iloc[0]

    def exchange_amount(case: str, key: str) -> float:
        row = exchange.loc[
            (exchange["case"] == case) & (exchange["exchange_key"] == key) & (exchange["normalization_mode"] == context.normalization_mode)
        ]
        if row.empty:
            raise RuntimeError(f"Missing exchange `{key}` for case `{case}` in {relpath(EXCHANGE_TABLE_PATH)}.")
        return float(row["input_amount_total_over_life"].iloc[0])

    gen_annual_from_exchange = exchange_amount("GEN", "electricity_medium_voltage") / context.service_life_years
    bau_elec_annual_from_exchange = exchange_amount("BAU", "electricity_medium_voltage") / context.service_life_years
    bau_gas_annual_from_exchange = (
        exchange.loc[
            (exchange["case"] == "BAU")
            & (exchange["exchange_key"] == "natural_gas_high_pressure")
            & (exchange["normalization_mode"] == context.normalization_mode),
            "input_amount_total_over_life",
        ].iloc[0]
        * float(load_b6_case_config()["bau"]["natural_gas_kwh_per_m3"])
        / context.service_life_years
    )
    bau_oil_annual_from_exchange = (
        exchange.loc[
            (exchange["case"] == "BAU")
            & (exchange["exchange_key"] == "light_fuel_oil")
            & (exchange["normalization_mode"] == context.normalization_mode),
            "input_amount_total_over_life",
        ].iloc[0]
        * float(load_b6_case_config()["bau"]["fuel_oil_kwh_per_kg"])
        / context.service_life_years
    )

    electricity_rows = exchange.loc[
        (exchange["exchange_key"] == "electricity_medium_voltage")
        & (exchange["normalization_mode"] == context.normalization_mode)
    ]
    bau_direct_rows = exchange.loc[
        (exchange["case"] == "BAU")
        & (exchange["exchange_key"].isin(["carbon_dioxide_fossil", "methane_fossil", "dinitrogen_monoxide"]))
    ]

    checks = [
        {
            "check": "GEN annual electricity reconciles with corrected B6 exchange table",
            "value": gen_annual_from_exchange,
            "expected": float(annual_row["GEN_total_B6_kWh_el_annual"]),
            "status": "pass"
            if np.isclose(gen_annual_from_exchange, float(annual_row["GEN_total_B6_kWh_el_annual"]), rtol=0, atol=1e-6)
            else "fail",
        },
        {
            "check": "BAU annual electricity reconciles with corrected B6 exchange table",
            "value": bau_elec_annual_from_exchange,
            "expected": float(annual_row["BAU_total_site_energy_electric_kWh_annual"]),
            "status": "pass"
            if np.isclose(
                bau_elec_annual_from_exchange,
                float(annual_row["BAU_total_site_energy_electric_kWh_annual"]),
                rtol=0,
                atol=1e-6,
            )
            else "fail",
        },
        {
            "check": "BAU annual gas fuel reconciles with corrected B6 exchange table",
            "value": bau_gas_annual_from_exchange,
            "expected": float(annual_row["BAU_total_site_energy_gas_kWh_fuel_annual"]),
            "status": "pass"
            if np.isclose(
                bau_gas_annual_from_exchange,
                float(annual_row["BAU_total_site_energy_gas_kWh_fuel_annual"]),
                rtol=0,
                atol=1e-6,
            )
            else "fail",
        },
        {
            "check": "BAU annual oil fuel reconciles with corrected B6 exchange table",
            "value": bau_oil_annual_from_exchange,
            "expected": float(annual_row["BAU_total_site_energy_oil_kWh_fuel_annual"]),
            "status": "pass"
            if np.isclose(
                bau_oil_annual_from_exchange,
                float(annual_row["BAU_total_site_energy_oil_kWh_fuel_annual"]),
                rtol=0,
                atol=1e-6,
            )
            else "fail",
        },
        {
            "check": "GEN and BAU denominator match",
            "value": float(annual_row["same_delivered_denominator_confirmed"]),
            "expected": 1.0,
            "status": "pass" if int(annual_row["same_delivered_denominator_confirmed"]) == 1 else "fail",
        },
        {
            "check": "Historical broken output_elec.csv is not used as Stage B6 source",
            "value": relpath(GEN_HOURLY_PATH),
            "expected": "Corrected HEATNETS rerun export",
            "status": "pass",
        },
        {
            "check": "No duplicate electricity exchange rows remain in authoritative B6 table",
            "value": int(len(electricity_rows)),
            "expected": 2,
            "status": "pass" if int(len(electricity_rows)) == 2 else "fail",
        },
        {
            "check": "BAU direct emissions remain biosphere flows",
            "value": int((bau_direct_rows["exchange_kind"] == "biosphere").all()),
            "expected": 1,
            "status": "pass" if bool((bau_direct_rows["exchange_kind"] == "biosphere").all()) else "fail",
        },
        {
            "check": "GEN HEATNETS rerun has zero negative HVAC hours",
            "value": int(heatnets_validation["negative_hvac_hours"].iloc[0]),
            "expected": 0,
            "status": "pass" if int(heatnets_validation["negative_hvac_hours"].iloc[0]) == 0 else "fail",
        },
    ]
    return pd.DataFrame(checks)


def _comparison_tables(gen: CaseResult, bau: CaseResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    ghg = gen.annual_ghg[["year", "annual_dynamic_GWP100_kgCO2e_per_fu", "annual_dynamic_GWP100_kgCO2e_case_total", "cumulative_dynamic_GWP100_kgCO2e_per_fu", "cumulative_dynamic_GWP100_kgCO2e_case_total"]].merge(
        bau.annual_ghg[["year", "annual_dynamic_GWP100_kgCO2e_per_fu", "annual_dynamic_GWP100_kgCO2e_case_total", "cumulative_dynamic_GWP100_kgCO2e_per_fu", "cumulative_dynamic_GWP100_kgCO2e_case_total"]],
        on="year",
        how="outer",
        suffixes=("_GEN", "_BAU"),
    ).sort_values("year").fillna(0.0)
    ghg["annual_difference_GEN_minus_BAU_kgCO2e_per_fu"] = (
        ghg["annual_dynamic_GWP100_kgCO2e_per_fu_GEN"] - ghg["annual_dynamic_GWP100_kgCO2e_per_fu_BAU"]
    )
    ghg["annual_difference_GEN_minus_BAU_kgCO2e_case_total"] = (
        ghg["annual_dynamic_GWP100_kgCO2e_case_total_GEN"] - ghg["annual_dynamic_GWP100_kgCO2e_case_total_BAU"]
    )
    ghg["cumulative_difference_GEN_minus_BAU_kgCO2e_per_fu"] = (
        ghg["cumulative_dynamic_GWP100_kgCO2e_per_fu_GEN"] - ghg["cumulative_dynamic_GWP100_kgCO2e_per_fu_BAU"]
    )
    ghg["cumulative_difference_GEN_minus_BAU_kgCO2e_case_total"] = (
        ghg["cumulative_dynamic_GWP100_kgCO2e_case_total_GEN"]
        - ghg["cumulative_dynamic_GWP100_kgCO2e_case_total_BAU"]
    )

    rf = gen.annual_rf[["year", "annual_radiative_forcing_W_per_m2_per_fu", "annual_radiative_forcing_W_per_m2_case_total", "cumulative_discrete_annual_rf_Wyr_per_m2_per_fu", "cumulative_discrete_annual_rf_Wyr_per_m2_case_total"]].merge(
        bau.annual_rf[["year", "annual_radiative_forcing_W_per_m2_per_fu", "annual_radiative_forcing_W_per_m2_case_total", "cumulative_discrete_annual_rf_Wyr_per_m2_per_fu", "cumulative_discrete_annual_rf_Wyr_per_m2_case_total"]],
        on="year",
        how="outer",
        suffixes=("_GEN", "_BAU"),
    ).sort_values("year").fillna(0.0)
    rf["annual_difference_GEN_minus_BAU_W_per_m2_per_fu"] = (
        rf["annual_radiative_forcing_W_per_m2_per_fu_GEN"] - rf["annual_radiative_forcing_W_per_m2_per_fu_BAU"]
    )
    rf["annual_difference_GEN_minus_BAU_W_per_m2_case_total"] = (
        rf["annual_radiative_forcing_W_per_m2_case_total_GEN"] - rf["annual_radiative_forcing_W_per_m2_case_total_BAU"]
    )
    rf["cumulative_difference_GEN_minus_BAU_Wyr_per_m2_per_fu"] = (
        rf["cumulative_discrete_annual_rf_Wyr_per_m2_per_fu_GEN"]
        - rf["cumulative_discrete_annual_rf_Wyr_per_m2_per_fu_BAU"]
    )
    rf["cumulative_difference_GEN_minus_BAU_Wyr_per_m2_case_total"] = (
        rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN"]
        - rf["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU"]
    )
    return ghg, rf


def _summary_rows(
    context: InventoryContext,
    method: tuple[str, ...],
    ready: ReadyStatus,
    sanity: pd.DataFrame,
    gen: CaseResult,
    bau: CaseResult,
    horizon_years: int,
    t0: dt.datetime,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"category": "project", "key": "brightway_project", "value": context.project},
        {"category": "mode", "key": "normalization_mode", "value": context.normalization_mode},
        {"category": "mode", "key": "foreground_set", "value": context.foreground_set},
        {"category": "mode", "key": "denominator_lifetime_kwh_th", "value": context.denominator_lifetime_kwh_th},
        {"category": "mode", "key": "denominator_annual_kwh_th", "value": context.denominator_annual_kwh_th},
        {"category": "mode", "key": "service_life_years", "value": context.service_life_years},
        {"category": "inputs", "key": "gen_hourly_source", "value": relpath(GEN_HOURLY_PATH)},
        {"category": "inputs", "key": "bau_hourly_source", "value": relpath(BAU_HOURLY_GROUP_PATH)},
        {"category": "inputs", "key": "authoritative_exchange_table", "value": relpath(EXCHANGE_TABLE_PATH)},
        {"category": "methods", "key": "static_climate_method", "value": " | ".join(method)},
        {"category": "methods", "key": "dynamic_ghg_method", "value": "dynamic_characterization IPCC AR6 GWP100"},
        {"category": "methods", "key": "dynamic_rf_method", "value": "dynamic_characterization IPCC AR6 radiative forcing"},
        {"category": "methods", "key": "time_horizon_years", "value": horizon_years},
        {"category": "methods", "key": "time_horizon_start", "value": t0.strftime("%Y-%m-%d")},
        {"category": "scope", "key": "stage_scope", "value": "B6-only operational stage"},
        {"category": "scope", "key": "cambium_used_in_integrated_lcia", "value": False},
        {"category": "scope", "key": "historical_output_elec_used", "value": False},
        {"category": "scope", "key": "gen_wrapper_key", "value": str(context.gen_wrapper_key)},
        {"category": "scope", "key": "bau_wrapper_key", "value": str(context.bau_wrapper_key)},
        {"category": "qa", "key": "gen_accepted", "value": ready.gen_accepted},
        {"category": "qa", "key": "bau_accepted", "value": ready.bau_accepted},
        {"category": "qa", "key": "dhw_accepted", "value": ready.dhw_accepted},
        {"category": "qa", "key": "denominator_sync_complete", "value": ready.denominator_sync_complete},
        {"category": "qa", "key": "ready_for_b6_dynamic_lcia", "value": ready.ready_for_b6_dynamic_lcia},
        {
            "category": "results",
            "key": "gen_dynamic_GWP100_total_kgCO2e_case_total",
            "value": float(gen.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        },
        {
            "category": "results",
            "key": "bau_dynamic_GWP100_total_kgCO2e_case_total",
            "value": float(bau.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()),
        },
        {
            "category": "results",
            "key": "gen_peak_annual_rf_W_per_m2_case_total",
            "value": float(gen.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].max()),
        },
        {
            "category": "results",
            "key": "bau_peak_annual_rf_W_per_m2_case_total",
            "value": float(bau.annual_rf["annual_radiative_forcing_W_per_m2_case_total"].max()),
        },
    ]

    for _, row in sanity.iterrows():
        rows.append(
            {
                "category": "sanity_check",
                "key": row["check"],
                "value": row["status"],
            }
        )
    return pd.DataFrame(rows)


def _write_summary_markdown(
    context: InventoryContext,
    ready: ReadyStatus,
    sanity: pd.DataFrame,
    gen: CaseResult,
    bau: CaseResult,
    ghg_compare: pd.DataFrame,
    rf_compare: pd.DataFrame,
) -> None:
    gen_top_ghg = gen.top_gwp_sources.groupby("flow_source_label")["dynamic_GWP100_kgCO2e_case_total"].sum().abs().sort_values(ascending=False).head(3)
    bau_top_ghg = bau.top_gwp_sources.groupby("flow_source_label")["dynamic_GWP100_kgCO2e_case_total"].sum().abs().sort_values(ascending=False).head(3)
    gen_top_rf = gen.top_rf_flows.groupby("flow_label")["radiative_forcing_W_per_m2_case_total"].sum().abs().sort_values(ascending=False).head(3)
    bau_top_rf = bau.top_rf_flows.groupby("flow_label")["radiative_forcing_W_per_m2_case_total"].sum().abs().sort_values(ascending=False).head(3)

    def bullet_lines(series: pd.Series) -> str:
        return "\n".join(f"- {idx}: {value:,.6g}" for idx, value in series.items())

    summary = f"""# Stage B6 Dynamic LCIA Summary

## What Was Calculated

- Stage B6 only for GEN and BAU using the synchronized HEATNETS-authoritative denominator mode.
- Time-explicit static climate LCIA from the dynamic inventory using the project Brightway climate method.
- Dynamic GHG using `dynamic_characterization` IPCC AR6 GWP100.
- Dynamic radiative forcing trajectories using `dynamic_characterization` IPCC AR6 radiative forcing.
- All outputs were scaled both per functional unit and to the audited HEATNETS lifetime denominator of `{context.denominator_lifetime_kwh_th:,.3f} kWh_th`.

## Methods Used

- Brightway project: `{context.project}`
- Foreground set: `{context.foreground_set}`
- Normalization mode: `{context.normalization_mode}`
- B6-only timed wrapper activities:
  - GEN: `{context.gen_wrapper_key}`
  - BAU: `{context.bau_wrapper_key}`
- Existing prospective background family: `{"`, `".join(context.background_dbs)}`
- Cambium was not used in the integrated dynamic LCIA pathway.

## Robustness

- Latest QA report status:
  - GEN accepted: `{ready.gen_accepted}`
  - BAU accepted: `{ready.bau_accepted}`
  - DHW accepted: `{ready.dhw_accepted}`
  - Denominator synchronization complete: `{ready.denominator_sync_complete}`
  - Ready for B6-only dynamic LCIA: `{ready.ready_for_b6_dynamic_lcia}`
- Sanity checks passed: `{int((sanity["status"] == "pass").sum())}` / `{len(sanity)}`
- Most robust outputs for the next step are the B6-only dynamic GHG comparison and the annual RF trajectory comparison because they run directly on the authoritative synchronized B6 wrappers and the corrected exchange table.

## Main Drivers

- GEN dynamic GHG top flow/source contributors:
{bullet_lines(gen_top_ghg)}
- BAU dynamic GHG top flow/source contributors:
{bullet_lines(bau_top_ghg)}
- GEN RF top biosphere flows:
{bullet_lines(gen_top_rf)}
- BAU RF top biosphere flows:
{bullet_lines(bau_top_rf)}

## Readiness For A/B/C Combination

- B6-only dynamic LCIA is now ready as a standalone stage.
- The B6-only run used the authoritative HEATNETS denominator consistently and did not rely on the broken historical `output_elec.csv`.
- The next combined A/B/C + B6 step can proceed from the synchronized foreground set, but it should use `GEN_TOTAL_with_B6` and `BAU_TOTAL_with_B6` only after confirming the desired combined-stage reporting basis.

## Quick Result Snapshot

- GEN total dynamic GWP100 over the modeled horizon, case-scaled: `{float(gen.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()):,.3f} kg CO2e`
- BAU total dynamic GWP100 over the modeled horizon, case-scaled: `{float(bau.annual_ghg["annual_dynamic_GWP100_kgCO2e_case_total"].sum()):,.3f} kg CO2e`
- GEN final cumulative dynamic GWP100, case-scaled: `{float(ghg_compare["cumulative_dynamic_GWP100_kgCO2e_case_total_GEN"].iloc[-1]):,.3f} kg CO2e`
- BAU final cumulative dynamic GWP100, case-scaled: `{float(ghg_compare["cumulative_dynamic_GWP100_kgCO2e_case_total_BAU"].iloc[-1]):,.3f} kg CO2e`
- GEN final cumulative annual RF sum, case-scaled: `{float(rf_compare["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN"].iloc[-1]):,.6g} W*yr/m^2`
- BAU final cumulative annual RF sum, case-scaled: `{float(rf_compare["cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU"].iloc[-1]):,.6g} W*yr/m^2`
"""
    write_markdown(OUTPUT_ROOT / "b6_dynamic_lcia_summary.md", summary)


def run(time_horizon_years: int = DEFAULT_HORIZON_YEARS) -> dict[str, Path]:
    ensure_workspace_tree()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    ready = _require_ready()
    context = _load_context()

    import bw2data as bd

    bd.projects.set_current(context.project)

    required_dbs = {context.gen_fg_db, context.bau_fg_db, *context.background_dbs}
    missing = [db_name for db_name in required_dbs if db_name not in bd.databases]
    if missing:
        raise RuntimeError(f"Missing required Brightway databases in project `{context.project}`: {missing}")

    for key in [context.gen_wrapper_key, context.bau_wrapper_key]:
        if key[0] not in bd.databases:
            raise RuntimeError(f"Wrapper database `{key[0]}` is missing in project `{context.project}`.")
        if not any(activity["code"] == key[1] for activity in bd.Database(key[0])):
            raise RuntimeError(
                f"Timed B6-only wrapper `{key}` is missing. Re-run the authoritative B6 inventory builder first."
            )

    method = _choose_climate_method(bd)

    gen = _run_case(
        label="GEN",
        fg_db=context.gen_fg_db,
        wrapper_key=context.gen_wrapper_key,
        context=context,
        method=method,
        t0=DEFAULT_T0,
        horizon_years=time_horizon_years,
    )
    bau = _run_case(
        label=_case_label("BAU"),
        fg_db=context.bau_fg_db,
        wrapper_key=context.bau_wrapper_key,
        context=context,
        method=method,
        t0=DEFAULT_T0,
        horizon_years=time_horizon_years,
    )

    ghg_compare, rf_compare = _comparison_tables(gen, bau)
    sanity = _sanity_checks(context)
    summary_rows = _summary_rows(
        context=context,
        method=method,
        ready=ready,
        sanity=sanity,
        gen=gen,
        bau=bau,
        horizon_years=time_horizon_years,
        t0=DEFAULT_T0,
    )

    outputs = {
        "annual_gen_ghg": OUTPUT_ROOT / "annual_B6_GHG_GEN.csv",
        "annual_bau_ghg": OUTPUT_ROOT / "annual_B6_GHG_BAU.csv",
        "annual_gen_rf": OUTPUT_ROOT / "annual_B6_RF_GEN.csv",
        "annual_bau_rf": OUTPUT_ROOT / "annual_B6_RF_BAU.csv",
        "cumulative_ghg_compare": OUTPUT_ROOT / "cumulative_B6_GHG_GEN_vs_BAU.csv",
        "cumulative_rf_compare": OUTPUT_ROOT / "cumulative_B6_RF_GEN_vs_BAU.csv",
        "top_ghg_flows_gen": OUTPUT_ROOT / "top_biosphere_flows_GHG_GEN.csv",
        "top_ghg_flows_bau": OUTPUT_ROOT / "top_biosphere_flows_GHG_BAU.csv",
        "top_rf_flows_gen": OUTPUT_ROOT / "top_biosphere_flows_RF_GEN.csv",
        "top_rf_flows_bau": OUTPUT_ROOT / "top_biosphere_flows_RF_BAU.csv",
        "summary_csv": OUTPUT_ROOT / "denominator_and_method_summary.csv",
        "sanity_csv": OUTPUT_ROOT / "sanity_checks.csv",
        "manifest_json": OUTPUT_ROOT / "b6_dynamic_lcia_manifest.json",
    }

    gen.annual_ghg.to_csv(outputs["annual_gen_ghg"], index=False)
    bau.annual_ghg.to_csv(outputs["annual_bau_ghg"], index=False)
    gen.annual_rf.to_csv(outputs["annual_gen_rf"], index=False)
    bau.annual_rf.to_csv(outputs["annual_bau_rf"], index=False)
    ghg_compare.to_csv(outputs["cumulative_ghg_compare"], index=False)
    rf_compare.to_csv(outputs["cumulative_rf_compare"], index=False)
    gen.top_gwp_flows.to_csv(outputs["top_ghg_flows_gen"], index=False)
    bau.top_gwp_flows.to_csv(outputs["top_ghg_flows_bau"], index=False)
    gen.top_rf_flows.to_csv(outputs["top_rf_flows_gen"], index=False)
    bau.top_rf_flows.to_csv(outputs["top_rf_flows_bau"], index=False)
    summary_rows.to_csv(outputs["summary_csv"], index=False)
    sanity.to_csv(outputs["sanity_csv"], index=False)

    gen.inv_df.to_csv(DETAIL_ROOT / "GEN_dynamic_inventory.csv", index=False)
    bau.inv_df.to_csv(DETAIL_ROOT / "BAU_dynamic_inventory.csv", index=False)
    gen.gwp_df.to_csv(DETAIL_ROOT / "GEN_dynamic_GWP_detailed.csv", index=False)
    bau.gwp_df.to_csv(DETAIL_ROOT / "BAU_dynamic_GWP_detailed.csv", index=False)
    gen.rf_df.to_csv(DETAIL_ROOT / "GEN_dynamic_RF_detailed.csv", index=False)
    bau.rf_df.to_csv(DETAIL_ROOT / "BAU_dynamic_RF_detailed.csv", index=False)
    gen.climate_summary.to_csv(DETAIL_ROOT / "GEN_climate_summary.csv", index=False)
    bau.climate_summary.to_csv(DETAIL_ROOT / "BAU_climate_summary.csv", index=False)
    gen.top_gwp_sources.to_csv(DETAIL_ROOT / "GEN_top_GHG_sources.csv", index=False)
    bau.top_gwp_sources.to_csv(DETAIL_ROOT / "BAU_top_GHG_sources.csv", index=False)

    _plot_compare_lines(
        ghg_compare,
        gen_col="annual_dynamic_GWP100_kgCO2e_case_total_GEN",
        bau_col="annual_dynamic_GWP100_kgCO2e_case_total_BAU",
        ylabel="Dynamic GWP100 (kg CO2e per case-year)",
        title=f"Annual Stage B6 dynamic GHG: GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "annual_B6_GHG_GEN_vs_BAU",
    )
    _plot_compare_lines(
        ghg_compare,
        gen_col="cumulative_dynamic_GWP100_kgCO2e_case_total_GEN",
        bau_col="cumulative_dynamic_GWP100_kgCO2e_case_total_BAU",
        ylabel="Cumulative dynamic GWP100 (kg CO2e per case)",
        title=f"Cumulative Stage B6 dynamic GHG: GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "cumulative_B6_GHG_GEN_vs_BAU",
    )
    _plot_compare_lines(
        rf_compare,
        gen_col="annual_radiative_forcing_W_per_m2_case_total_GEN",
        bau_col="annual_radiative_forcing_W_per_m2_case_total_BAU",
        ylabel="Annual radiative forcing (W/m² per case-year)",
        title=f"Annual Stage B6 radiative forcing: GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "annual_B6_RF_GEN_vs_BAU",
    )
    _plot_compare_lines(
        rf_compare,
        gen_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_GEN",
        bau_col="cumulative_discrete_annual_rf_Wyr_per_m2_case_total_BAU",
        ylabel="Cumulative annual RF sum (W*yr/m² per case)",
        title=f"Cumulative Stage B6 radiative forcing: GEN vs REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "cumulative_B6_RF_GEN_vs_BAU",
    )

    _plot_stacked_bars(
        gen.top_gwp_sources,
        category_col="flow_source_label",
        value_col="dynamic_GWP100_kgCO2e_case_total",
        ylabel="Dynamic GWP100 contribution (kg CO2e per case-year)",
        title=f"Stage B6 annual GHG contributors, GEN\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "stacked_annual_B6_GHG_contributors_GEN",
    )
    _plot_stacked_bars(
        bau.top_gwp_sources,
        category_col="flow_source_label",
        value_col="dynamic_GWP100_kgCO2e_case_total",
        ylabel="Dynamic GWP100 contribution (kg CO2e per case-year)",
        title=f"Stage B6 annual GHG contributors, REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "stacked_annual_B6_GHG_contributors_BAU",
    )
    _plot_top_flow_lines(
        gen.top_rf_flows,
        flow_col="flow_label",
        value_col="radiative_forcing_W_per_m2_case_total",
        ylabel="Radiative forcing contribution (W/m² per case-year)",
        title=f"Stage B6 top RF-contributing biosphere flows, GEN\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "top_B6_RF_biosphere_flows_GEN",
    )
    _plot_top_flow_lines(
        bau.top_rf_flows,
        flow_col="flow_label",
        value_col="radiative_forcing_W_per_m2_case_total",
        ylabel="Radiative forcing contribution (W/m² per case-year)",
        title=f"Stage B6 top RF-contributing biosphere flows, REF\n{_title_suffix(context)}",
        basepath=FIGURE_ROOT / "top_B6_RF_biosphere_flows_BAU",
    )

    _write_summary_markdown(
        context=context,
        ready=ready,
        sanity=sanity,
        gen=gen,
        bau=bau,
        ghg_compare=ghg_compare,
        rf_compare=rf_compare,
    )

    manifest = {
        "project": context.project,
        "normalization_mode": context.normalization_mode,
        "foreground_set": context.foreground_set,
        "time_horizon_years": time_horizon_years,
        "time_horizon_start": DEFAULT_T0.strftime("%Y-%m-%d"),
        "gen_wrapper_key": list(context.gen_wrapper_key),
        "bau_wrapper_key": list(context.bau_wrapper_key),
        "gen_b6_key": list(context.gen_b6_key),
        "bau_b6_key": list(context.bau_b6_key),
        "output_root": relpath(OUTPUT_ROOT),
        "figures_root": relpath(FIGURE_ROOT),
        "details_root": relpath(DETAIL_ROOT),
        "generated_files": {name: relpath(path) for name, path in outputs.items()},
    }
    write_json(outputs["manifest_json"], manifest)
    return outputs


def refresh_sanity_only() -> dict[str, Path]:
    ensure_workspace_tree()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    context = _load_context()
    sanity = _sanity_checks(context)
    output = OUTPUT_ROOT / "sanity_checks.csv"
    sanity.to_csv(output, index=False)
    return {"sanity_csv": output}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage B6-only dynamic LCIA for GEN and BAU on the HEATNETS-authoritative synchronized foregrounds."
    )
    parser.add_argument(
        "--time-horizon-years",
        type=int,
        default=DEFAULT_HORIZON_YEARS,
        help="Dynamic characterization horizon in years. Defaults to 100.",
    )
    parser.add_argument(
        "--refresh-sanity-only",
        action="store_true",
        help="Rebuild only the B6 sanity CSV from current local tables without rerunning Timex.",
    )
    args = parser.parse_args()
    outputs = refresh_sanity_only() if args.refresh_sanity_only else run(time_horizon_years=args.time_horizon_years)
    for name, path in outputs.items():
        print(f"{name}: {relpath(path)}")


if __name__ == "__main__":
    main()
