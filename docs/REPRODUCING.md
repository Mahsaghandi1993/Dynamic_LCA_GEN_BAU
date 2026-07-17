# Reproducing the results, step by step

Two tiers of reproduction are supported. **Tier 1** needs only Python and this
repository. **Tier 2** re-runs the full LCA and additionally requires licensed and
confidential inputs that we cannot redistribute (see `docs/DATA.md`).

---

## Tier 1 — regenerate all manuscript figures and check every reported number

Requires: Python ≥ 3.10, `pandas`, `numpy`, `matplotlib`.

```bash
pip install pandas numpy matplotlib
python figures/scripts/make_results_figures.py      # Figs. 5, 6a, 6b, 7, 8, 9, 10
python figures/scripts/make_case_figures.py         # Figs. 3, 4
python figures/scripts/make_si_borefield_figure.py  # Supplementary Fig. S4
python figures/scripts/make_graphical_abstract.py   # graphical abstract
```

Every input the figure scripts read is a CSV in `data/derived/`, so you can also
verify each number in the manuscript directly against these tables:

| Manuscript item | File in `data/derived/` |
|---|---|
| Table 2 (headline, 3 methods) | `T1_headline_metrics.csv` |
| Table 3 (stage contributions) | `T2_stage_dynamic_contributions.csv`, `stage_[A|B|C]_method_comparison.csv` |
| Table 5 / Fig. 8 (sensitivity S1–S7) | `T3_sensitivity_scenario_matrix.csv`, `s1_s8_brightway_scenario_results.csv` |
| Table 6 / Fig. 9 (paired Monte Carlo) | `mc_draws_three_method.csv`, `mc_summary_three_method.csv` |
| Fig. 7 (trajectories, both grids) | `database_trajectory_comparison.csv`, `database_trajectory_comparison_RF.csv` |
| Fig. 10 (non-climate categories) | `total_method_comparison_[GEN|BAU].csv` |
| Stage B6 operational inventory | `b6_operational_exchange_table.csv`, `b6_denominator_summary.csv` |
| Full LCI (Supplementary S3) | `full_detailed_lci_table_for_SI_REF_final.csv` |

Note: `BAU` in CSV column names ≡ `REF` in the manuscript.

---

## Tier 2 — re-run the full LCA pipeline

### Prerequisites

1. Python ≥ 3.10 with the Brightway 2.5 ecosystem:
   `pip install brightway25 bw_temporalis bw_timex premise pyyaml scipy`
2. A **licensed ecoinvent 3.9.1 (cutoff)** database (https://ecoinvent.org).
3. `premise` prospective backgrounds generated from it (REMIND SSP2-PkBudg1000 and
   SSP2-NPi; snapshot years 2025, 2040, 2045, 2050, 2055).
4. The Framingham case operational data (confidential; see `docs/DATA.md`). Without
   it, the pipeline runs on the public configuration in `config/` but will not
   reproduce the case-specific Stage B6 loads exactly.

### Pipeline (run from the repository root)

| Step | Command | What it does |
|---|---|---|
| 1 | `python -m scripts.b6.setup_fresh_brightway_project` | creates the Brightway project and imports ecoinvent 3.9.1 |
| 2 | `python -m scripts.b6.ensure_premise_backgrounds` | generates/verifies the premise SSP2 background snapshots |
| 3 | `python -m scripts.b6.build_bw_timex_b6_inventory` | builds the calendar-dated Stage B6 operational inventories (GEN and REF) |
| 4 | `python -m scripts.b6.build_corrected_non_b6_foregrounds` | builds Stage A1–A5, B2–B4, C1–C4 foregrounds for both systems |
| 5 | `python -m scripts.b6.run_conventional_static_lca` | conventional static LCIA (frozen 2025 background) |
| 6 | `python -m scripts.b6.run_full_dynamic_lcia` | time-explicit relinking + dynamic GWP100 / radiative forcing |
| 7 | `python -m scripts.b6.run_method_comparison` | three-method comparison tables incl. non-climate categories |
| 8 | `python -m scripts.b6.run_ref_final_s1_s8_brightway` | deterministic sensitivity cases (S1–S7 of the manuscript) |
| 9 | `python -m scripts.b6.run_three_method_MC_2026_06_12` | 10,000-draw three-method paired Monte Carlo |

Outputs are written under `export/` (created at run time). Steps 5–9 each finish in
minutes; step 6 (full dynamic LCIA) is the long one.

### Key modeling choices (fixed in `config/`)

- Functional unit: 89.375 GWh_th delivered heating + cooling + DHW, 37 buildings, 50 yr.
- REF fuel split 70 % gas / 30 % oil (`fuel_share_framingham_acs.yaml`, ACS 2020 B25040).
- Fugitive CH4 leakage 2.04 % of delivered gas (`b6_case_config.yaml`).
- Replacement schedule: GEN pumps yr 15/30/45, GEN heat pumps yr 25, REF equipment yr 20/40.
