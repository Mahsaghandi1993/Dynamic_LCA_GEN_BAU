# Data availability and restrictions

Data are managed under the Data Management Plan of U.S. NSF Award No. 2502121 and
the NSF Public Access Policy. This file states exactly what is shared, what cannot
be, and why.

## Openly shared (this repository and the Zenodo archive)

- **Pipeline code** (`scripts/`) — BSD 3-Clause.
- **Public configuration** (`config/`) — case parameters, fuel shares, archetype
  mapping, robustness settings. Nothing here is licensed or confidential.
- **Derived aggregate results** (`data/derived/`) — the final tables behind every
  figure and table in the manuscript, including the full aggregated LCI
  (`full_detailed_lci_table_for_SI_REF_final.csv`).
- **Figures** (`figures/`) and the scripts that regenerate them.

## Not shareable

| Item | Reason |
|---|---|
| ecoinvent 3.9.1 (cutoff) database | commercial license (https://ecoinvent.org) |
| premise background databases (SSP2 snapshots) | derived from ecoinvent; same license applies |
| Raw Eversource/Framingham building, load, and operational files | third-party confidentiality agreement |
| Local Brightway project directories | embed licensed ecoinvent data |

The aggregated derivatives in `data/derived/` were reviewed to contain no
building-level or customer-identifiable information. Researchers with their own
ecoinvent license can re-run the entire pipeline (see `docs/REPRODUCING.md`);
questions about the restricted case data can be directed to the corresponding
author (Mahsa_Ghandi@student.uml.edu) and will be handled within the terms of the
confidentiality agreement.
