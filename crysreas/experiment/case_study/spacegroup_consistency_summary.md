# Space-group consistency summary

- Source parquet: `/home/wyy603/Projects/crysreas/checkpoints_merged/thinking/conditional+thinking.parquet`
- Target final-CIF space groups: `Fm-3m, Fd-3m, P3m1`
- Site match: percentage of structures whose predicted atoms-per-site pattern exactly matches the final CIF equivalent-site pattern.
- Volume match: percentage of structures whose predicted volume is within 1% of the final CIF volume.
- Spacegroup match: percentage of structures whose predicted space-group number matches the final CIF space-group number.
- Bond length match: micro-average over extracted X-Y bond-length claims; a pair matches when the predicted mean is within 0.05 A of the final CIF first-shell mean.

| space group | n structures | site match | volume match | spacegroup match | bond length match | bond pairs |
| --- | --- | --- | --- | --- | --- | --- |
| Fm-3m | 2437 | 99.75% (2431/2437) | 88.39% (2154/2437) | 99.63% (2428/2437) | 85.80% (4768/5557) | 5557 |
| Fd-3m | 67 | 100.00% (67/67) | 37.31% (25/67) | 100.00% (67/67) | 64.75% (90/139) | 139 |
| P3m1 | 325 | 81.85% (266/325) | 18.77% (61/325) | 82.77% (269/325) | 27.62% (221/800) | 800 |
| overall | 2829 | 97.70% (2764/2829) | 79.18% (2240/2829) | 97.70% (2764/2829) | 78.19% (5079/6496) | 6496 |
