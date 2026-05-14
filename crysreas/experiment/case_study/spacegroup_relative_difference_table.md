# Space-group relative difference table

- Source parquet: `/home/wyy603/Projects/crysreas/checkpoints_merged/thinking/conditional+thinking.parquet`
- Target final-CIF space groups: `Fm-3m, Fd-3m, P3m1`
- Site match: percentage of structures whose predicted atoms-per-site pattern exactly matches the final CIF equivalent-site pattern.
- Volume rel. diff.: average `abs(predicted - final) / final`, reported as percent.
- Spacegroup match: percentage of structures whose predicted space-group number matches the final CIF space-group number.
- Bond rel. diff.: micro-average over extracted X-Y bond-length claims using `abs(predicted mean - final mean) / final mean`, reported as percent.
- All structures: every row in the parquet that can be parsed and evaluated, not only the target space groups.

| space group | n structures | site match | volume rel. diff. | spacegroup match | bond rel. diff. | bond pairs |
| --- | --- | --- | --- | --- | --- | --- |
| Fm-3m | 2437 | 99.75% (2431/2437) | 0.71% (n=2437) | 99.63% (2428/2437) | 1.08% (n=5555) | 5557 |
| Fd-3m | 67 | 100.00% (67/67) | 2.46% (n=67) | 100.00% (67/67) | 2.02% (n=139) | 139 |
| P3m1 | 325 | 81.85% (266/325) | 3.81% (n=325) | 82.77% (269/325) | 21.91% (n=737) | 800 |
| All structures | 16382 | 75.19% (12317/16382) | 5.05% (n=16382) | 73.44% (12031/16382) | 23.25% (n=40259) | 40388 |

## LaTeX table

```latex
\begin{table}[t]
\centering
\caption{Thinking-trace consistency by final-CIF space group. Site and space-group columns report match percentages; volume and bond columns report average relative differences.}
\label{tab:thinking_trace_relative_difference}
\begin{tabular}{lrrrrrr}
\toprule
Space group & $N$ & Site match (\%) & Volume rel. diff. (\%) & SG match (\%) & Bond rel. diff. (\%) & Bond pairs \\
\midrule
Fm-3m & 2437 & 99.75 & 0.71 & 99.63 & 1.08 & 5557 \\
Fd-3m & 67 & 100.00 & 2.46 & 100.00 & 2.02 & 139 \\
P3m1 & 325 & 81.85 & 3.81 & 82.77 & 21.91 & 800 \\
All structures & 16382 & 75.19 & 5.05 & 73.44 & 23.25 & 40388 \\
\bottomrule
\end{tabular}
\end{table}
```
