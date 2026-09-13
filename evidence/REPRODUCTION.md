# Evidence and reproduction guide

This package compiles the paper without Python or access to the original research tree. Numerical evidence and optional plotting tools are separate from LaTeX compilation.

## Numerical checks

Run `python evidence/verify_evidence.py` from the source directory. Only the Python standard library is required. The script checks supplied source hashes, M2/M3/M4 identity records, aggregate metrics and displayed derived quantities. It neither trains a model nor replays model inference or bootstrap sampling.

## Figure regeneration

Install Python with NumPy, pandas and Matplotlib. Run:

```
python evidence/regenerate_figures.py
```

The default output is `evidence/regenerated_figures/`. To deliberately replace the four manuscript figure files, use `--output figures_revision`. Each figure is saved as PDF and PNG. The saved curves are plotted separately for each source model, without averaging them into an ensemble. PDF bytes can differ across software versions; the numerical inputs are protected by SHA-256 in `provenance.json`.

| Supplementary figure | Output basename | Frozen inputs |
| --- | --- | --- |
| S1, all 21 conditions | domain_effects | domain_ablation.csv; reductions are parent minus child MAE |
| S2, CALCE protocol groups | calce_protocols | calce_protocols.csv; changes are B1234 minus B MAE |
| S3, seven-cell trajectories | calce_trajectories | trajectory_arrays/folds/*/predictions/*/{B,B1234}.npz |
| S4, accuracy and latency | stage_cost | stage_cost.csv |

There are 84 supplied saved-array files: two stages for six source folds and seven cells. They contain saved query outputs and the matching query observations used for plotting, not the original measurement dataset or model checkpoints. The 42-array range diagnostic is computed separately for each stage. The archived research-tree audit is not part of this public release; the portable figure script above is the intended entry point.

## Table source mapping

All CSV/JSON values remain unrounded. Round only after computing a displayed difference or percentage; SOH-ratio records require multiplication by 100 to obtain SOH percentage points.

| LaTeX source | Input records in evidence/ |
| --- | --- |
| s_m1.tex | m1_screen.json |
| s_m2.tex | m2_fixed_source.json (Base+M1+M2, source_lodo_fixed_extended); m2_v1.json, m2_v2.json, m2_v3.json |
| s_m3_all.tex, main M3 matched table | m3_matched_original.csv, m3_controls.csv |
| s_m3_selected.tex | m3_selection_original.csv |
| s_effects.tex | aggregate.csv, module_effects.csv |
| branch_intervals.tex | paired_intervals.csv |
| s_initialization.tex | m4_initializations.csv, m4_initialization_summary.json |
| s_neural_audit.tex | neural_checkpoint_audit.json, neural_metric_audit.json |
| s_ranges.tex | calce_ranges.csv, calce_210_trajectories.csv, supplied trajectory arrays |
| s_cost.tex | stage_cost.csv, aggregate.csv |
| calce_protocols.tex / combined calce_main.tex | calce_protocols.csv, calce_protocols_original.csv |
| evidence index | provenance.json |

The M3 candidate and release-acceptance hashes are checked against `m3_selection.json`. M2 remains the frozen v3 candidate. M4 remains the private-function candidate with global multiplier 0.5; fold-specific selected gains can be zero.

## Historical scope

`prior_asset_audit.md`, `prior_inventory.json` and `prior_scan_summary.json` describe a previously completed full-directory scan. This revision reuses those records; it does not repeat that scan. Historical reports are preserved verbatim, including their period-specific model names and rounded values. They do not supersede the current manuscript, and historical candidate scores must not be spliced into the final comparison.

The historical XJTU comparison is retained for coverage and development context. The current primary comparison covers 12 families across three datasets; the seven extra neural families have XJTU-only historical results. The 189/9,855/93 replay counts do not apply to all historical families.

## Compilation and reference checks

Compile each of `main.tex` and `supplementary.tex` repeatedly until references stabilize, or use `compile.bat`. The README lists TeX dependencies. Main-text references to supplementary tables and figures use explicit S numbers; after structural edits, check them against the labels in `supplementary.aux`. Both documents use `table_style.tex` for a common table font size. Compilation logs and the delivery manifest are included in the verification directory of the release.
