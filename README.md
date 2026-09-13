# Support-conditioned prediction combination for few-shot battery SOH estimation

This repository contains the LaTeX source, figures, tables, final PDFs and
portable frozen evidence for the manuscript:

> Support-conditioned prediction combination for few-shot battery state-of-health estimation under distribution shift

## Contents

- `main.tex`, `bibliography.tex`: main manuscript source.
- `supplementary.tex`: supplementary-material source.
- `main.pdf`, `supplementary.pdf`: compiled manuscript and supplement.
- `figures_*`, `tables`, `tables_revision`: source assets used by the two documents.
- `evidence/`: public, derived result records and scripts for checking reported
  values and regenerating selected supplementary figures.
- `reproduction/`: the M1--M4 research implementation, protocol files, tests,
  data-preparation utilities and deployment scripts recovered from the paper's
  experiment environment.

The original battery measurements, trained model checkpoints and private
research-tree files are not redistributed. The datasets must be obtained from
their original providers, as described in the Data availability section of the
manuscript.

## Compile the papers

Use a local TeX distribution with the packages listed in `README.txt`, then
run from the repository root:

```text
compile.bat
```

Alternatively, compile `main.tex` and `supplementary.tex` separately with
pdfLaTeX. The documents do not require Python or model training to compile.

## Check frozen evidence

The files in `evidence/` are supplied result records rather than raw battery
data. They allow numerical consistency checks and selected figure regeneration:

```text
python evidence/verify_public_evidence.py
python evidence/regenerate_figures.py
```

The first command checks the public aggregate, ablation, interval and
trajectory records against the values used by the manuscript tables. The
second command regenerates the four supplementary diagnostic figures from the
supplied frozen records. It does not train models or claim to reproduce the
original private training run.

## Reproducibility scope

This release supports document compilation, table-value auditing and selected
figure regeneration. The recovered implementation is included, but full
end-to-end retraining still requires the original battery datasets and local
cache/result directories, which are not redistributed. The scripts retain the
original relative-directory assumptions; see `reproduction/README.md` before
running them.
