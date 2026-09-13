# v1.0.0 — First public reproduction release

This release accompanies the manuscript:

*Support-conditioned prediction combination for few-shot battery state-of-health estimation under distribution shift*

## Included

- Compilable main and supplementary LaTeX sources and PDFs.
- All manuscript table sources and released figure assets.
- Public frozen evidence records for the reported aggregates, ablations,
  intervals, costs and selected prediction trajectories.
- Public evidence verification and figure-regeneration scripts.
- Recovered M1--M4 research implementation, protocol files, tests and data
  preparation utilities.
- Python environment version information and reproduction guidance.

## Reproduction scope

The release supports manuscript compilation, frozen-result verification,
selected figure regeneration and inspection of the M1--M4 implementation.
The original battery datasets, trained checkpoints and local experiment caches
are not redistributed. Dataset access and reuse remain subject to the terms of
the original providers.

The implementation retains the relative-directory assumptions of the original
experiment environment. See `README.md` and `reproduction/README.md` before
running experiments.
