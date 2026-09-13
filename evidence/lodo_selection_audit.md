# Leave-one-dataset-out selection-level audit

The audit evaluates 35 frozen candidates (seven architecture families by five global steps). In each round, one dataset is excluded from candidate selection and the other two development datasets determine the selected candidate and global step.

| Held-out dataset | Selection result | LODO MAE | Base GPR MAE | Relative reduction |
|---|---|---:|---:|---:|
| XJTU | plain / 1.0 | 2.6855 | 3.0115 | 10.83% |
| MATR | condition_mixed / 1.0 | 2.0178 | 2.6862 | 24.89% |
| Tongji | plain / 0.5 | 2.9666 | 6.6695 | 55.52% |

The relative advantage remains in all three held-out datasets. This is a frozen-candidate selection-level audit rather than complete nested cross-validation: every architecture is not retrained from raw data inside a fully nested loop.
