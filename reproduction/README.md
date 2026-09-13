# Reproduction source from the paper-2 experiment environment

This directory contains the M1--M4 research implementation, protocol files,
tests and deployment utilities used during development of the manuscript.
The directory layout follows the original experiment tree:

```text
reproduction/
  模块研发/{m1,m2,m3,m4,evaluation_v3}/
  实验部署/
  开发基线选择依据/
```

The implementation expects the public datasets to be placed under a sibling
`数据集/` directory and the generated caches/results under the corresponding
`实验部署/` and `模块研发/results/` directories. The datasets and generated
training caches are intentionally not included in this repository.

The scripts are the original research implementation rather than a newly
constructed toy reimplementation. Before running a full experiment, inspect
the protocol JSON files and set the working root to the local checkout. The
captured environment used for the reported runs is listed in
`实验部署/requirements.lock.txt`.

The public frozen records in the top-level `evidence/` directory provide a
portable way to check the reported aggregates and regenerate selected figures
without rebuilding the private training caches.
