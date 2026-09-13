# XJTU 跨工况开发选型（K=10）

这是六工况留出开发结果，不是 XJTU→HUST。表中只显示已覆盖全部六工况、55 个电芯的种子；n=1 的跨种子标准差不可估计。筛选（336/336）和领先随机竞争者确认（168/168）均已完成；严格轨道的开发冠军为 GPR / source-only，冠军结论及其适用范围见 `../../冠军选型与模块侦察结论_v1.md`。

| 信息轨道 | 方法 | 设置 | 完整种子数 | MAE（SOH百分点） | RMSE（SOH百分点） |
|---|---|---|---:|---:|---:|
| full_supervision | Full_target_LOCO_Ridge | other_target_cells_full_labels | 1 | 1.6589 (n=1) | 2.0695 (n=1) |
| online_cycle | MLP_cycle | full_finetune | 3 | 4.1897 ± 0.1226 | 5.2201 ± 0.3291 |
| online_cycle | PINN_cycle | full_finetune | 3 | 4.2400 ± 0.5180 | 5.0165 ± 0.5342 |
| online_cycle | PINN_cycle | source_bias | 3 | 4.3077 ± 0.3209 | 5.2606 ± 0.3458 |
| online_cycle | PINN_cycle | last_block | 3 | 4.3226 ± 0.9549 | 5.2061 ± 0.8160 |
| online_cycle | MLP_cycle | target_only | 3 | 5.0738 ± 0.2048 | 6.4999 ± 0.1613 |
| online_cycle | PINN_cycle | target_only | 3 | 5.7408 ± 0.1962 | 7.4087 ± 0.2795 |
| online_cycle | MLP_cycle | last_block | 3 | 6.0100 ± 0.6000 | 7.1966 ± 0.9174 |
| online_cycle | MLP_cycle | source_only | 3 | 6.2185 ± 0.3153 | 7.9690 ± 0.6286 |
| online_cycle | MLP_cycle | head_only | 3 | 6.3706 ± 0.7326 | 7.5914 ± 0.9860 |
| online_cycle | MLP_cycle | source_bias | 3 | 6.9226 ± 0.7904 | 8.2105 ± 1.0594 |
| online_cycle | PINN_cycle | head_only | 3 | 13.7962 ± 6.0511 | 14.5555 ± 5.9977 |
| online_cycle | PINN_cycle | source_only | 3 | 14.8372 ± 6.1493 | 15.6943 ± 6.1017 |
| online_cycle | Linear_trend | fixed_rule | 1 | 36.5196 (n=1) | 43.3105 (n=1) |
| online_cycle | Exponential_trend | fixed_rule | 1 | 51.0281 (n=1) | 63.0238 (n=1) |
| strict | GPR | source_only | 1 | 3.0115 (n=1) | 3.8271 (n=1) |
| strict | GPR | source_bias | 1 | 3.0129 (n=1) | 3.9536 (n=1) |
| strict | RandomForest | source_bias | 10 | 3.3594 ± 0.0284 | 4.2643 ± 0.0272 |
| strict | XGBoost | source_bias | 10 | 3.3884 ± 0.0243 | 4.2562 ± 0.0250 |
| strict | MAML_MLP | meta_adaptation | 10 | 3.6683 ± 0.1649 | 4.8075 ± 0.1641 |
| strict | PatchTSTLite | last_block | 10 | 3.6697 ± 0.3186 | 4.9787 ± 0.4846 |
| strict | TCN | full_finetune | 3 | 3.7192 ± 0.0339 | 4.9841 ± 0.0847 |
| strict | CNN1D | source_bias | 3 | 3.7529 ± 0.1506 | 4.9076 ± 0.2594 |
| strict | CNN1D | full_finetune | 3 | 3.7583 ± 0.0552 | 4.9372 ± 0.1491 |
| strict | TCN | last_block | 3 | 3.7714 ± 0.1346 | 5.1357 ± 0.2199 |
| strict | PatchTSTLite | source_bias | 10 | 3.7769 ± 0.3084 | 4.8656 ± 0.4039 |
| strict | Transformer | source_bias | 3 | 3.7984 ± 0.2469 | 4.9262 ± 0.2362 |
| strict | CNN1D | last_block | 3 | 3.8135 ± 0.1130 | 4.9572 ± 0.1976 |
| strict | SVR_RBF | source_bias | 1 | 3.8250 (n=1) | 4.8582 (n=1) |
| strict | Transformer | head_only | 3 | 3.8451 ± 0.2743 | 5.0281 ± 0.2303 |
| strict | PatchTSTLite | full_finetune | 10 | 3.8985 ± 0.3417 | 5.0883 ± 0.4506 |
| strict | TCN | source_bias | 3 | 3.9851 ± 0.1325 | 5.3220 ± 0.1447 |
| strict | MLP | full_finetune | 3 | 4.0224 ± 0.0845 | 5.0688 ± 0.0848 |
| strict | PLSR | source_only | 1 | 4.1068 (n=1) | 5.2606 (n=1) |
| strict | Transformer | last_block | 3 | 4.1142 ± 0.1375 | 5.3889 ± 0.1740 |
| strict | GRU | last_block | 3 | 4.1485 ± 0.3528 | 5.4869 ± 0.5585 |
| strict | GRU | source_bias | 3 | 4.1549 ± 0.2823 | 5.5527 ± 0.4328 |
| strict | Transformer | full_finetune | 3 | 4.1833 ± 0.3331 | 5.8281 ± 0.3766 |
| strict | GRU | full_finetune | 3 | 4.2176 ± 0.3687 | 5.5975 ± 0.2726 |
| strict | LSTM | full_finetune | 3 | 4.2803 ± 0.4271 | 5.9955 ± 0.2937 |
| strict | PatchTSTLite | target_only | 10 | 4.3237 ± 0.9533 | 5.9557 ± 1.0971 |
| strict | XGBoost | source_only | 10 | 4.4572 ± 0.0863 | 5.3137 ± 0.0882 |
| strict | RandomForest | source_only | 10 | 4.4933 ± 0.0176 | 5.3853 ± 0.0200 |
| strict | LSTM | head_only | 3 | 4.5200 ± 0.1919 | 6.0551 ± 0.2279 |
| strict | Dummy_mean | fixed_rule | 1 | 4.5413 (n=1) | 5.9337 (n=1) |
| strict | LSTM | source_bias | 3 | 4.5654 ± 0.2655 | 6.1145 ± 0.2855 |
| strict | MLP | target_only | 3 | 4.6646 ± 0.0817 | 6.2547 ± 0.0766 |
| strict | PatchTSTLite | head_only | 10 | 4.7121 ± 1.1622 | 5.9525 ± 1.1153 |
| strict | LSTM | last_block | 3 | 4.7144 ± 0.2826 | 6.3787 ± 0.3422 |
| strict | Transformer | source_only | 3 | 4.7248 ± 0.9061 | 5.9434 ± 0.8359 |
| strict | PatchTST_CI_SOH | source_bias | 3 | 4.7313 ± 0.1734 | 6.2065 ± 0.1830 |
| strict | PLSR | source_bias | 1 | 4.7458 (n=1) | 5.9740 (n=1) |
| strict | RandomForest | target_only | 10 | 4.7733 ± 0.0043 | 6.4885 ± 0.0043 |
| strict | PatchTST_CI_SOH | full_finetune | 3 | 4.8274 ± 0.1904 | 6.3503 ± 0.1342 |
| strict | PatchTST_CI_SOH | head_only | 3 | 4.8434 ± 0.2550 | 6.4283 ± 0.1724 |
| strict | XGBoost | target_only | 10 | 4.8550 ± 0.0092 | 6.4810 ± 0.0065 |
| strict | PatchTST_CI_SOH | last_block | 3 | 4.8788 ± 0.1364 | 6.4362 ± 0.1620 |
| strict | CNN1D | target_only | 3 | 4.8846 ± 0.1322 | 6.5406 ± 0.3229 |
| strict | SVR_RBF | target_only | 1 | 4.9009 (n=1) | 6.4653 (n=1) |
| strict | GPR | target_only | 1 | 4.9887 (n=1) | 6.5971 (n=1) |
| strict | Ridge | source_bias | 1 | 4.9948 (n=1) | 6.3090 (n=1) |
| strict | TCN | target_only | 3 | 5.0516 ± 0.2102 | 6.5391 ± 0.4611 |
| strict | Support_mean | fixed_rule | 1 | 5.0675 (n=1) | 6.6143 (n=1) |
| strict | GRU | target_only | 3 | 5.0892 ± 0.1381 | 6.5714 ± 0.1745 |
| strict | CNN_LSTM | target_only | 3 | 5.1012 ± 0.1671 | 6.6598 ± 0.1882 |
| strict | Transformer | target_only | 3 | 5.1295 ± 0.8338 | 6.9505 ± 0.7669 |
| strict | PatchTST_CI_SOH | source_only | 3 | 5.1326 ± 0.5363 | 6.6957 ± 0.6320 |
| strict | SVR_RBF | source_only | 1 | 5.1659 (n=1) | 6.6191 (n=1) |
| strict | MLP | last_block | 3 | 5.2060 ± 0.3319 | 6.4102 ± 0.5987 |
| strict | Last_support | fixed_rule | 1 | 5.2319 (n=1) | 7.0292 (n=1) |
| strict | MLP | source_bias | 3 | 5.3319 ± 1.3757 | 6.3328 ± 1.5528 |
| strict | GRU | head_only | 3 | 5.3409 ± 0.8927 | 6.6686 ± 1.0821 |
| strict | PatchTST_CI_SOH | target_only | 3 | 5.4147 ± 0.3916 | 6.8518 ± 0.3631 |
| strict | LSTM | target_only | 3 | 5.4241 ± 0.3249 | 6.8356 ± 0.2801 |
| strict | LSTM | source_only | 3 | 5.6487 ± 1.4164 | 7.0896 ± 1.2632 |
| strict | CNN1D | head_only | 3 | 5.6913 ± 1.8828 | 6.6870 ± 1.7678 |
| strict | TCN | head_only | 3 | 6.3806 ± 2.5443 | 7.5789 ± 2.3675 |
| strict | GRU | source_only | 3 | 6.9439 ± 1.2515 | 8.2976 ± 1.3681 |
| strict | PatchTSTLite | source_only | 10 | 6.9949 ± 1.2919 | 8.0896 ± 1.2895 |
| strict | Ridge | source_only | 1 | 7.0950 (n=1) | 8.1990 (n=1) |
| strict | PLSR | target_only | 1 | 7.4484 (n=1) | 9.8162 (n=1) |
| strict | MLP | head_only | 3 | 7.7017 ± 0.1849 | 8.5458 ± 0.3069 |
| strict | Ridge | target_only | 1 | 7.9519 (n=1) | 10.2231 (n=1) |
| strict | MLP | source_only | 3 | 8.5333 ± 0.1292 | 9.5884 ± 0.1563 |
| strict | CNN_LSTM | source_bias | 3 | 8.6714 ± 4.9902 | 11.0478 ± 5.8265 |
| strict | CNN_LSTM | full_finetune | 3 | 13.8847 ± 8.4518 | 15.6021 ± 8.7282 |
| strict | CNN_LSTM | head_only | 3 | 15.3120 ± 9.8190 | 17.1251 ± 9.7090 |
| strict | CNN_LSTM | last_block | 3 | 15.3756 ± 10.3366 | 17.3147 ± 10.2237 |
| strict | CNN1D | source_only | 3 | 16.0149 ± 4.1048 | 17.1097 ± 4.0403 |
| strict | TCN | source_only | 3 | 18.2163 ± 4.9641 | 19.3564 ± 5.0672 |
| strict | CNN_LSTM | source_only | 3 | 18.5648 ± 13.3724 | 20.2885 ± 13.3702 |
| uda | MMD_MLP | transductive_no_target_labels | 3 | 3.4046 ± 0.2621 | 4.5231 ± 0.3358 |
| uda | DeepCORAL_MLP | transductive_no_target_labels | 3 | 7.7059 ± 1.9891 | 8.6785 ± 2.0525 |
| uda | DANN_MLP | transductive_no_target_labels | 3 | 14.7258 ± 15.6558 | 15.9035 ± 15.2603 |
