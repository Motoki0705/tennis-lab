# clip_000 heatmap復号の指標

値は median px / p90 px / >20px率。coverageは採点数/対象数。

## 主評価 [505,1010)

| 方式 | LOVO | coverage | 全view median px | 有効3D率 |
|---|---:|---:|---:|---:|
| annotation_oracle | 7.859 / 20.722 / 11.15% | 1345/1462 (92.00%) | 2.955 | 98.81% |
| axis_logit | 9.197 / 27.346 / 16.59% | 826/1462 (56.50%) | 4.638 | 82.97% |
| axis_logit_raw | 10.016 / 29.957 / 18.41% | 793/1462 (54.24%) | 5.415 | 78.81% |
| centroid_3 | 8.884 / 26.033 / 15.62% | 826/1462 (56.50%) | 4.643 | 83.76% |
| centroid_3_raw | 9.925 / 27.332 / 18.18% | 759/1462 (51.92%) | 5.838 | 79.21% |
| centroid_5 | 9.344 / 26.658 / 15.64% | 831/1462 (56.84%) | 4.156 | 83.96% |
| centroid_5_raw | 9.499 / 27.694 / 16.15% | 811/1462 (55.47%) | 4.473 | 81.58% |
| gaussian_5 | 10.196 / 26.376 / 16.65% | 829/1462 (56.70%) | 4.129 | 82.57% |
| gaussian_5_raw | 10.363 / 29.437 / 17.44% | 797/1462 (54.51%) | 4.483 | 79.80% |
| legacy_affine | 9.615 / 27.867 / 18.69% | 824/1462 (56.36%) | 4.296 | 82.38% |
| legacy_affine_raw | 10.015 / 31.693 / 19.87% | 785/1462 (53.69%) | 5.216 | 76.04% |
| legacy_redecoded | 9.340 / 27.078 / 16.36% | 825/1462 (56.43%) | 4.625 | 83.17% |
| legacy_redecoded_raw | 9.798 / 29.810 / 17.91% | 793/1462 (54.24%) | 5.321 | 79.01% |
| legacy_stride | 10.323 / 28.805 / 15.43% | 823/1462 (56.29%) | 5.822 | 83.17% |
| legacy_stride_raw | 10.639 / 29.823 / 17.82% | 797/1462 (54.51%) | 6.144 | 79.21% |
| pipeline_baseline | 9.340 / 27.078 / 16.36% | 825/1462 (56.43%) | 4.625 | 83.17% |
| quadratic_log_3 | 9.311 / 27.996 / 16.00% | 831/1462 (56.84%) | 4.482 | 83.76% |
| quadratic_log_3_raw | 9.538 / 28.736 / 16.98% | 807/1462 (55.20%) | 5.022 | 80.00% |
| quadratic_log_5 | 10.200 / 28.382 / 17.97% | 829/1462 (56.70%) | 4.080 | 81.98% |
| quadratic_log_5_raw | 10.346 / 30.563 / 18.26% | 805/1462 (55.06%) | 4.302 | 80.00% |
| quadratic_logit_3 | 9.230 / 28.232 / 15.92% | 829/1462 (56.70%) | 4.492 | 83.56% |
| quadratic_logit_3_raw | 9.555 / 28.148 / 16.85% | 807/1462 (55.20%) | 4.949 | 81.39% |
| quadratic_logit_5 | 10.152 / 30.028 / 17.45% | 831/1462 (56.84%) | 4.077 | 82.77% |
| quadratic_logit_5_raw | 10.340 / 32.716 / 18.81% | 776/1462 (53.08%) | 4.468 | 77.82% |
| selected_affine | 9.491 / 29.633 / 18.60% | 828/1462 (56.63%) | 4.074 | 83.37% |
| selected_affine_raw | 9.642 / 30.292 / 18.23% | 801/1462 (54.79%) | 4.701 | 78.42% |
| selected_stride | 10.146 / 28.697 / 14.54% | 832/1462 (56.91%) | 5.546 | 83.96% |
| selected_stride_raw | 10.377 / 29.461 / 16.54% | 810/1462 (55.40%) | 5.872 | 81.58% |
| softargmax_t0p5 | 9.532 / 25.959 / 15.85% | 833/1462 (56.98%) | 4.010 | 84.16% |
| softargmax_t0p5_raw | 9.487 / 26.521 / 16.00% | 825/1462 (56.43%) | 4.100 | 81.58% |
| softargmax_t2p0 | 9.384 / 25.952 / 15.92% | 829/1462 (56.70%) | 4.374 | 83.96% |
| softargmax_t2p0_raw | 9.718 / 27.217 / 17.35% | 778/1462 (53.21%) | 5.280 | 80.40% |
| without_loess | 9.798 / 29.810 / 17.91% | 793/1462 (54.24%) | 5.321 | 79.01% |

## 前半 [0,505)

| 方式 | LOVO | coverage |
|---|---:|---:|
| pipeline_baseline | 10.716 / 28.483 / 16.44% | 821/1497 (54.84%) |
| without_loess | 11.271 / 29.887 / 18.24% | 795/1497 (53.11%) |
| quadratic_logit_3 | 10.480 / 26.086 / 15.41% | 824/1497 (55.04%) |
| selected_affine | 10.128 / 27.425 / 14.96% | 822/1497 (54.91%) |
| selected_affine_raw | 10.707 / 27.292 / 16.52% | 793/1497 (52.97%) |
| legacy_stride | 11.247 / 28.766 / 19.11% | 827/1497 (55.24%) |
| legacy_affine | 10.529 / 28.697 / 16.32% | 821/1497 (54.84%) |
| annotation_oracle | 9.655 / 22.524 / 11.46% | 1422/1497 (94.99%) |

## 全区間 [0,1010)

| 方式 | LOVO | coverage |
|---|---:|---:|
| pipeline_baseline | 10.208 / 28.196 / 16.40% | 1646/2959 (55.63%) |
| without_loess | 10.496 / 29.870 / 18.07% | 1588/2959 (53.67%) |
| quadratic_logit_3 | 9.952 / 26.533 / 15.67% | 1653/2959 (55.86%) |
| selected_affine | 9.829 / 28.259 / 16.79% | 1650/2959 (55.76%) |
| selected_affine_raw | 10.109 / 28.087 / 17.38% | 1594/2959 (53.87%) |
| legacy_stride | 10.836 / 28.785 / 17.27% | 1650/2959 (55.76%) |
| legacy_affine | 10.017 / 28.443 / 17.51% | 1645/2959 (55.59%) |
| annotation_oracle | 8.790 / 21.392 / 11.31% | 2767/2959 (93.51%) |

## LOVO別・区間別

| 方式 | 除外camera | 区間 | median / p90 / >20px | 採点数/対象数 |
|---|---|---|---:|---:|
| pipeline_baseline | cam0 | evaluation | 7.920 / 14.235 / 4.06% | 271/473 |
| pipeline_baseline | cam0 | tuning | 10.498 / 18.153 / 8.78% | 262/497 |
| pipeline_baseline | cam0 | all | 9.708 / 16.645 / 6.38% | 533/970 |
| pipeline_baseline | cam1 | evaluation | 11.990 / 25.147 / 22.94% | 231/484 |
| pipeline_baseline | cam1 | tuning | 14.323 / 40.693 / 36.02% | 236/501 |
| pipeline_baseline | cam1 | all | 13.081 / 35.881 / 29.55% | 467/985 |
| pipeline_baseline | cam2 | evaluation | 9.262 / 40.718 / 21.98% | 323/505 |
| pipeline_baseline | cam2 | tuning | 9.657 / 19.111 / 8.36% | 323/499 |
| pipeline_baseline | cam2 | all | 9.524 / 29.923 / 15.17% | 646/1004 |
| without_loess | cam0 | evaluation | 8.860 / 14.254 / 4.63% | 259/473 |
| without_loess | cam0 | tuning | 10.471 / 21.952 / 11.67% | 257/497 |
| without_loess | cam0 | all | 9.856 / 18.528 / 8.14% | 516/970 |
| without_loess | cam1 | evaluation | 11.748 / 28.661 / 23.85% | 218/484 |
| without_loess | cam1 | tuning | 14.600 / 42.379 / 36.82% | 220/501 |
| without_loess | cam1 | all | 12.943 / 38.707 / 30.37% | 438/985 |
| without_loess | cam2 | evaluation | 9.495 / 41.456 / 24.68% | 316/505 |
| without_loess | cam2 | tuning | 10.060 / 21.388 / 10.69% | 318/499 |
| without_loess | cam2 | all | 9.884 / 30.687 / 17.67% | 634/1004 |
| quadratic_logit_3 | cam0 | evaluation | 8.373 / 13.119 / 4.40% | 273/473 |
| quadratic_logit_3 | cam0 | tuning | 9.567 / 18.565 / 9.40% | 266/497 |
| quadratic_logit_3 | cam0 | all | 9.012 / 16.135 / 6.86% | 539/970 |
| quadratic_logit_3 | cam1 | evaluation | 10.583 / 24.374 / 21.12% | 232/484 |
| quadratic_logit_3 | cam1 | tuning | 13.532 / 38.333 / 32.77% | 235/501 |
| quadratic_logit_3 | cam1 | all | 11.999 / 32.386 / 26.98% | 467/985 |
| quadratic_logit_3 | cam2 | evaluation | 9.076 / 40.160 / 21.91% | 324/505 |
| quadratic_logit_3 | cam2 | tuning | 9.994 / 19.301 / 7.74% | 323/499 |
| quadratic_logit_3 | cam2 | all | 9.744 / 29.708 / 14.84% | 647/1004 |
| selected_affine | cam0 | evaluation | 8.051 / 12.986 / 4.04% | 272/473 |
| selected_affine | cam0 | tuning | 9.139 / 16.937 / 6.06% | 264/497 |
| selected_affine | cam0 | all | 8.677 / 15.454 / 5.04% | 536/970 |
| selected_affine | cam1 | evaluation | 12.352 / 27.057 / 31.03% | 232/484 |
| selected_affine | cam1 | tuning | 15.436 / 37.763 / 35.74% | 235/501 |
| selected_affine | cam1 | all | 13.998 / 33.143 / 33.40% | 467/985 |
| selected_affine | cam2 | evaluation | 8.311 / 43.290 / 21.91% | 324/505 |
| selected_affine | cam2 | tuning | 9.234 / 18.300 / 7.12% | 323/499 |
| selected_affine | cam2 | all | 9.030 / 31.382 / 14.53% | 647/1004 |
| selected_affine_raw | cam0 | evaluation | 8.239 / 13.551 / 4.60% | 261/473 |
| selected_affine_raw | cam0 | tuning | 9.652 / 18.248 / 9.34% | 257/497 |
| selected_affine_raw | cam0 | all | 9.032 / 16.391 / 6.95% | 518/970 |
| selected_affine_raw | cam1 | evaluation | 13.132 / 28.753 / 26.70% | 221/484 |
| selected_affine_raw | cam1 | tuning | 15.397 / 40.240 / 33.79% | 219/501 |
| selected_affine_raw | cam1 | all | 14.380 / 34.837 / 30.23% | 440/985 |
| selected_affine_raw | cam2 | evaluation | 8.819 / 47.823 / 23.51% | 319/505 |
| selected_affine_raw | cam2 | tuning | 9.481 / 20.674 / 10.41% | 317/499 |
| selected_affine_raw | cam2 | all | 9.271 / 31.746 / 16.98% | 636/1004 |
| legacy_stride | cam0 | evaluation | 8.748 / 15.530 / 4.85% | 268/473 |
| legacy_stride | cam0 | tuning | 11.585 / 20.028 / 10.07% | 268/497 |
| legacy_stride | cam0 | all | 10.736 / 17.945 / 7.46% | 536/970 |
| legacy_stride | cam1 | evaluation | 11.523 / 21.793 / 13.85% | 231/484 |
| legacy_stride | cam1 | tuning | 12.675 / 39.981 / 35.59% | 236/501 |
| legacy_stride | cam1 | all | 12.088 / 36.259 / 24.84% | 467/985 |
| legacy_stride | cam2 | evaluation | 10.304 / 39.727 / 25.31% | 324/505 |
| legacy_stride | cam2 | tuning | 10.520 / 22.223 / 14.55% | 323/499 |
| legacy_stride | cam2 | all | 10.446 / 30.041 / 19.94% | 647/1004 |
| legacy_affine | cam0 | evaluation | 7.960 / 14.172 / 4.44% | 270/473 |
| legacy_affine | cam0 | tuning | 9.803 / 17.822 / 8.81% | 261/497 |
| legacy_affine | cam0 | all | 8.908 / 15.990 / 6.59% | 531/970 |
| legacy_affine | cam1 | evaluation | 13.324 / 27.273 / 30.30% | 231/484 |
| legacy_affine | cam1 | tuning | 15.971 / 39.554 / 37.71% | 236/501 |
| legacy_affine | cam1 | all | 14.822 / 35.377 / 34.05% | 467/985 |
| legacy_affine | cam2 | evaluation | 8.563 / 43.104 / 22.29% | 323/505 |
| legacy_affine | cam2 | tuning | 9.185 / 18.753 / 6.79% | 324/499 |
| legacy_affine | cam2 | all | 9.025 / 31.767 / 14.53% | 647/1004 |
| annotation_oracle | cam0 | evaluation | 6.258 / 10.443 / 0.22% | 447/473 |
| annotation_oracle | cam0 | tuning | 7.722 / 11.938 / 0.43% | 460/497 |
| annotation_oracle | cam0 | all | 7.076 / 11.072 / 0.33% | 907/970 |
| annotation_oracle | cam1 | evaluation | 10.139 / 19.360 / 8.35% | 443/484 |
| annotation_oracle | cam1 | tuning | 11.394 / 24.631 / 13.47% | 475/501 |
| annotation_oracle | cam1 | all | 11.049 / 20.341 / 11.00% | 918/985 |
| annotation_oracle | cam2 | evaluation | 7.887 / 41.457 / 24.62% | 455/505 |
| annotation_oracle | cam2 | tuning | 9.960 / 30.659 / 19.92% | 487/499 |
| annotation_oracle | cam2 | all | 8.698 / 38.342 / 22.19% | 942/1004 |

## baselineと選定方式の共通採点点

| 方式 | 区間 | LOVO | 採点数 |
|---|---|---:|---:|
| pipeline_baseline | evaluation | 9.356 / 27.092 / 16.40% | 823 |
| pipeline_baseline | tuning | 10.699 / 28.311 / 16.26% | 818 |
| pipeline_baseline | all | 10.193 / 28.005 / 16.33% | 1641 |
| selected_affine | evaluation | 9.500 / 29.538 / 18.59% | 823 |
| selected_affine | tuning | 10.147 / 27.554 / 15.04% | 818 |
| selected_affine | all | 9.837 / 28.240 / 16.82% | 1641 |
