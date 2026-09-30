# 同一validationの精度比較

点要約は最大weight成分の平均。r18/e9は保存済み同一frameの参照値で、検出器は再推論しない。
位置NLLはsource px²密度。HDRは全GMMのR²領域、面積は平均source px²。MC設定はr19と同じ。
e9のgap密度は一様で、50/90/95%ともcoverage1・全画面の自明な領域。較正成功と扱わない。
推定位置は参考値、unknownはN/A、不在は位置N/Aで存在NLLだけ。すべての母数はJSONを参照。

## 観測位置誤差 p50 / p90 / p95 (px)

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 23007 | 23.66 / 175.2 / 295.2 | 6.087 / 232.1 / 397.2 | 3.436 / 134.3 / 288.3 |
| meiji/cam0/observed/observed | 7556 | 28.32 / 187.7 / 277.6 | 7.603 / 267.7 / 407.9 | 4.319 / 156.7 / 279.2 |
| meiji/cam1/observed/observed | 7999 | 16.29 / 165 / 361.9 | 4.7 / 180.8 / 407 | 2.649 / 136.6 / 348.1 |
| meiji/cam2/observed/observed | 7452 | 25.18 / 163.5 / 283.7 | 6.663 / 214.8 / 369.1 | 3.775 / 96.02 / 262.4 |
| meiji/selection/observed/observed | 10939 | 22.44 / 94.25 / 195.6 | 5.65 / 93.47 / 275.1 | 3.117 / 54.33 / 142 |
| meiji/calibration/observed/observed | 12068 | 24.86 / 245.4 / 403.7 | 6.588 / 297.9 / 484.9 | 3.833 / 244.3 / 413.6 |
| meiji/selection/cam0/observed/observed | 3604 | 28.22 / 116.7 / 199.8 | 7.101 / 160.7 / 304.6 | 3.759 / 93.89 / 169 |
| meiji/selection/cam1/observed/observed | 3828 | 15.35 / 42.53 / 209.2 | 4.307 / 17.92 / 251.1 | 2.417 / 14.21 / 133.5 |
| meiji/selection/cam2/observed/observed | 3507 | 23.76 / 82.75 / 169.2 | 6.266 / 65.71 / 222.7 | 3.415 / 38.72 / 86.25 |
| meiji/calibration/cam0/observed/observed | 3952 | 28.39 / 243.2 / 322 | 8.267 / 315.2 / 471.2 | 5.41 / 259.8 / 379.7 |
| meiji/calibration/cam1/observed/observed | 4171 | 17.7 / 265.5 / 505.9 | 5.065 / 279.6 / 554.9 | 2.91 / 274.5 / 549.2 |
| meiji/calibration/cam2/observed/observed | 3945 | 26.87 / 239.5 / 370.4 | 7.091 / 300 / 445.5 | 4.169 / 186.7 / 327.5 |
| tracknet/observed/observed | 1538 | 16.44 / 24.68 / 28.69 | 3.388 / 7.229 / 9.898 | 2.342 / 5.546 / 8.395 |
| chat_annotation/observed/observed | 6622 | 29.81 / 170.3 / 315.3 | 6.675 / 132.7 / 371.9 | 4.133 / 114.7 / 279.3 |
## 位置NLL observed / artificial gaps

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 23007 | 9.578 | 10.5 | 6.675 |
| meiji/evidence_gap/observed | 5077 | 10.75 | 14.54 | 10.13 |
| meiji/cam0/observed/observed | 7556 | 9.84 | 10.67 | 7.411 |
| meiji/cam0/evidence_gap/observed | 1697 | 10.45 | 14.54 | 10.08 |
| meiji/cam1/observed/observed | 7999 | 9.314 | 10.3 | 6.172 |
| meiji/cam1/evidence_gap/observed | 1738 | 10.98 | 14.54 | 10.31 |
| meiji/cam2/observed/observed | 7452 | 9.596 | 10.53 | 6.468 |
| meiji/cam2/evidence_gap/observed | 1642 | 10.8 | 14.54 | 9.989 |
| meiji/selection/observed/observed | 10939 | 9.054 | 10.39 | 5.849 |
| meiji/selection/evidence_gap/observed | 2493 | 10.28 | 14.54 | 9.576 |
| meiji/calibration/observed/observed | 12068 | 10.05 | 10.59 | 7.423 |
| meiji/calibration/evidence_gap/observed | 2584 | 11.2 | 14.54 | 10.66 |
| meiji/selection/cam0/observed/observed | 3604 | 9.51 | 10.58 | 6.406 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 10.25 | 14.54 | 9.458 |
| meiji/selection/cam1/observed/observed | 3828 | 8.624 | 10.17 | 5.179 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 10.41 | 14.54 | 9.658 |
| meiji/selection/cam2/observed/observed | 3507 | 9.056 | 10.43 | 6.007 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 10.16 | 14.54 | 9.607 |
| meiji/calibration/cam0/observed/observed | 3952 | 10.14 | 10.74 | 8.328 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 10.64 | 14.54 | 10.66 |
| meiji/calibration/cam1/observed/observed | 4171 | 9.947 | 10.42 | 7.083 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 11.54 | 14.54 | 10.95 |
| meiji/calibration/cam2/observed/observed | 3945 | 10.08 | 10.63 | 6.877 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 11.43 | 14.54 | 10.36 |
| tracknet/observed/observed | 1538 | 7.827 | 9.068 | 4.476 |
| tracknet/evidence_gap/observed | 349 | 10.36 | 13.73 | 9.548 |
| chat_annotation/observed/observed | 6622 | 9.719 | 10.43 | 6.543 |
| chat_annotation/evidence_gap/observed | 1502 | 11.8 | 14.54 | 10.91 |
## HDR 50%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 23007 | 0.4565 / 2322 | 0.9996 / 4.025e+05 | 0.4757 / 720.3 |
| meiji/evidence_gap/observed | 5077 | 0.5655 / 6991 | 1 / 2.071e+06 | 0.5625 / 7479 |
| meiji/cam0/observed/observed | 7556 | 0.2779 / 1981 | 0.9995 / 3.407e+05 | 0.4389 / 733.5 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.5522 / 5355 | 1 / 2.071e+06 | 0.5522 / 7343 |
| meiji/cam1/observed/observed | 7999 | 0.6648 / 1908 | 1 / 4.646e+05 | 0.5146 / 550.1 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.5961 / 8256 | 1 / 2.071e+06 | 0.5967 / 6735 |
| meiji/cam2/observed/observed | 7452 | 0.4138 / 3111 | 0.9993 / 3.984e+05 | 0.4714 / 889.6 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.5469 / 7342 | 1 / 2.071e+06 | 0.5371 / 8407 |
| meiji/selection/observed/observed | 10939 | 0.487 / 2261 | 0.9995 / 4.129e+05 | 0.5002 / 674.5 |
| meiji/selection/evidence_gap/observed | 2493 | 0.6237 / 6995 | 1 / 2.071e+06 | 0.5973 / 7069 |
| meiji/calibration/observed/observed | 12068 | 0.4288 / 2377 | 0.9997 / 3.93e+05 | 0.4535 / 761.8 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.5093 / 6987 | 1 / 2.071e+06 | 0.529 / 7875 |
| meiji/selection/cam0/observed/observed | 3604 | 0.2775 / 1964 | 1 / 3.555e+05 | 0.4625 / 772.2 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.5737 / 5405 | 1 / 2.071e+06 | 0.5907 / 8770 |
| meiji/selection/cam1/observed/observed | 3828 | 0.7168 / 1777 | 1 / 4.737e+05 | 0.5517 / 692.6 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.6486 / 8542 | 1 / 2.071e+06 | 0.6405 / 6328 |
| meiji/selection/cam2/observed/observed | 3507 | 0.4514 / 3094 | 0.9986 / 4.054e+05 | 0.4827 / 554.4 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.6481 / 6954 | 1 / 2.071e+06 | 0.5576 / 6134 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.2783 / 1997 | 0.999 / 3.272e+05 | 0.4173 / 698.2 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.532 / 5309 | 1 / 2.071e+06 | 0.516 / 6006 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.6171 / 2028 | 1 / 4.562e+05 | 0.4805 / 419.4 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.5441 / 7973 | 1 / 2.071e+06 | 0.5533 / 7139 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.3805 / 3125 | 1 / 3.921e+05 | 0.4613 / 1188 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.4491 / 7718 | 1 / 2.071e+06 | 0.5174 / 1.06e+04 |
| tracknet/observed/observed | 1538 | 0.3583 / 820 | 1 / 2.637e+05 | 0.4857 / 29.78 |
| tracknet/evidence_gap/observed | 349 | 0.4269 / 6392 | 1 / 9.196e+05 | 0.4556 / 3685 |
| chat_annotation/observed/observed | 6622 | 0.3689 / 1.142e+04 | 0.9937 / 5.175e+05 | 0.4485 / 4196 |
| chat_annotation/evidence_gap/observed | 1502 | 0.4634 / 3.499e+04 | 1 / 2.071e+06 | 0.5672 / 4.401e+04 |
## HDR 90%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 23007 | 0.8732 / 2.058e+04 | 1 / 1.682e+06 | 0.8254 / 7444 |
| meiji/evidence_gap/observed | 5077 | 0.8523 / 4.926e+04 | 1 / 2.071e+06 | 0.8789 / 3.51e+04 |
| meiji/cam0/observed/observed | 7556 | 0.8475 / 1.919e+04 | 1 / 1.67e+06 | 0.8073 / 7324 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.8662 / 3.834e+04 | 1 / 2.071e+06 | 0.8639 / 3.588e+04 |
| meiji/cam1/observed/observed | 7999 | 0.8847 / 1.506e+04 | 1 / 1.695e+06 | 0.8382 / 6197 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.8343 / 5.513e+04 | 1 / 2.071e+06 | 0.8677 / 3.061e+04 |
| meiji/cam2/observed/observed | 7452 | 0.8867 / 2.793e+04 | 1 / 1.681e+06 | 0.8301 / 8905 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.8569 / 5.434e+04 | 1 / 2.071e+06 | 0.9062 / 3.905e+04 |
| meiji/selection/observed/observed | 10939 | 0.9264 / 1.878e+04 | 1 / 1.688e+06 | 0.8596 / 6232 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9057 / 4.841e+04 | 1 / 2.071e+06 | 0.9382 / 3.39e+04 |
| meiji/calibration/observed/observed | 12068 | 0.8249 / 2.223e+04 | 1 / 1.677e+06 | 0.7945 / 8543 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8007 / 5.008e+04 | 1 / 2.071e+06 | 0.8216 / 3.626e+04 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9054 / 1.921e+04 | 1 / 1.678e+06 | 0.8496 / 8027 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9074 / 3.902e+04 | 1 / 2.071e+06 | 0.9415 / 4.39e+04 |
| meiji/selection/cam1/observed/observed | 3828 | 0.936 / 1.326e+04 | 1 / 1.703e+06 | 0.8783 / 5180 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.8983 / 5.623e+04 | 1 / 2.071e+06 | 0.9225 / 2.769e+04 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9376 / 2.435e+04 | 1 / 1.683e+06 | 0.8494 / 5536 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.912 / 4.958e+04 | 1 / 2.071e+06 | 0.9517 / 3.037e+04 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.7948 / 1.917e+04 | 1 / 1.663e+06 | 0.7687 / 6682 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8276 / 3.77e+04 | 1 / 2.071e+06 | 0.7911 / 2.837e+04 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8377 / 1.671e+04 | 1 / 1.688e+06 | 0.8015 / 7131 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.7709 / 5.403e+04 | 1 / 2.071e+06 | 0.8133 / 3.35e+04 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.8416 / 3.112e+04 | 1 / 1.679e+06 | 0.8129 / 1.19e+04 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8036 / 5.895e+04 | 1 / 2.071e+06 | 0.8623 / 4.743e+04 |
| tracknet/observed/observed | 1538 | 0.9928 / 4640 | 1 / 7.818e+05 | 0.8596 / 167.4 |
| tracknet/evidence_gap/observed | 349 | 0.9341 / 3.982e+04 | 1 / 9.196e+05 | 0.9255 / 1.545e+04 |
| chat_annotation/observed/observed | 6622 | 0.9467 / 7.328e+04 | 0.9989 / 1.695e+06 | 0.825 / 2.789e+04 |
| chat_annotation/evidence_gap/observed | 1502 | 0.8935 / 1.862e+05 | 1 / 2.071e+06 | 0.9321 / 1.84e+05 |
## HDR 95%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 23007 | 0.9076 / 3.893e+04 | 1 / 1.875e+06 | 0.881 / 1.178e+04 |
| meiji/evidence_gap/observed | 5077 | 0.9066 / 8.657e+04 | 1 / 2.071e+06 | 0.908 / 5.005e+04 |
| meiji/cam0/observed/observed | 7556 | 0.9043 / 4.024e+04 | 1 / 1.869e+06 | 0.8706 / 1.149e+04 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.9263 / 7.379e+04 | 1 / 2.071e+06 | 0.8981 / 5.138e+04 |
| meiji/cam1/observed/observed | 7999 | 0.9009 / 2.735e+04 | 1 / 1.881e+06 | 0.8871 / 9944 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.8809 / 9.189e+04 | 1 / 2.071e+06 | 0.8895 / 4.402e+04 |
| meiji/cam2/observed/observed | 7452 | 0.9181 / 5.004e+04 | 1 / 1.874e+06 | 0.8851 / 1.404e+04 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.9135 / 9.415e+04 | 1 / 2.071e+06 | 0.9379 / 5.505e+04 |
| meiji/selection/observed/observed | 10939 | 0.9579 / 3.536e+04 | 1 / 1.878e+06 | 0.9168 / 9840 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9551 / 8.529e+04 | 1 / 2.071e+06 | 0.9627 / 4.83e+04 |
| meiji/calibration/observed/observed | 12068 | 0.862 / 4.217e+04 | 1 / 1.872e+06 | 0.8486 / 1.354e+04 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8599 / 8.78e+04 | 1 / 2.071e+06 | 0.8553 / 5.173e+04 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9567 / 3.987e+04 | 1 / 1.873e+06 | 0.9148 / 1.245e+04 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9622 / 7.566e+04 | 1 / 2.071e+06 | 0.9695 / 6.225e+04 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9498 / 2.378e+04 | 1 / 1.886e+06 | 0.9271 / 7955 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9457 / 9.293e+04 | 1 / 2.071e+06 | 0.9457 / 3.974e+04 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9678 / 4.338e+04 | 1 / 1.876e+06 | 0.9076 / 9219 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.9579 / 8.69e+04 | 1 / 2.071e+06 | 0.974 / 4.328e+04 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.8565 / 4.058e+04 | 1 / 1.866e+06 | 0.8302 / 1.062e+04 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8927 / 7.203e+04 | 1 / 2.071e+06 | 0.8311 / 4.12e+04 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8559 / 3.062e+04 | 1 / 1.877e+06 | 0.8504 / 1.177e+04 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.8167 / 9.085e+04 | 1 / 2.071e+06 | 0.8339 / 4.825e+04 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.874 / 5.596e+04 | 1 / 1.874e+06 | 0.8651 / 1.832e+04 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8707 / 1.011e+05 | 1 / 2.071e+06 | 0.903 / 6.643e+04 |
| tracknet/observed/observed | 1538 | 0.9974 / 7990 | 1 / 8.504e+05 | 0.9109 / 310.4 |
| tracknet/evidence_gap/observed | 349 | 0.9771 / 6.376e+04 | 1 / 9.196e+05 | 0.957 / 2.375e+04 |
| chat_annotation/observed/observed | 6622 | 0.966 / 1.105e+05 | 0.9992 / 1.879e+06 | 0.8818 / 4.083e+04 |
| chat_annotation/evidence_gap/observed | 1502 | 0.9361 / 2.75e+05 | 1 / 2.071e+06 | 0.9614 / 2.547e+05 |
## 存在NLL（detectorはamodal存在を出さないためN/A）

| group/condition | n | r18_refiner | e9_detector | anchored_s44 |
|---|---:|---|---|---|
| meiji/observed/observed | 0/23007 | 0.00783 | N/A | 0.008491 |
| meiji/observed/absent | 0 | N/A | N/A | N/A |
| meiji/evidence_gap/observed | 0/5077 | 0.01182 | N/A | 0.01044 |
| meiji/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/cam0/observed/observed | 0/7556 | 0.006846 | N/A | 0.008182 |
| meiji/cam0/observed/absent | 0 | N/A | N/A | N/A |
| meiji/cam0/evidence_gap/observed | 0/1697 | 0.009568 | N/A | 0.01388 |
| meiji/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/cam1/observed/observed | 0/7999 | 0.006433 | N/A | 0.007846 |
| meiji/cam1/observed/absent | 0 | N/A | N/A | N/A |
| meiji/cam1/evidence_gap/observed | 0/1738 | 0.01383 | N/A | 0.006562 |
| meiji/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/cam2/observed/observed | 0/7452 | 0.01033 | N/A | 0.009498 |
| meiji/cam2/observed/absent | 0 | N/A | N/A | N/A |
| meiji/cam2/evidence_gap/observed | 0/1642 | 0.01203 | N/A | 0.01098 |
| meiji/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/selection/observed/observed | 0/10939 | 0.007835 | N/A | 0.008845 |
| meiji/selection/observed/absent | 0 | N/A | N/A | N/A |
| meiji/selection/evidence_gap/observed | 0/2493 | 0.01192 | N/A | 0.01291 |
| meiji/selection/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/observed/observed | 0/12068 | 0.007826 | N/A | 0.00817 |
| meiji/calibration/observed/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/evidence_gap/observed | 0/2584 | 0.01173 | N/A | 0.008051 |
| meiji/calibration/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam0/observed/observed | 0/3604 | 0.006853 | N/A | 0.01324 |
| meiji/selection/cam0/observed/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam0/evidence_gap/observed | 0/821 | 0.009792 | N/A | 0.02423 |
| meiji/selection/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam1/observed/observed | 0/3828 | 0.005931 | N/A | 0.007145 |
| meiji/selection/cam1/observed/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam1/evidence_gap/observed | 0/865 | 0.01408 | N/A | 0.006655 |
| meiji/selection/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam2/observed/observed | 0/3507 | 0.01092 | N/A | 0.006182 |
| meiji/selection/cam2/observed/absent | 0 | N/A | N/A | N/A |
| meiji/selection/cam2/evidence_gap/observed | 0/807 | 0.01177 | N/A | 0.008099 |
| meiji/selection/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam0/observed/observed | 0/3952 | 0.00684 | N/A | 0.003566 |
| meiji/calibration/cam0/observed/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam0/evidence_gap/observed | 0/876 | 0.009359 | N/A | 0.00417 |
| meiji/calibration/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam1/observed/observed | 0/4171 | 0.006894 | N/A | 0.00849 |
| meiji/calibration/cam1/observed/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam1/evidence_gap/observed | 0/873 | 0.01358 | N/A | 0.00647 |
| meiji/calibration/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam2/observed/observed | 0/3945 | 0.0098 | N/A | 0.01245 |
| meiji/calibration/cam2/observed/absent | 0 | N/A | N/A | N/A |
| meiji/calibration/cam2/evidence_gap/observed | 0/835 | 0.01228 | N/A | 0.01377 |
| meiji/calibration/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A |
| tracknet/observed/observed | 0/1538 | 0.01094 | N/A | 0.0002405 |
| tracknet/observed/absent | 0 | N/A | N/A | N/A |
| tracknet/evidence_gap/observed | 0/349 | 0.0123 | N/A | 0.001871 |
| tracknet/evidence_gap/absent | 0 | N/A | N/A | N/A |
| chat_annotation/observed/observed | 0/6622 | 0.03063 | N/A | 0.02263 |
| chat_annotation/observed/absent | 0/269 | 2.213 | N/A | 1.877 |
| chat_annotation/evidence_gap/observed | 0/1502 | 0.03847 | N/A | 0.05365 |
| chat_annotation/evidence_gap/absent | 0/66 | 2.772 | N/A | 2.965 |
