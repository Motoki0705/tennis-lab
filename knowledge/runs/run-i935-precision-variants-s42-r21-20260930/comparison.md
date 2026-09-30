# 同一validationの精度比較

点要約は最大weight成分の平均。r18/e9は保存済み同一frameの参照値で、検出器は再推論しない。
位置NLLはsource px²密度。HDRは全GMMのR²領域、面積は平均source px²。MC設定はr19と同じ。
e9のgap密度は一様で、50/90/95%ともcoverage1・全画面の自明な領域。較正成功と扱わない。
推定位置は参考値、unknownはN/A、不在は位置N/Aで存在NLLだけ。すべての母数はJSONを参照。

## 観測位置誤差 p50 / p90 / p95 (px)

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 23007 | 23.66 / 175.2 / 295.2 | 6.087 / 232.1 / 397.2 | 10.52 / 162.8 / 301.5 | 4.197 / 169.4 / 316.6 | 3.109 / 133.7 / 273 |
| meiji/cam0/observed/observed | 7556 | 28.32 / 187.7 / 277.6 | 7.603 / 267.7 / 407.9 | 11.83 / 168.7 / 271.3 | 5.267 / 197.8 / 322.6 | 3.885 / 150.1 / 266.8 |
| meiji/cam1/observed/observed | 7999 | 16.29 / 165 / 361.9 | 4.7 / 180.8 / 407 | 9.103 / 187.7 / 412.1 | 3.357 / 164.7 / 386 | 2.377 / 135.5 / 337.5 |
| meiji/cam2/observed/observed | 7452 | 25.18 / 163.5 / 283.7 | 6.663 / 214.8 / 369.1 | 10.96 / 147.4 / 287.1 | 4.574 / 143.4 / 295 | 3.416 / 108.2 / 230.3 |
| meiji/selection/observed/observed | 10939 | 22.44 / 94.25 / 195.6 | 5.65 / 93.47 / 275.1 | 10.04 / 82.24 / 176.3 | 3.814 / 85.13 / 191 | 2.772 / 56.98 / 144.4 |
| meiji/calibration/observed/observed | 12068 | 24.86 / 245.4 / 403.7 | 6.588 / 297.9 / 484.9 | 11.08 / 253.8 / 421.9 | 4.681 / 266.6 / 439.1 | 3.528 / 219.9 / 377.6 |
| meiji/selection/cam0/observed/observed | 3604 | 28.22 / 116.7 / 199.8 | 7.101 / 160.7 / 304.6 | 11.42 / 122.1 / 173.1 | 4.522 / 132.9 / 197.7 | 3.452 / 93.49 / 162.8 |
| meiji/selection/cam1/observed/observed | 3828 | 15.35 / 42.53 / 209.2 | 4.307 / 17.92 / 251.1 | 8.686 / 27.89 / 205.5 | 3.06 / 20.23 / 209.4 | 2.143 / 15.34 / 113.2 |
| meiji/selection/cam2/observed/observed | 3507 | 23.76 / 82.75 / 169.2 | 6.266 / 65.71 / 222.7 | 10.35 / 56.78 / 160.6 | 4.283 / 59.64 / 156.6 | 3.053 / 44.86 / 124.6 |
| meiji/calibration/cam0/observed/observed | 3952 | 28.39 / 243.2 / 322 | 8.267 / 315.2 / 471.2 | 12.59 / 244.6 / 346.8 | 6.323 / 272.9 / 451.8 | 4.705 / 225.1 / 338.4 |
| meiji/calibration/cam1/observed/observed | 4171 | 17.7 / 265.5 / 505.9 | 5.065 / 279.6 / 554.9 | 9.599 / 299.4 / 575.2 | 3.708 / 275.4 / 529 | 2.696 / 275.5 / 513.8 |
| meiji/calibration/cam2/observed/observed | 3945 | 26.87 / 239.5 / 370.4 | 7.091 / 300 / 445.5 | 11.69 / 213.5 / 323.6 | 4.97 / 230.5 / 335.9 | 3.806 / 172.3 / 293.6 |
| tracknet/observed/observed | 1538 | 16.44 / 24.68 / 28.69 | 3.388 / 7.229 / 9.898 | 7.021 / 12.19 / 15.16 | 2.531 / 5.923 / 8.767 | 2.219 / 5.49 / 8.486 |
| chat_annotation/observed/observed | 6622 | 29.81 / 170.3 / 315.3 | 6.675 / 132.7 / 371.9 | 13.58 / 150.1 / 283.1 | 4.748 / 124.4 / 251.1 | 3.585 / 94.35 / 211 |
## 位置NLL observed / artificial gaps

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 23007 | 9.578 | 10.5 | 8.786 | 6.65 | 6.543 |
| meiji/evidence_gap/observed | 5077 | 10.75 | 14.54 | 9.993 | 10.94 | 9.976 |
| meiji/cam0/observed/observed | 7556 | 9.84 | 10.67 | 8.81 | 7.138 | 7.271 |
| meiji/cam0/evidence_gap/observed | 1697 | 10.45 | 14.54 | 9.784 | 10.97 | 9.841 |
| meiji/cam1/observed/observed | 7999 | 9.314 | 10.3 | 8.948 | 6.155 | 6.016 |
| meiji/cam1/evidence_gap/observed | 1738 | 10.98 | 14.54 | 10.24 | 11.06 | 10.11 |
| meiji/cam2/observed/observed | 7452 | 9.596 | 10.53 | 8.587 | 6.687 | 6.372 |
| meiji/cam2/evidence_gap/observed | 1642 | 10.8 | 14.54 | 9.946 | 10.8 | 9.977 |
| meiji/selection/observed/observed | 10939 | 9.054 | 10.39 | 8.014 | 6.11 | 5.686 |
| meiji/selection/evidence_gap/observed | 2493 | 10.28 | 14.54 | 9.49 | 10.49 | 9.473 |
| meiji/calibration/observed/observed | 12068 | 10.05 | 10.59 | 9.485 | 7.14 | 7.32 |
| meiji/calibration/evidence_gap/observed | 2584 | 11.2 | 14.54 | 10.48 | 11.38 | 10.46 |
| meiji/selection/cam0/observed/observed | 3604 | 9.51 | 10.58 | 8.161 | 6.598 | 6.298 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 10.25 | 14.54 | 9.43 | 10.51 | 9.499 |
| meiji/selection/cam1/observed/observed | 3828 | 8.624 | 10.17 | 7.733 | 5.505 | 5.016 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 10.41 | 14.54 | 9.584 | 10.57 | 9.5 |
| meiji/selection/cam2/observed/observed | 3507 | 9.056 | 10.43 | 8.169 | 6.269 | 5.787 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 10.16 | 14.54 | 9.45 | 10.39 | 9.418 |
| meiji/calibration/cam0/observed/observed | 3952 | 10.14 | 10.74 | 9.402 | 7.631 | 8.157 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 10.64 | 14.54 | 10.12 | 11.4 | 10.16 |
| meiji/calibration/cam1/observed/observed | 4171 | 9.947 | 10.42 | 10.06 | 6.752 | 6.934 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 11.54 | 14.54 | 10.89 | 11.54 | 10.71 |
| meiji/calibration/cam2/observed/observed | 3945 | 10.08 | 10.63 | 8.958 | 7.058 | 6.891 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 11.43 | 14.54 | 10.43 | 11.2 | 10.52 |
| tracknet/observed/observed | 1538 | 7.827 | 9.068 | 6.344 | 4.563 | 4.409 |
| tracknet/evidence_gap/observed | 349 | 10.36 | 13.73 | 9.551 | 10.06 | 9.368 |
| chat_annotation/observed/observed | 6622 | 9.719 | 10.43 | 8.47 | 6.616 | 6.286 |
| chat_annotation/evidence_gap/observed | 1502 | 11.8 | 14.54 | 11.15 | 11.68 | 10.92 |
## HDR 50%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 23007 | 0.4565 / 2322 | 0.9996 / 4.025e+05 | 0.6383 / 2580 | 0.4627 / 798 | 0.5155 / 1278 |
| meiji/evidence_gap/observed | 5077 | 0.5655 / 6991 | 1 / 2.071e+06 | 0.6386 / 4472 | 0.5135 / 9631 | 0.5208 / 8300 |
| meiji/cam0/observed/observed | 7556 | 0.2779 / 1981 | 0.9995 / 3.407e+05 | 0.5944 / 2116 | 0.479 / 732.7 | 0.4472 / 983.2 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.5522 / 5355 | 1 / 2.071e+06 | 0.6211 / 3131 | 0.4532 / 7776 | 0.4856 / 7380 |
| meiji/cam1/observed/observed | 7999 | 0.6648 / 1908 | 1 / 4.646e+05 | 0.6978 / 1710 | 0.4421 / 289.2 | 0.5536 / 956.4 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.5961 / 8256 | 1 / 2.071e+06 | 0.6686 / 4555 | 0.5098 / 1.029e+04 | 0.5817 / 7819 |
| meiji/cam2/observed/observed | 7452 | 0.4138 / 3111 | 0.9993 / 3.984e+05 | 0.619 / 3984 | 0.4683 / 1410 | 0.5437 / 1922 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.5469 / 7342 | 1 / 2.071e+06 | 0.6248 / 5769 | 0.5798 / 1.085e+04 | 0.4927 / 9760 |
| meiji/selection/observed/observed | 10939 | 0.487 / 2261 | 0.9995 / 4.129e+05 | 0.6725 / 2064 | 0.4868 / 688.9 | 0.5463 / 1061 |
| meiji/selection/evidence_gap/observed | 2493 | 0.6237 / 6995 | 1 / 2.071e+06 | 0.6915 / 3931 | 0.5744 / 9457 | 0.5732 / 7908 |
| meiji/calibration/observed/observed | 12068 | 0.4288 / 2377 | 0.9997 / 3.93e+05 | 0.6074 / 3048 | 0.4408 / 896.8 | 0.4875 / 1475 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.5093 / 6987 | 1 / 2.071e+06 | 0.5875 / 4994 | 0.4547 / 9799 | 0.4702 / 8679 |
| meiji/selection/cam0/observed/observed | 3604 | 0.2775 / 1964 | 1 / 3.555e+05 | 0.6251 / 1974 | 0.5019 / 699.5 | 0.4723 / 1021 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.5737 / 5405 | 1 / 2.071e+06 | 0.6468 / 3145 | 0.5286 / 8039 | 0.5274 / 9032 |
| meiji/selection/cam1/observed/observed | 3828 | 0.7168 / 1777 | 1 / 4.737e+05 | 0.739 / 1258 | 0.4744 / 346.3 | 0.6016 / 865.9 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.6486 / 8542 | 1 / 2.071e+06 | 0.7376 / 4572 | 0.5538 / 1.03e+04 | 0.6474 / 6942 |
| meiji/selection/cam2/observed/observed | 3507 | 0.4514 / 3094 | 0.9986 / 4.054e+05 | 0.6484 / 3036 | 0.4847 / 1052 | 0.562 / 1315 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.6481 / 6954 | 1 / 2.071e+06 | 0.6877 / 4043 | 0.6431 / 9997 | 0.5403 / 7801 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.2783 / 1997 | 0.999 / 3.272e+05 | 0.5663 / 2244 | 0.458 / 762.9 | 0.4243 / 948.9 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.532 / 5309 | 1 / 2.071e+06 | 0.597 / 3119 | 0.3824 / 7530 | 0.4463 / 5832 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.6171 / 2028 | 1 / 4.562e+05 | 0.66 / 2125 | 0.4124 / 236.9 | 0.5095 / 1040 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.5441 / 7973 | 1 / 2.071e+06 | 0.6002 / 4538 | 0.4662 / 1.028e+04 | 0.5166 / 8689 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.3805 / 3125 | 1 / 3.921e+05 | 0.5929 / 4828 | 0.4537 / 1729 | 0.5275 / 2462 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.4491 / 7718 | 1 / 2.071e+06 | 0.5641 / 7438 | 0.5186 / 1.167e+04 | 0.4467 / 1.165e+04 |
| tracknet/observed/observed | 1538 | 0.3583 / 820 | 1 / 2.637e+05 | 0.6352 / 327.5 | 0.5033 / 69.4 | 0.4915 / 31.32 |
| tracknet/evidence_gap/observed | 349 | 0.4269 / 6392 | 1 / 9.196e+05 | 0.6447 / 4470 | 0.6963 / 9389 | 0.3467 / 2666 |
| chat_annotation/observed/observed | 6622 | 0.3689 / 1.142e+04 | 0.9937 / 5.175e+05 | 0.5362 / 9310 | 0.503 / 8390 | 0.5109 / 6976 |
| chat_annotation/evidence_gap/observed | 1502 | 0.4634 / 3.499e+04 | 1 / 2.071e+06 | 0.5553 / 2.965e+04 | 0.5752 / 4.633e+04 | 0.4887 / 3.992e+04 |
## HDR 90%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 23007 | 0.8732 / 2.058e+04 | 1 / 1.682e+06 | 0.8873 / 1.366e+04 | 0.843 / 1.073e+04 | 0.836 / 1.009e+04 |
| meiji/evidence_gap/observed | 5077 | 0.8523 / 4.926e+04 | 1 / 2.071e+06 | 0.8407 / 3.711e+04 | 0.847 / 3.886e+04 | 0.8915 / 3.826e+04 |
| meiji/cam0/observed/observed | 7556 | 0.8475 / 1.919e+04 | 1 / 1.67e+06 | 0.8716 / 1.219e+04 | 0.8641 / 1.242e+04 | 0.7885 / 8753 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.8662 / 3.834e+04 | 1 / 2.071e+06 | 0.8244 / 2.525e+04 | 0.8256 / 3.367e+04 | 0.8886 / 3.681e+04 |
| meiji/cam1/observed/observed | 7999 | 0.8847 / 1.506e+04 | 1 / 1.695e+06 | 0.8856 / 9295 | 0.8214 / 5283 | 0.8624 / 9313 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.8343 / 5.513e+04 | 1 / 2.071e+06 | 0.832 / 2.97e+04 | 0.8562 / 3.84e+04 | 0.8964 / 3.526e+04 |
| meiji/cam2/observed/observed | 7452 | 0.8867 / 2.793e+04 | 1 / 1.681e+06 | 0.905 / 1.985e+04 | 0.8449 / 1.488e+04 | 0.856 / 1.229e+04 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.8569 / 5.434e+04 | 1 / 2.071e+06 | 0.8666 / 5.721e+04 | 0.8593 / 4.471e+04 | 0.8892 / 4.293e+04 |
| meiji/selection/observed/observed | 10939 | 0.9264 / 1.878e+04 | 1 / 1.688e+06 | 0.9377 / 1.062e+04 | 0.8786 / 9559 | 0.8753 / 9081 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9057 / 4.841e+04 | 1 / 2.071e+06 | 0.9045 / 2.873e+04 | 0.9073 / 3.793e+04 | 0.9438 / 3.642e+04 |
| meiji/calibration/observed/observed | 12068 | 0.8249 / 2.223e+04 | 1 / 1.677e+06 | 0.8416 / 1.642e+04 | 0.8107 / 1.18e+04 | 0.8005 / 1.101e+04 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8007 / 5.008e+04 | 1 / 2.071e+06 | 0.779 / 4.52e+04 | 0.7887 / 3.976e+04 | 0.8409 / 4.003e+04 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9054 / 1.921e+04 | 1 / 1.678e+06 | 0.9218 / 1.077e+04 | 0.9107 / 1.249e+04 | 0.8393 / 1.036e+04 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9074 / 3.902e+04 | 1 / 2.071e+06 | 0.894 / 2.783e+04 | 0.8904 / 3.568e+04 | 0.9403 / 4.328e+04 |
| meiji/selection/cam1/observed/observed | 3828 | 0.936 / 1.326e+04 | 1 / 1.703e+06 | 0.9389 / 6907 | 0.8673 / 5288 | 0.9075 / 7580 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.8983 / 5.623e+04 | 1 / 2.071e+06 | 0.8936 / 2.619e+04 | 0.9121 / 3.828e+04 | 0.9434 / 3.036e+04 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9376 / 2.435e+04 | 1 / 1.683e+06 | 0.9527 / 1.453e+04 | 0.858 / 1.121e+04 | 0.8771 / 9404 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.912 / 4.958e+04 | 1 / 2.071e+06 | 0.9269 / 3.236e+04 | 0.9195 / 3.983e+04 | 0.948 / 3.595e+04 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.7948 / 1.917e+04 | 1 / 1.663e+06 | 0.8259 / 1.349e+04 | 0.8216 / 1.235e+04 | 0.7422 / 7287 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8276 / 3.77e+04 | 1 / 2.071e+06 | 0.7591 / 2.283e+04 | 0.7648 / 3.179e+04 | 0.8402 / 3.074e+04 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8377 / 1.671e+04 | 1 / 1.688e+06 | 0.8367 / 1.149e+04 | 0.7792 / 5279 | 0.8209 / 1.09e+04 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.7709 / 5.403e+04 | 1 / 2.071e+06 | 0.7709 / 3.318e+04 | 0.8007 / 3.851e+04 | 0.8499 / 4.011e+04 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.8416 / 3.112e+04 | 1 / 1.679e+06 | 0.8626 / 2.458e+04 | 0.8332 / 1.814e+04 | 0.8373 / 1.486e+04 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8036 / 5.895e+04 | 1 / 2.071e+06 | 0.8084 / 8.124e+04 | 0.8012 / 4.943e+04 | 0.8323 / 4.969e+04 |
| tracknet/observed/observed | 1538 | 0.9928 / 4640 | 1 / 7.818e+05 | 0.9883 / 1406 | 0.8921 / 544.3 | 0.8498 / 196.4 |
| tracknet/evidence_gap/observed | 349 | 0.9341 / 3.982e+04 | 1 / 9.196e+05 | 0.9169 / 1.923e+04 | 0.9742 / 3.441e+04 | 0.8997 / 1.097e+04 |
| chat_annotation/observed/observed | 6622 | 0.9467 / 7.328e+04 | 0.9989 / 1.695e+06 | 0.9283 / 4.758e+04 | 0.8719 / 4.722e+04 | 0.8594 / 3.788e+04 |
| chat_annotation/evidence_gap/observed | 1502 | 0.8935 / 1.862e+05 | 1 / 2.071e+06 | 0.8908 / 1.458e+05 | 0.9095 / 1.795e+05 | 0.9021 / 1.558e+05 |
## HDR 95%: coverage / mean area px²

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 23007 | 0.9076 / 3.893e+04 | 1 / 1.875e+06 | 0.9054 / 2.011e+04 | 0.8989 / 1.758e+04 | 0.8827 / 1.52e+04 |
| meiji/evidence_gap/observed | 5077 | 0.9066 / 8.657e+04 | 1 / 2.071e+06 | 0.9053 / 7.161e+04 | 0.8877 / 5.448e+04 | 0.9226 / 5.383e+04 |
| meiji/cam0/observed/observed | 7556 | 0.9043 / 4.024e+04 | 1 / 1.869e+06 | 0.8999 / 1.803e+04 | 0.9169 / 2.097e+04 | 0.8483 / 1.335e+04 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.9263 / 7.379e+04 | 1 / 2.071e+06 | 0.9004 / 5.098e+04 | 0.8751 / 4.811e+04 | 0.9234 / 5.222e+04 |
| meiji/cam1/observed/observed | 7999 | 0.9009 / 2.735e+04 | 1 / 1.881e+06 | 0.8966 / 1.424e+04 | 0.8792 / 9070 | 0.9019 / 1.427e+04 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.8809 / 9.189e+04 | 1 / 2.071e+06 | 0.8872 / 5.711e+04 | 0.8895 / 5.314e+04 | 0.9189 / 4.991e+04 |
| meiji/cam2/observed/observed | 7452 | 0.9181 / 5.004e+04 | 1 / 1.874e+06 | 0.9203 / 2.851e+04 | 0.9019 / 2.327e+04 | 0.8969 / 1.806e+04 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.9135 / 9.415e+04 | 1 / 2.071e+06 | 0.9294 / 1.083e+05 | 0.8989 / 6.249e+04 | 0.9257 / 5.966e+04 |
| meiji/selection/observed/observed | 10939 | 0.9579 / 3.536e+04 | 1 / 1.878e+06 | 0.9526 / 1.564e+04 | 0.9318 / 1.575e+04 | 0.9217 / 1.378e+04 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9551 / 8.529e+04 | 1 / 2.071e+06 | 0.9503 / 5.488e+04 | 0.9434 / 5.32e+04 | 0.9663 / 5.099e+04 |
| meiji/calibration/observed/observed | 12068 | 0.862 / 4.217e+04 | 1 / 1.872e+06 | 0.8625 / 2.415e+04 | 0.8692 / 1.923e+04 | 0.8474 / 1.648e+04 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8599 / 8.78e+04 | 1 / 2.071e+06 | 0.8618 / 8.775e+04 | 0.834 / 5.572e+04 | 0.8804 / 5.657e+04 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9567 / 3.987e+04 | 1 / 1.873e+06 | 0.9453 / 1.607e+04 | 0.9575 / 2.102e+04 | 0.9001 / 1.579e+04 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9622 / 7.566e+04 | 1 / 2.071e+06 | 0.9622 / 5.467e+04 | 0.9403 / 5.097e+04 | 0.9695 / 6.008e+04 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9498 / 2.378e+04 | 1 / 1.886e+06 | 0.948 / 1.06e+04 | 0.9214 / 8862 | 0.942 / 1.151e+04 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9457 / 9.293e+04 | 1 / 2.071e+06 | 0.9306 / 4.925e+04 | 0.9445 / 5.304e+04 | 0.9549 / 4.312e+04 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9678 / 4.338e+04 | 1 / 1.876e+06 | 0.9652 / 2.071e+04 | 0.9167 / 1.786e+04 | 0.9216 / 1.419e+04 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.9579 / 8.69e+04 | 1 / 2.071e+06 | 0.9591 / 6.112e+04 | 0.9455 / 5.564e+04 | 0.9752 / 5.018e+04 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.8565 / 4.058e+04 | 1 / 1.866e+06 | 0.8586 / 1.981e+04 | 0.8798 / 2.092e+04 | 0.8011 / 1.113e+04 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8927 / 7.203e+04 | 1 / 2.071e+06 | 0.8425 / 4.751e+04 | 0.8139 / 4.543e+04 | 0.8801 / 4.484e+04 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8559 / 3.062e+04 | 1 / 1.877e+06 | 0.8494 / 1.758e+04 | 0.8406 / 9262 | 0.865 / 1.68e+04 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.8167 / 9.085e+04 | 1 / 2.071e+06 | 0.8442 / 6.49e+04 | 0.8351 / 5.324e+04 | 0.8832 / 5.663e+04 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.874 / 5.596e+04 | 1 / 1.874e+06 | 0.8804 / 3.544e+04 | 0.8887 / 2.807e+04 | 0.875 / 2.149e+04 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8707 / 1.011e+05 | 1 / 2.071e+06 | 0.9006 / 1.539e+05 | 0.8539 / 6.911e+04 | 0.8778 / 6.882e+04 |
| tracknet/observed/observed | 1538 | 0.9974 / 7990 | 1 / 8.504e+05 | 0.9974 / 2040 | 0.9363 / 960.1 | 0.9038 / 294.6 |
| tracknet/evidence_gap/observed | 349 | 0.9771 / 6.376e+04 | 1 / 9.196e+05 | 0.9484 / 2.783e+04 | 0.9828 / 4.787e+04 | 0.9656 / 1.571e+04 |
| chat_annotation/observed/observed | 6622 | 0.966 / 1.105e+05 | 0.9992 / 1.879e+06 | 0.9592 / 6.717e+04 | 0.92 / 6.678e+04 | 0.9105 / 5.334e+04 |
| chat_annotation/evidence_gap/observed | 1502 | 0.9361 / 2.75e+05 | 1 / 2.071e+06 | 0.9368 / 2.272e+05 | 0.9394 / 2.45e+05 | 0.9481 / 2.099e+05 |
## 存在NLL（detectorはamodal存在を出さないためN/A）

| group/condition | n | r18_refiner | e9_detector | absolute_12k | anchored_3k | anchored_12k |
|---|---:|---|---|---|---|---|
| meiji/observed/observed | 0/23007 | 0.00783 | N/A | 0.007994 | 0.00337 | 0.006354 |
| meiji/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/evidence_gap/observed | 0/5077 | 0.01182 | N/A | 0.002863 | 0.006115 | 0.004372 |
| meiji/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam0/observed/observed | 0/7556 | 0.006846 | N/A | 0.00882 | 0.002908 | 0.004724 |
| meiji/cam0/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam0/evidence_gap/observed | 0/1697 | 0.009568 | N/A | 0.0024 | 0.005177 | 0.003701 |
| meiji/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam1/observed/observed | 0/7999 | 0.006433 | N/A | 0.004323 | 0.00191 | 0.003739 |
| meiji/cam1/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam1/evidence_gap/observed | 0/1738 | 0.01383 | N/A | 0.002547 | 0.005587 | 0.001823 |
| meiji/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam2/observed/observed | 0/7452 | 0.01033 | N/A | 0.0111 | 0.005406 | 0.01081 |
| meiji/cam2/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/cam2/evidence_gap/observed | 0/1642 | 0.01203 | N/A | 0.003675 | 0.007645 | 0.007765 |
| meiji/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/observed/observed | 0/10939 | 0.007835 | N/A | 0.007198 | 0.003333 | 0.006238 |
| meiji/selection/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/evidence_gap/observed | 0/2493 | 0.01192 | N/A | 0.002541 | 0.006022 | 0.004304 |
| meiji/selection/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/observed/observed | 0/12068 | 0.007826 | N/A | 0.008715 | 0.003403 | 0.006459 |
| meiji/calibration/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/evidence_gap/observed | 0/2584 | 0.01173 | N/A | 0.003173 | 0.006206 | 0.004438 |
| meiji/calibration/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam0/observed/observed | 0/3604 | 0.006853 | N/A | 0.008687 | 0.003136 | 0.008251 |
| meiji/selection/cam0/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam0/evidence_gap/observed | 0/821 | 0.009792 | N/A | 0.002522 | 0.006144 | 0.006099 |
| meiji/selection/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam1/observed/observed | 0/3828 | 0.005931 | N/A | 0.003826 | 0.002087 | 0.00503 |
| meiji/selection/cam1/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam1/evidence_gap/observed | 0/865 | 0.01408 | N/A | 0.002512 | 0.005844 | 0.001665 |
| meiji/selection/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam2/observed/observed | 0/3507 | 0.01092 | N/A | 0.009349 | 0.004895 | 0.005488 |
| meiji/selection/cam2/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/selection/cam2/evidence_gap/observed | 0/807 | 0.01177 | N/A | 0.002593 | 0.006087 | 0.005308 |
| meiji/selection/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam0/observed/observed | 0/3952 | 0.00684 | N/A | 0.008942 | 0.0027 | 0.001507 |
| meiji/calibration/cam0/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam0/evidence_gap/observed | 0/876 | 0.009359 | N/A | 0.002286 | 0.00427 | 0.001453 |
| meiji/calibration/cam0/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam1/observed/observed | 0/4171 | 0.006894 | N/A | 0.004779 | 0.001747 | 0.002555 |
| meiji/calibration/cam1/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam1/evidence_gap/observed | 0/873 | 0.01358 | N/A | 0.002582 | 0.005332 | 0.001979 |
| meiji/calibration/cam1/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam2/observed/observed | 0/3945 | 0.0098 | N/A | 0.01265 | 0.005859 | 0.01555 |
| meiji/calibration/cam2/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| meiji/calibration/cam2/evidence_gap/observed | 0/835 | 0.01228 | N/A | 0.00472 | 0.009149 | 0.01014 |
| meiji/calibration/cam2/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| tracknet/observed/observed | 0/1538 | 0.01094 | N/A | 0.006194 | 0.001605 | 0.0005175 |
| tracknet/observed/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| tracknet/evidence_gap/observed | 0/349 | 0.0123 | N/A | 0.005413 | 0.00404 | 0.001363 |
| tracknet/evidence_gap/absent | 0 | N/A | N/A | N/A | N/A | N/A |
| chat_annotation/observed/observed | 0/6622 | 0.03063 | N/A | 0.02888 | 0.02242 | 0.03327 |
| chat_annotation/observed/absent | 0/269 | 2.213 | N/A | 1.796 | 2.214 | 2.071 |
| chat_annotation/evidence_gap/observed | 0/1502 | 0.03847 | N/A | 0.02109 | 0.02898 | 0.04444 |
| chat_annotation/evidence_gap/absent | 0/66 | 2.772 | N/A | 3.318 | 2.829 | 2.972 |
