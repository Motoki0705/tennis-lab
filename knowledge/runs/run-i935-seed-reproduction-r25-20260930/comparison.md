# 三seed再現性と固定倍率の診断（run25）

判定: **FAIL、9/10**。通常入力の位置3分位点と通常/gap NLLの全条件をseed43・44それぞれに要求する事前規則を適用。
通常observed n=23,007、人工gap内observed n=5,077。Meiji video_000・全camera・両half合算。
NLLはsource px²密度のnat、誤差はpx、HDR面積はGaussianの画面外tailを含むR²上のpx²。
倍率はseed42でfit済みの1.8125148752087792を全seedへ固定適用。再fitなし、LOCO OOFでも独立testでもない。
HDR MC2048 / seed1729 / levels 0.5,0.9,0.95。誤差は最大weight成分の平均。平均・weight・presenceは不変。

## 事前10比較

| seed | 条件 | 指標 | 値 | 対照 | 判定 |
|---|---|---|---:|---:|---|
| 43 | observed | median_error_px | 3.108157 | 6.087080 | PASS |
| 43 | observed | p90_error_px | 121.699871 | 232.119665 | PASS |
| 43 | observed | p95_error_px | 274.372820 | 397.172760 | PASS |
| 43 | observed | mean_nll_px | 6.874575 | 8.785725 | PASS |
| 43 | evidence_gap | mean_nll_px | 9.838953 | 9.992509 | PASS |
| 44 | observed | median_error_px | 3.435645 | 6.087080 | PASS |
| 44 | observed | p90_error_px | 134.263992 | 232.119665 | PASS |
| 44 | observed | p95_error_px | 288.340971 | 397.172760 | PASS |
| 44 | observed | mean_nll_px | 6.674796 | 8.785725 | PASS |
| 44 | evidence_gap | mean_nll_px | 10.127395 | 9.992509 | FAIL |

位置の対照はe9 top-1、NLLの対照はabsolute_12k seed42。倍率適用前で判定し、同値を許容しない。

## 固定倍率のMeiji全体診断

| seed | 条件 | raw NLL → fixed NLL | HDR50 → | HDR90 → | HDR95 → | area90 → |
|---|---|---|---|---|---|---|
| 42 | observed | 6.54326 → 6.42075 | 0.515452 → 0.670752 | 0.83605 → 0.899031 | 0.882688 → 0.924849 | 10093.3 → 18075.6 |
| 42 | evidence_gap | 9.97619 → 10.0716 | 0.52078 → 0.701792 | 0.891471 → 0.93244 | 0.922592 → 0.952137 | 38257.8 → 68849.9 |
| 43 | observed | 6.87458 → 6.52577 | 0.455774 → 0.616465 | 0.798192 → 0.875907 | 0.851697 → 0.907637 | 6525.52 → 11673 |
| 43 | evidence_gap | 9.83895 → 9.8732 | 0.565885 → 0.710262 | 0.856214 → 0.909395 | 0.892653 → 0.928698 | 23893.2 → 42676.2 |
| 44 | observed | 6.6748 → 6.51426 | 0.475725 → 0.637893 | 0.825444 → 0.89777 | 0.881036 → 0.927978 | 7444.19 → 13258.3 |
| 44 | evidence_gap | 10.1274 → 10.1546 | 0.562537 → 0.736262 | 0.878865 → 0.920819 | 0.908017 → 0.936774 | 35099.5 → 62239.3 |

seed44のraw gap NLLはselection/calibration双方およびcam0/1/2それぞれでもabsolute_12kより悪い。
seed平均でこの不合格を打ち消さない。全source/camera/halfの診断は以下。TrackNet/chatにcamera・Meiji halfは定義されない。

## 倍率適用前: median_error_px

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 55.89532 | 47.6271 | 51.96493 | 47.6271 | 55.89532 | 8.268226 |
| chat_annotation/observed/observed | 6622 | 3.584885 | 3.813964 | 4.132709 | 3.584885 | 4.132709 | 0.5478242 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 30.67581 | 24.1171 | 25.02508 | 24.1171 | 30.67581 | 6.558714 |
| meiji/calibration/cam0/observed/observed | 3952 | 4.705221 | 4.687874 | 5.410098 | 4.687874 | 5.410098 | 0.7222245 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 34.98801 | 26.9144 | 34.05153 | 26.9144 | 34.98801 | 8.073606 |
| meiji/calibration/cam1/observed/observed | 4171 | 2.696034 | 2.669117 | 2.909837 | 2.669117 | 2.909837 | 0.2407205 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 39.18638 | 32.83803 | 34.54797 | 32.83803 | 39.18638 | 6.348355 |
| meiji/calibration/cam2/observed/observed | 3945 | 3.806422 | 3.76091 | 4.168666 | 3.76091 | 4.168666 | 0.4077561 |
| meiji/calibration/evidence_gap/observed | 2584 | 35.34307 | 27.31874 | 30.50269 | 27.31874 | 35.34307 | 8.024332 |
| meiji/calibration/observed/observed | 12068 | 3.528022 | 3.472921 | 3.832917 | 3.472921 | 3.832917 | 0.3599951 |
| meiji/cam0/evidence_gap/observed | 1697 | 28.38159 | 21.05427 | 24.27201 | 21.05427 | 28.38159 | 7.327314 |
| meiji/cam0/observed/observed | 7556 | 3.884903 | 3.869939 | 4.318505 | 3.869939 | 4.318505 | 0.4485656 |
| meiji/cam1/evidence_gap/observed | 1738 | 30.16937 | 23.90969 | 29.15417 | 23.90969 | 30.16937 | 6.259687 |
| meiji/cam1/observed/observed | 7999 | 2.376736 | 2.443152 | 2.649498 | 2.376736 | 2.649498 | 0.2727619 |
| meiji/cam2/evidence_gap/observed | 1642 | 31.90894 | 26.8867 | 29.80881 | 26.8867 | 31.90894 | 5.02224 |
| meiji/cam2/observed/observed | 7452 | 3.416084 | 3.446112 | 3.774621 | 3.416084 | 3.774621 | 0.358537 |
| meiji/evidence_gap/observed | 5077 | 30.29857 | 24.09872 | 28.00832 | 24.09872 | 30.29857 | 6.199848 |
| meiji/observed/observed | 23007 | 3.10859 | 3.108157 | 3.435645 | 3.108157 | 3.435645 | 0.3274885 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 25.53301 | 18.48645 | 23.69451 | 18.48645 | 25.53301 | 7.046558 |
| meiji/selection/cam0/observed/observed | 3604 | 3.451549 | 3.421782 | 3.759111 | 3.421782 | 3.759111 | 0.3373291 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 27.82075 | 21.38681 | 26.60101 | 21.38681 | 27.82075 | 6.43394 |
| meiji/selection/cam1/observed/observed | 3828 | 2.143421 | 2.24614 | 2.417311 | 2.143421 | 2.417311 | 0.2738903 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 26.96091 | 24.71054 | 27.78876 | 24.71054 | 27.78876 | 3.078222 |
| meiji/selection/cam2/observed/observed | 3507 | 3.053257 | 3.109331 | 3.415222 | 3.053257 | 3.415222 | 0.3619645 |
| meiji/selection/evidence_gap/observed | 2493 | 27.10016 | 21.70955 | 26.34753 | 21.70955 | 27.10016 | 5.390612 |
| meiji/selection/observed/observed | 10939 | 2.771717 | 2.826797 | 3.116511 | 2.771717 | 3.116511 | 0.3447933 |
| tracknet/evidence_gap/observed | 349 | 28.87838 | 24.8753 | 34.14414 | 24.8753 | 34.14414 | 9.268843 |
| tracknet/observed/observed | 1538 | 2.21854 | 2.180174 | 2.34239 | 2.180174 | 2.34239 | 0.1622153 |

## 倍率適用前: p90_error_px

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 212.3454 | 207.2498 | 278.3731 | 207.2498 | 278.3731 | 71.12327 |
| chat_annotation/observed/observed | 6622 | 94.34503 | 88.82453 | 114.7268 | 88.82453 | 114.7268 | 25.90222 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 209.1303 | 228.9999 | 238.6483 | 209.1303 | 238.6483 | 29.51793 |
| meiji/calibration/cam0/observed/observed | 3952 | 225.1095 | 249.0603 | 259.847 | 225.1095 | 259.847 | 34.73746 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 243.3954 | 243.5188 | 255.3124 | 243.3954 | 255.3124 | 11.91696 |
| meiji/calibration/cam1/observed/observed | 4171 | 275.4548 | 258.5239 | 274.4784 | 258.5239 | 275.4548 | 16.9309 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 183.9917 | 188.07 | 183.0198 | 183.0198 | 188.07 | 5.050219 |
| meiji/calibration/cam2/observed/observed | 3945 | 172.3436 | 157.9955 | 186.7188 | 157.9955 | 186.7188 | 28.7233 |
| meiji/calibration/evidence_gap/observed | 2584 | 206.2813 | 210.1231 | 225.7451 | 206.2813 | 225.7451 | 19.46388 |
| meiji/calibration/observed/observed | 12068 | 219.8596 | 223.9972 | 244.3104 | 219.8596 | 244.3104 | 24.45088 |
| meiji/cam0/evidence_gap/observed | 1697 | 184.5024 | 184.2912 | 173.9694 | 173.9694 | 184.5024 | 10.53291 |
| meiji/cam0/observed/observed | 7556 | 150.0675 | 160.4271 | 156.6568 | 150.0675 | 160.4271 | 10.35955 |
| meiji/cam1/evidence_gap/observed | 1738 | 146.5264 | 140.0197 | 159.12 | 140.0197 | 159.12 | 19.10028 |
| meiji/cam1/observed/observed | 7999 | 135.4792 | 120.0212 | 136.638 | 120.0212 | 136.638 | 16.61687 |
| meiji/cam2/evidence_gap/observed | 1642 | 139.8553 | 148.6881 | 137.9131 | 137.9131 | 148.6881 | 10.775 |
| meiji/cam2/observed/observed | 7452 | 108.2268 | 83.96506 | 96.01656 | 83.96506 | 108.2268 | 24.26173 |
| meiji/evidence_gap/observed | 5077 | 156.0403 | 155.5164 | 156.9472 | 155.5164 | 156.9472 | 1.43082 |
| meiji/observed/observed | 23007 | 133.7146 | 121.6999 | 134.264 | 121.6999 | 134.264 | 12.56412 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 140.4916 | 173.0195 | 102.1847 | 102.1847 | 173.0195 | 70.83482 |
| meiji/selection/cam0/observed/observed | 3604 | 93.49202 | 92.126 | 93.8944 | 92.126 | 93.8944 | 1.768404 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 84.0628 | 71.67464 | 101.2803 | 71.67464 | 101.2803 | 29.60568 |
| meiji/selection/cam1/observed/observed | 3828 | 15.33539 | 12.64124 | 14.21246 | 12.64124 | 15.33539 | 2.694149 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 85.77158 | 71.17901 | 71.27449 | 71.17901 | 85.77158 | 14.59258 |
| meiji/selection/cam2/observed/observed | 3507 | 44.86455 | 36.71617 | 38.71849 | 36.71617 | 44.86455 | 8.148373 |
| meiji/selection/evidence_gap/observed | 2493 | 102.4113 | 93.82879 | 92.73569 | 92.73569 | 102.4113 | 9.675568 |
| meiji/selection/observed/observed | 10939 | 56.9752 | 49.72712 | 54.33184 | 49.72712 | 56.9752 | 7.248082 |
| tracknet/evidence_gap/observed | 349 | 64.44217 | 55.03675 | 64.92156 | 55.03675 | 64.92156 | 9.884811 |
| tracknet/observed/observed | 1538 | 5.489628 | 5.426447 | 5.546021 | 5.426447 | 5.546021 | 0.1195742 |

## 倍率適用前: p95_error_px

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 368.2852 | 364.2551 | 413.9663 | 364.2551 | 413.9663 | 49.71119 |
| chat_annotation/observed/observed | 6622 | 210.9582 | 203.1308 | 279.2569 | 203.1308 | 279.2569 | 76.1261 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 360.9866 | 325.7501 | 423.0554 | 325.7501 | 423.0554 | 97.30536 |
| meiji/calibration/cam0/observed/observed | 3952 | 338.4369 | 371.365 | 379.7062 | 338.4369 | 379.7062 | 41.26927 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 385.1126 | 413.9549 | 424.5848 | 385.1126 | 424.5848 | 39.47219 |
| meiji/calibration/cam1/observed/observed | 4171 | 513.8349 | 527.5864 | 549.238 | 513.8349 | 549.238 | 35.40313 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 242.2515 | 275.4642 | 296.5148 | 242.2515 | 296.5148 | 54.26327 |
| meiji/calibration/cam2/observed/observed | 3945 | 293.5927 | 299.1639 | 327.5157 | 293.5927 | 327.5157 | 33.92298 |
| meiji/calibration/evidence_gap/observed | 2584 | 346.3454 | 329.1341 | 404.034 | 329.1341 | 404.034 | 74.89988 |
| meiji/calibration/observed/observed | 12068 | 377.6049 | 386.6316 | 413.6078 | 377.6049 | 413.6078 | 36.00289 |
| meiji/cam0/evidence_gap/observed | 1697 | 273.1659 | 246.2573 | 349.3467 | 246.2573 | 349.3467 | 103.0894 |
| meiji/cam0/observed/observed | 7556 | 266.7649 | 273.3103 | 279.2417 | 266.7649 | 279.2417 | 12.47676 |
| meiji/cam1/evidence_gap/observed | 1738 | 253.5863 | 248.7034 | 269.3404 | 248.7034 | 269.3404 | 20.63705 |
| meiji/cam1/observed/observed | 7999 | 337.5125 | 307.6418 | 348.127 | 307.6418 | 348.127 | 40.48524 |
| meiji/cam2/evidence_gap/observed | 1642 | 217.7836 | 209.7238 | 236.0021 | 209.7238 | 236.0021 | 26.27831 |
| meiji/cam2/observed/observed | 7452 | 230.3423 | 220.1784 | 262.3843 | 220.1784 | 262.3843 | 42.20582 |
| meiji/evidence_gap/observed | 5077 | 246.776 | 243.6115 | 272.5223 | 243.6115 | 272.5223 | 28.91081 |
| meiji/observed/observed | 23007 | 273.0216 | 274.3728 | 288.341 | 273.0216 | 288.341 | 15.31934 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 219.8762 | 215.8308 | 183.5356 | 183.5356 | 219.8762 | 36.34052 |
| meiji/selection/cam0/observed/observed | 3604 | 162.8136 | 176.7018 | 169.0178 | 162.8136 | 176.7018 | 13.88819 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 129.5889 | 128.6567 | 160.0165 | 128.6567 | 160.0165 | 31.35978 |
| meiji/selection/cam1/observed/observed | 3828 | 113.2155 | 85.1192 | 133.534 | 85.1192 | 133.534 | 48.41478 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 131.5526 | 118.2183 | 131.9438 | 118.2183 | 131.9438 | 13.7255 |
| meiji/selection/cam2/observed/observed | 3507 | 124.5659 | 84.03335 | 86.25029 | 84.03335 | 124.5659 | 40.53253 |
| meiji/selection/evidence_gap/observed | 2493 | 172.4191 | 175.6859 | 164.9468 | 164.9468 | 175.6859 | 10.73915 |
| meiji/selection/observed/observed | 10939 | 144.3507 | 132.1421 | 142.0496 | 132.1421 | 144.3507 | 12.2086 |
| tracknet/evidence_gap/observed | 349 | 74.84521 | 62.63691 | 92.31101 | 62.63691 | 92.31101 | 29.6741 |
| tracknet/observed/observed | 1538 | 8.485675 | 8.192533 | 8.395386 | 8.192533 | 8.485675 | 0.2931423 |

## 倍率適用前: mean_nll_px

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 10.9235 | 10.66857 | 10.91317 | 10.66857 | 10.9235 | 0.2549319 |
| chat_annotation/observed/observed | 6622 | 6.285514 | 6.50774 | 6.543121 | 6.285514 | 6.543121 | 0.2576072 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 10.16141 | 10.12388 | 10.65718 | 10.12388 | 10.65718 | 0.533306 |
| meiji/calibration/cam0/observed/observed | 3952 | 8.157344 | 9.006646 | 8.327537 | 8.157344 | 9.006646 | 0.8493024 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 10.71017 | 10.93808 | 10.9507 | 10.71017 | 10.9507 | 0.2405308 |
| meiji/calibration/cam1/observed/observed | 4171 | 6.933623 | 7.346012 | 7.08348 | 6.933623 | 7.346012 | 0.4123889 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 10.51751 | 10.12619 | 10.35838 | 10.12619 | 10.51751 | 0.3913184 |
| meiji/calibration/cam2/observed/observed | 3945 | 6.890952 | 7.118477 | 6.876583 | 6.876583 | 7.118477 | 0.2418946 |
| meiji/calibration/evidence_gap/observed | 2584 | 10.46188 | 10.3997 | 10.65979 | 10.3997 | 10.65979 | 0.2600884 |
| meiji/calibration/observed/observed | 12068 | 7.320415 | 7.815452 | 7.423247 | 7.320415 | 7.815452 | 0.4950367 |
| meiji/cam0/evidence_gap/observed | 1697 | 9.840702 | 9.730792 | 10.07686 | 9.730792 | 10.07686 | 0.3460682 |
| meiji/cam0/observed/observed | 7556 | 7.270616 | 7.823501 | 7.411171 | 7.270616 | 7.823501 | 0.5528846 |
| meiji/cam1/evidence_gap/observed | 1738 | 10.10778 | 10.08665 | 10.30754 | 10.08665 | 10.30754 | 0.2208851 |
| meiji/cam1/observed/observed | 7999 | 6.016087 | 6.29771 | 6.172299 | 6.016087 | 6.29771 | 0.2816228 |
| meiji/cam2/evidence_gap/observed | 1642 | 9.976926 | 9.688554 | 9.988949 | 9.688554 | 9.988949 | 0.3003947 |
| meiji/cam2/observed/observed | 7452 | 6.371607 | 6.531616 | 6.467525 | 6.371607 | 6.531616 | 0.1600086 |
| meiji/evidence_gap/observed | 5077 | 9.976188 | 9.838953 | 10.1274 | 9.838953 | 10.1274 | 0.2884428 |
| meiji/observed/observed | 23007 | 6.543255 | 6.874575 | 6.674796 | 6.543255 | 6.874575 | 0.3313201 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 9.498508 | 9.311373 | 9.45766 | 9.311373 | 9.498508 | 0.1871351 |
| meiji/selection/cam0/observed/observed | 3604 | 6.298267 | 6.526111 | 6.406321 | 6.298267 | 6.526111 | 0.2278448 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 9.499823 | 9.227349 | 9.658427 | 9.227349 | 9.658427 | 0.4310783 |
| meiji/selection/cam1/observed/observed | 3828 | 5.016336 | 5.155476 | 5.179474 | 5.016336 | 5.179474 | 0.1631377 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 9.417589 | 9.235735 | 9.606703 | 9.235735 | 9.606703 | 0.3709682 |
| meiji/selection/cam2/observed/observed | 3507 | 5.787399 | 5.871459 | 6.00738 | 5.787399 | 6.00738 | 0.2199804 |
| meiji/selection/evidence_gap/observed | 2493 | 9.47277 | 9.257734 | 9.575567 | 9.257734 | 9.575567 | 0.3178322 |
| meiji/selection/observed/observed | 10939 | 5.685885 | 5.836592 | 5.849098 | 5.685885 | 5.849098 | 0.1632133 |
| tracknet/evidence_gap/observed | 349 | 9.368357 | 8.843677 | 9.547883 | 8.843677 | 9.547883 | 0.7042061 |
| tracknet/observed/observed | 1538 | 4.409039 | 4.426782 | 4.476051 | 4.409039 | 4.476051 | 0.0670126 |

## 倍率適用前: coverage_0.5

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.4886818 | 0.60253 | 0.5672437 | 0.4886818 | 0.60253 | 0.1138482 |
| chat_annotation/observed/observed | 6622 | 0.5108728 | 0.4503171 | 0.448505 | 0.448505 | 0.5108728 | 0.06236786 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.446347 | 0.5239726 | 0.5159817 | 0.446347 | 0.5239726 | 0.07762557 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.4243421 | 0.3793016 | 0.4172571 | 0.3793016 | 0.4243421 | 0.04504049 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.5166094 | 0.5463918 | 0.5532646 | 0.5166094 | 0.5532646 | 0.03665521 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.5094702 | 0.4792616 | 0.4804603 | 0.4792616 | 0.5094702 | 0.03020858 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.4467066 | 0.5065868 | 0.5173653 | 0.4467066 | 0.5173653 | 0.07065868 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.5275032 | 0.4504436 | 0.4613435 | 0.4504436 | 0.5275032 | 0.07705957 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.4702012 | 0.5259288 | 0.5290248 | 0.4702012 | 0.5290248 | 0.05882353 |
| meiji/calibration/observed/observed | 12068 | 0.4874876 | 0.4371064 | 0.4535134 | 0.4371064 | 0.4874876 | 0.05038117 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.4855628 | 0.5562758 | 0.5521509 | 0.4855628 | 0.5562758 | 0.07071302 |
| meiji/cam0/observed/observed | 7556 | 0.4471943 | 0.4002118 | 0.4388565 | 0.4002118 | 0.4471943 | 0.04698253 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.5817031 | 0.6012658 | 0.5966628 | 0.5817031 | 0.6012658 | 0.01956272 |
| meiji/cam1/observed/observed | 7999 | 0.5535692 | 0.5059382 | 0.5145643 | 0.5059382 | 0.5535692 | 0.04763095 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.4926918 | 0.5383678 | 0.5371498 | 0.4926918 | 0.5383678 | 0.045676 |
| meiji/cam2/observed/observed | 7452 | 0.5437466 | 0.4582662 | 0.4714171 | 0.4582662 | 0.5437466 | 0.08548041 |
| meiji/evidence_gap/observed | 5077 | 0.52078 | 0.5658854 | 0.5625369 | 0.52078 | 0.5658854 | 0.04510538 |
| meiji/observed/observed | 23007 | 0.5154518 | 0.4557743 | 0.4757248 | 0.4557743 | 0.5154518 | 0.05967749 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.5274056 | 0.590743 | 0.590743 | 0.5274056 | 0.590743 | 0.06333739 |
| meiji/selection/cam0/observed/observed | 3604 | 0.4722531 | 0.423141 | 0.4625416 | 0.423141 | 0.4722531 | 0.0491121 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.6473988 | 0.6566474 | 0.6404624 | 0.6404624 | 0.6566474 | 0.01618497 |
| meiji/selection/cam1/observed/observed | 3828 | 0.6016196 | 0.5350052 | 0.5517241 | 0.5350052 | 0.6016196 | 0.06661442 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.5402726 | 0.5712515 | 0.5576208 | 0.5402726 | 0.5712515 | 0.03097893 |
| meiji/selection/cam2/observed/observed | 3507 | 0.5620188 | 0.4670659 | 0.4827488 | 0.4670659 | 0.5620188 | 0.09495295 |
| meiji/selection/evidence_gap/observed | 2493 | 0.573205 | 0.6073004 | 0.5972724 | 0.573205 | 0.6073004 | 0.03409547 |
| meiji/selection/observed/observed | 10939 | 0.5463022 | 0.476369 | 0.5002285 | 0.476369 | 0.5463022 | 0.06993327 |
| tracknet/evidence_gap/observed | 349 | 0.3467049 | 0.6561605 | 0.4555874 | 0.3467049 | 0.6561605 | 0.3094556 |
| tracknet/observed/observed | 1538 | 0.4915475 | 0.453186 | 0.4856957 | 0.453186 | 0.4915475 | 0.03836151 |

## 倍率適用前: coverage_0.9

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.9021305 | 0.924767 | 0.9320905 | 0.9021305 | 0.9320905 | 0.02996005 |
| chat_annotation/observed/observed | 6622 | 0.859408 | 0.8059499 | 0.8249773 | 0.8059499 | 0.859408 | 0.05345817 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8401826 | 0.7979452 | 0.7910959 | 0.7910959 | 0.8401826 | 0.04908676 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.7421559 | 0.7052126 | 0.7687247 | 0.7052126 | 0.7687247 | 0.06351215 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.8499427 | 0.8087056 | 0.8132875 | 0.8087056 | 0.8499427 | 0.04123711 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8209063 | 0.8093982 | 0.8014865 | 0.8014865 | 0.8209063 | 0.0194198 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8323353 | 0.8167665 | 0.8622754 | 0.8167665 | 0.8622754 | 0.04550898 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.8372624 | 0.7865653 | 0.8129278 | 0.7865653 | 0.8372624 | 0.05069708 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8409443 | 0.8076625 | 0.8215944 | 0.8076625 | 0.8409443 | 0.03328173 |
| meiji/calibration/observed/observed | 12068 | 0.800464 | 0.7678157 | 0.7944978 | 0.7678157 | 0.800464 | 0.03264833 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.888627 | 0.8444313 | 0.8638774 | 0.8444313 | 0.888627 | 0.04419564 |
| meiji/cam0/observed/observed | 7556 | 0.7885124 | 0.7406035 | 0.8073055 | 0.7406035 | 0.8073055 | 0.06670196 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.8964327 | 0.8670886 | 0.867664 | 0.8670886 | 0.8964327 | 0.02934407 |
| meiji/cam1/observed/observed | 7999 | 0.8623578 | 0.8409801 | 0.8382298 | 0.8382298 | 0.8623578 | 0.02412802 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.8891596 | 0.8568819 | 0.9062119 | 0.8568819 | 0.9062119 | 0.04933009 |
| meiji/cam2/observed/observed | 7452 | 0.8560118 | 0.8106549 | 0.8301127 | 0.8106549 | 0.8560118 | 0.04535695 |
| meiji/evidence_gap/observed | 5077 | 0.8914713 | 0.8562143 | 0.8788655 | 0.8562143 | 0.8914713 | 0.03525704 |
| meiji/observed/observed | 23007 | 0.8360499 | 0.7981919 | 0.8254444 | 0.7981919 | 0.8360499 | 0.03785804 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9403167 | 0.8940317 | 0.9415347 | 0.8940317 | 0.9415347 | 0.04750305 |
| meiji/selection/cam0/observed/observed | 3604 | 0.8393452 | 0.7794118 | 0.8496115 | 0.7794118 | 0.8496115 | 0.07019978 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9433526 | 0.9260116 | 0.9225434 | 0.9225434 | 0.9433526 | 0.02080925 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9075235 | 0.8753918 | 0.8782654 | 0.8753918 | 0.9075235 | 0.03213166 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.9479554 | 0.8983891 | 0.9516729 | 0.8983891 | 0.9516729 | 0.05328377 |
| meiji/selection/cam2/observed/observed | 3507 | 0.8771029 | 0.8377531 | 0.849444 | 0.8377531 | 0.8771029 | 0.03934987 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9438428 | 0.9065383 | 0.938227 | 0.9065383 | 0.9438428 | 0.03730445 |
| meiji/selection/observed/observed | 10939 | 0.8753085 | 0.8317031 | 0.859585 | 0.8317031 | 0.8753085 | 0.04360545 |
| tracknet/evidence_gap/observed | 349 | 0.8997135 | 1 | 0.9255014 | 0.8997135 | 1 | 0.1002865 |
| tracknet/observed/observed | 1538 | 0.8498049 | 0.8257477 | 0.8595579 | 0.8257477 | 0.8595579 | 0.03381014 |

## 倍率適用前: coverage_0.95

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.9480692 | 0.9520639 | 0.9613848 | 0.9480692 | 0.9613848 | 0.01331558 |
| chat_annotation/observed/observed | 6622 | 0.91045 | 0.8663546 | 0.8817578 | 0.8663546 | 0.91045 | 0.04409544 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.880137 | 0.8333333 | 0.8310502 | 0.8310502 | 0.880137 | 0.04908676 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.8011134 | 0.7651822 | 0.8302126 | 0.7651822 | 0.8302126 | 0.06503036 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.8831615 | 0.838488 | 0.8339061 | 0.8339061 | 0.8831615 | 0.04925544 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8650204 | 0.8551906 | 0.8503956 | 0.8503956 | 0.8650204 | 0.01462479 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8778443 | 0.8694611 | 0.902994 | 0.8694611 | 0.902994 | 0.03353293 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.8750317 | 0.8423321 | 0.8651458 | 0.8423321 | 0.8750317 | 0.03269962 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.880418 | 0.8467492 | 0.8552632 | 0.8467492 | 0.880418 | 0.03366873 |
| meiji/calibration/observed/observed | 12068 | 0.8473649 | 0.8215114 | 0.8486079 | 0.8215114 | 0.8486079 | 0.02709645 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.9233942 | 0.8744844 | 0.8980554 | 0.8744844 | 0.9233942 | 0.04890984 |
| meiji/cam0/observed/observed | 7556 | 0.8483325 | 0.8004235 | 0.8705664 | 0.8004235 | 0.8705664 | 0.07014293 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.9188723 | 0.8906789 | 0.8895282 | 0.8895282 | 0.9188723 | 0.02934407 |
| meiji/cam1/observed/observed | 7999 | 0.9018627 | 0.8857357 | 0.8871109 | 0.8857357 | 0.9018627 | 0.01612702 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.9257004 | 0.9135201 | 0.9378806 | 0.9135201 | 0.9378806 | 0.02436054 |
| meiji/cam2/observed/observed | 7452 | 0.8969404 | 0.8671498 | 0.8851315 | 0.8671498 | 0.8969404 | 0.02979066 |
| meiji/evidence_gap/observed | 5077 | 0.9225921 | 0.8926531 | 0.9080165 | 0.8926531 | 0.9225921 | 0.02993894 |
| meiji/observed/observed | 23007 | 0.8826879 | 0.8516973 | 0.8810362 | 0.8516973 | 0.8826879 | 0.03099057 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9695493 | 0.9183922 | 0.9695493 | 0.9183922 | 0.9695493 | 0.05115713 |
| meiji/selection/cam0/observed/observed | 3604 | 0.900111 | 0.8390677 | 0.9148169 | 0.8390677 | 0.9148169 | 0.07574917 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9549133 | 0.9433526 | 0.9456647 | 0.9433526 | 0.9549133 | 0.01156069 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9420063 | 0.9190178 | 0.927116 | 0.9190178 | 0.9420063 | 0.02298851 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.9752169 | 0.9591078 | 0.9739777 | 0.9591078 | 0.9752169 | 0.01610905 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9215854 | 0.895067 | 0.9076133 | 0.895067 | 0.9215854 | 0.02651839 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9663057 | 0.9402327 | 0.9626955 | 0.9402327 | 0.9663057 | 0.026073 |
| meiji/selection/observed/observed | 10939 | 0.9216565 | 0.8849986 | 0.9168114 | 0.8849986 | 0.9216565 | 0.03665783 |
| tracknet/evidence_gap/observed | 349 | 0.965616 | 1 | 0.9570201 | 0.9570201 | 1 | 0.04297994 |
| tracknet/observed/observed | 1538 | 0.9037711 | 0.8816645 | 0.9109233 | 0.8816645 | 0.9109233 | 0.02925878 |

## 倍率適用前: area_px2_0.5

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 39918.95 | 33631.73 | 44012.65 | 33631.73 | 44012.65 | 10380.92 |
| chat_annotation/observed/observed | 6622 | 6976.279 | 4349.005 | 4195.512 | 4195.512 | 6976.279 | 2780.767 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 5832.479 | 3594.665 | 6005.882 | 3594.665 | 6005.882 | 2411.216 |
| meiji/calibration/cam0/observed/observed | 3952 | 948.9092 | 523.1586 | 698.2183 | 523.1586 | 948.9092 | 425.7506 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 8688.594 | 4852.252 | 7139.182 | 4852.252 | 8688.594 | 3836.341 |
| meiji/calibration/cam1/observed/observed | 4171 | 1039.52 | 750.8177 | 419.3729 | 419.3729 | 1039.52 | 620.1473 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 11654.2 | 7303.855 | 10604.33 | 7303.855 | 11654.2 | 4350.341 |
| meiji/calibration/cam2/observed/observed | 3945 | 2461.913 | 1478.069 | 1187.631 | 1187.631 | 2461.913 | 1274.281 |
| meiji/calibration/evidence_gap/observed | 2584 | 8678.656 | 5218.135 | 7874.721 | 5218.135 | 8678.656 | 3460.521 |
| meiji/calibration/observed/observed | 12068 | 1474.824 | 914.0012 | 761.8303 | 761.8303 | 1474.824 | 712.9934 |
| meiji/cam0/evidence_gap/observed | 1697 | 7380.447 | 4878.829 | 7343.136 | 4878.829 | 7380.447 | 2501.618 |
| meiji/cam0/observed/observed | 7556 | 983.1528 | 540.3571 | 733.4856 | 540.3571 | 983.1528 | 442.7957 |
| meiji/cam1/evidence_gap/observed | 1738 | 7819.453 | 4715.368 | 6735.475 | 4715.368 | 7819.453 | 3104.084 |
| meiji/cam1/observed/observed | 7999 | 956.4311 | 736.8014 | 550.13 | 550.13 | 956.4311 | 406.3011 |
| meiji/cam2/evidence_gap/observed | 1642 | 9760.335 | 7009.177 | 8407.321 | 7009.177 | 9760.335 | 2751.158 |
| meiji/cam2/observed/observed | 7452 | 1922.016 | 1199.938 | 889.6189 | 889.6189 | 1922.016 | 1032.397 |
| meiji/evidence_gap/observed | 5077 | 8300.433 | 5511.868 | 7479.295 | 5511.868 | 8300.433 | 2788.565 |
| meiji/observed/observed | 23007 | 1277.961 | 822.2953 | 720.3089 | 720.3089 | 1277.961 | 557.6526 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 9032.116 | 6249.02 | 8769.976 | 6249.02 | 9032.116 | 2783.095 |
| meiji/selection/cam0/observed/observed | 3604 | 1020.703 | 559.2163 | 772.1582 | 559.2163 | 1020.703 | 461.4866 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 6942.273 | 4577.218 | 6328.034 | 4577.218 | 6942.273 | 2365.055 |
| meiji/selection/cam1/observed/observed | 3828 | 865.897 | 721.5291 | 692.6034 | 692.6034 | 865.897 | 173.2936 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 7800.764 | 6704.275 | 6134.079 | 6134.079 | 7800.764 | 1666.685 |
| meiji/selection/cam2/observed/observed | 3507 | 1314.691 | 887.069 | 554.3868 | 554.3868 | 1314.691 | 760.3037 |
| meiji/selection/evidence_gap/observed | 2493 | 7908.403 | 5816.322 | 7069.435 | 5816.322 | 7908.403 | 2092.082 |
| meiji/selection/observed/observed | 10939 | 1060.781 | 721.1244 | 674.5021 | 674.5021 | 1060.781 | 386.2792 |
| tracknet/evidence_gap/observed | 349 | 2666.131 | 3232.83 | 3685.239 | 2666.131 | 3685.239 | 1019.108 |
| tracknet/observed/observed | 1538 | 31.32259 | 27.5951 | 29.7844 | 27.5951 | 31.32259 | 3.72749 |

## 倍率適用前: area_px2_0.9

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 155845.9 | 132798.9 | 184011.6 | 132798.9 | 184011.6 | 51212.63 |
| chat_annotation/observed/observed | 6622 | 37876.51 | 22643.46 | 27890.49 | 22643.46 | 37876.51 | 15233.05 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 30739.66 | 16480.88 | 28368.57 | 16480.88 | 30739.66 | 14258.78 |
| meiji/calibration/cam0/observed/observed | 3952 | 7286.825 | 3803.817 | 6681.971 | 3803.817 | 7286.825 | 3483.008 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 40109.18 | 21138.43 | 33499.23 | 21138.43 | 40109.18 | 18970.75 |
| meiji/calibration/cam1/observed/observed | 4171 | 10903.74 | 6654.653 | 7130.896 | 6654.653 | 10903.74 | 4249.088 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 49685.19 | 33636.85 | 47426.6 | 33636.85 | 49685.19 | 16048.34 |
| meiji/calibration/cam2/observed/observed | 3945 | 14855.58 | 11542.56 | 11900.1 | 11542.56 | 14855.58 | 3313.024 |
| meiji/calibration/evidence_gap/observed | 2584 | 40027.24 | 23598.25 | 36260.41 | 23598.25 | 40027.24 | 16428.99 |
| meiji/calibration/observed/observed | 12068 | 11011.13 | 7318.911 | 8542.923 | 7318.911 | 11011.13 | 3692.217 |
| meiji/cam0/evidence_gap/observed | 1697 | 36806.73 | 20880.96 | 35881.55 | 20880.96 | 36806.73 | 15925.77 |
| meiji/cam0/observed/observed | 7556 | 8752.583 | 4128.26 | 7323.527 | 4128.26 | 8752.583 | 4624.323 |
| meiji/cam1/evidence_gap/observed | 1738 | 35255.91 | 20272.15 | 30607.35 | 20272.15 | 35255.91 | 14983.76 |
| meiji/cam1/observed/observed | 7999 | 9313.294 | 5682.566 | 6197.28 | 5682.566 | 9313.294 | 3630.728 |
| meiji/cam2/evidence_gap/observed | 1642 | 42934.85 | 30839.01 | 39045.99 | 30839.01 | 42934.85 | 12095.85 |
| meiji/cam2/observed/observed | 7452 | 12290.14 | 9861.076 | 8904.986 | 8904.986 | 12290.14 | 3385.151 |
| meiji/evidence_gap/observed | 5077 | 38257.79 | 23893.17 | 35099.48 | 23893.17 | 38257.79 | 14364.62 |
| meiji/observed/observed | 23007 | 10093.35 | 6525.523 | 7444.194 | 6525.523 | 10093.35 | 3567.824 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 43280.24 | 25575.8 | 43897.85 | 25575.8 | 43897.85 | 18322.05 |
| meiji/selection/cam0/observed/observed | 3604 | 10359.87 | 4484.031 | 8027.032 | 4484.031 | 10359.87 | 5875.842 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 30357.76 | 19397.86 | 27688.72 | 19397.86 | 30357.76 | 10959.9 |
| meiji/selection/cam1/observed/observed | 3828 | 7580.338 | 4623.378 | 5180.009 | 4623.378 | 7580.338 | 2956.96 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 35950.3 | 27944.09 | 30374.59 | 27944.09 | 35950.3 | 8006.214 |
| meiji/selection/cam2/observed/observed | 3507 | 9404.288 | 7969.591 | 5535.808 | 5535.808 | 9404.288 | 3868.481 |
| meiji/selection/evidence_gap/observed | 2493 | 36423.76 | 24198.86 | 33896.18 | 24198.86 | 36423.76 | 12224.9 |
| meiji/selection/observed/observed | 10939 | 9080.845 | 5650.251 | 6232.066 | 5650.251 | 9080.845 | 3430.594 |
| tracknet/evidence_gap/observed | 349 | 10973.36 | 12222.83 | 15446.6 | 10973.36 | 15446.6 | 4473.239 |
| tracknet/observed/observed | 1538 | 196.4473 | 129.9065 | 167.4309 | 129.9065 | 196.4473 | 66.54082 |

## 倍率適用前: area_px2_0.95

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 209887.2 | 179709.7 | 254654 | 179709.7 | 254654 | 74944.28 |
| chat_annotation/observed/observed | 6622 | 53342.65 | 31909.64 | 40828.77 | 31909.64 | 53342.65 | 21433.01 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 44844.35 | 23362.9 | 41196.19 | 23362.9 | 44844.35 | 21481.45 |
| meiji/calibration/cam0/observed/observed | 3952 | 11127.92 | 5812.263 | 10621.18 | 5812.263 | 11127.92 | 5315.654 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 56631.29 | 29879.49 | 48252.21 | 29879.49 | 56631.29 | 26751.8 |
| meiji/calibration/cam1/observed/observed | 4171 | 16802.07 | 10261.68 | 11769.64 | 10261.68 | 16802.07 | 6540.384 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 68821.33 | 46892.09 | 66429.56 | 46892.09 | 68821.33 | 21929.23 |
| meiji/calibration/cam2/observed/observed | 3945 | 21492.26 | 16864.43 | 18324.1 | 16864.43 | 21492.26 | 4627.836 |
| meiji/calibration/evidence_gap/observed | 2584 | 56574.53 | 33167.8 | 51734.03 | 33167.8 | 56574.53 | 23406.73 |
| meiji/calibration/observed/observed | 12068 | 16477.12 | 10963.02 | 13536.18 | 10963.02 | 16477.12 | 5514.105 |
| meiji/cam0/evidence_gap/observed | 1697 | 52215.38 | 28912.49 | 51384.27 | 28912.49 | 52215.38 | 23302.9 |
| meiji/cam0/observed/observed | 7556 | 13352.46 | 6283.327 | 11492.48 | 6283.327 | 13352.46 | 7069.129 |
| meiji/cam1/evidence_gap/observed | 1738 | 49906.18 | 28561.08 | 44015.75 | 28561.08 | 49906.18 | 21345.1 |
| meiji/cam1/observed/observed | 7999 | 14270.37 | 8610.099 | 9944.074 | 8610.099 | 14270.37 | 5660.273 |
| meiji/cam2/evidence_gap/observed | 1642 | 59660.13 | 42697.87 | 55053.23 | 42697.87 | 59660.13 | 16962.25 |
| meiji/cam2/observed/observed | 7452 | 18056.98 | 14658.72 | 14039.04 | 14039.04 | 18056.98 | 4017.934 |
| meiji/evidence_gap/observed | 5077 | 53832.65 | 33250.65 | 50048.43 | 33250.65 | 53832.65 | 20582 |
| meiji/observed/observed | 23007 | 15195.4 | 9805.094 | 11778.97 | 9805.094 | 15195.4 | 5390.301 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 60080.21 | 34833.84 | 62254.87 | 34833.84 | 62254.87 | 27421.03 |
| meiji/selection/cam0/observed/observed | 3604 | 15791.8 | 6799.877 | 12447.91 | 6799.877 | 15791.8 | 8991.918 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 43118.87 | 27230.47 | 39740.1 | 27230.47 | 43118.87 | 15888.4 |
| meiji/selection/cam1/observed/observed | 3828 | 11511.83 | 6810.528 | 7954.932 | 6810.528 | 11511.83 | 4701.302 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 50181.07 | 38358.13 | 43282.18 | 38358.13 | 50181.07 | 11822.94 |
| meiji/selection/cam2/observed/observed | 3507 | 14192.65 | 12177.54 | 9218.809 | 9218.809 | 14192.65 | 4973.84 |
| meiji/selection/evidence_gap/observed | 2493 | 50990.69 | 33336.52 | 48301.31 | 33336.52 | 50990.69 | 17654.17 |
| meiji/selection/observed/observed | 10939 | 13781.38 | 8527.662 | 9840.397 | 8527.662 | 13781.38 | 5253.72 |
| tracknet/evidence_gap/observed | 349 | 15710.49 | 16975.17 | 23749.39 | 15710.49 | 23749.39 | 8038.897 |
| tracknet/observed/observed | 1538 | 294.5553 | 186.3784 | 310.407 | 186.3784 | 310.407 | 124.0286 |

## 固定倍率適用後: mean_nll_px

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 11.02831 | 10.85134 | 11.05829 | 10.85134 | 11.05829 | 0.2069474 |
| chat_annotation/observed/observed | 6622 | 6.322052 | 6.395475 | 6.482362 | 6.322052 | 6.482362 | 0.16031 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 10.17176 | 9.991258 | 10.43383 | 9.991258 | 10.43383 | 0.4425719 |
| meiji/calibration/cam0/observed/observed | 3952 | 7.715436 | 8.085807 | 7.822882 | 7.715436 | 8.085807 | 0.3703709 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 10.72837 | 10.76148 | 10.86474 | 10.72837 | 10.86474 | 0.1363755 |
| meiji/calibration/cam1/observed/observed | 4171 | 6.619564 | 6.765993 | 6.721757 | 6.619564 | 6.765993 | 0.1464284 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 10.52482 | 10.21782 | 10.41088 | 10.21782 | 10.52482 | 0.3069961 |
| meiji/calibration/cam2/observed/observed | 3945 | 6.790794 | 6.854985 | 6.783658 | 6.783658 | 6.854985 | 0.07132764 |
| meiji/calibration/evidence_gap/observed | 2584 | 10.4739 | 10.32469 | 10.572 | 10.32469 | 10.572 | 0.2473101 |
| meiji/calibration/observed/observed | 12068 | 7.034412 | 7.227294 | 7.102586 | 7.034412 | 7.227294 | 0.1928812 |
| meiji/cam0/evidence_gap/observed | 1697 | 9.913039 | 9.714019 | 10.03907 | 9.714019 | 10.03907 | 0.3250538 |
| meiji/cam0/observed/observed | 7556 | 7.046573 | 7.260084 | 7.154732 | 7.046573 | 7.260084 | 0.2135111 |
| meiji/cam1/evidence_gap/observed | 1738 | 10.23413 | 10.09086 | 10.34913 | 10.09086 | 10.34913 | 0.2582721 |
| meiji/cam1/observed/observed | 7999 | 5.904504 | 5.991011 | 6.016673 | 5.904504 | 6.016673 | 0.1121691 |
| meiji/cam2/evidence_gap/observed | 1642 | 10.06343 | 9.807338 | 10.06804 | 9.807338 | 10.06804 | 0.2607023 |
| meiji/cam2/observed/observed | 7452 | 6.340319 | 6.355217 | 6.398961 | 6.340319 | 6.398961 | 0.05864196 |
| meiji/evidence_gap/observed | 5077 | 10.0716 | 9.873204 | 10.15458 | 9.873204 | 10.15458 | 0.28138 |
| meiji/observed/observed | 23007 | 6.420746 | 6.525769 | 6.51426 | 6.420746 | 6.525769 | 0.1050236 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 9.636987 | 9.418206 | 9.617869 | 9.418206 | 9.636987 | 0.2187805 |
| meiji/selection/cam0/observed/observed | 3604 | 6.313126 | 6.354631 | 6.422066 | 6.313126 | 6.422066 | 0.1089402 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 9.735321 | 9.414044 | 9.828758 | 9.414044 | 9.828758 | 0.4147131 |
| meiji/selection/cam1/observed/observed | 3828 | 5.125372 | 5.146589 | 5.248411 | 5.125372 | 5.248411 | 0.1230394 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 9.586043 | 9.382615 | 9.713305 | 9.382615 | 9.713305 | 0.3306903 |
| meiji/selection/cam2/observed/observed | 3507 | 5.833583 | 5.79303 | 5.966219 | 5.79303 | 5.966219 | 0.1731884 |
| meiji/selection/evidence_gap/observed | 2493 | 9.654615 | 9.405241 | 9.721935 | 9.405241 | 9.721935 | 0.3166936 |
| meiji/selection/observed/observed | 10939 | 5.743743 | 5.751841 | 5.865214 | 5.743743 | 5.865214 | 0.1214709 |
| tracknet/evidence_gap/observed | 349 | 9.389859 | 9.140424 | 9.571357 | 9.140424 | 9.571357 | 0.430933 |
| tracknet/observed/observed | 1538 | 4.465109 | 4.43068 | 4.538904 | 4.43068 | 4.538904 | 0.1082231 |

## 固定倍率適用後: coverage_0.5

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.6617843 | 0.7576565 | 0.7703063 | 0.6617843 | 0.7703063 | 0.108522 |
| chat_annotation/observed/observed | 6622 | 0.6890667 | 0.6182422 | 0.6220175 | 0.6182422 | 0.6890667 | 0.07082452 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.586758 | 0.6461187 | 0.6598174 | 0.586758 | 0.6598174 | 0.07305936 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.5617409 | 0.5159413 | 0.5650304 | 0.5159413 | 0.5650304 | 0.04908907 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.674685 | 0.6517755 | 0.6884307 | 0.6517755 | 0.6884307 | 0.03665521 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.6708223 | 0.6382163 | 0.6343802 | 0.6343802 | 0.6708223 | 0.0364421 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.6083832 | 0.6814371 | 0.7125749 | 0.6083832 | 0.7125749 | 0.1041916 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.6775665 | 0.6131812 | 0.6220532 | 0.6131812 | 0.6775665 | 0.0643853 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.623452 | 0.6594427 | 0.6865325 | 0.623452 | 0.6865325 | 0.0630805 |
| meiji/calibration/observed/observed | 12068 | 0.6373053 | 0.5899901 | 0.60764 | 0.5899901 | 0.6373053 | 0.04731521 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.6599882 | 0.6882734 | 0.7218621 | 0.6599882 | 0.7218621 | 0.0618739 |
| meiji/cam0/observed/observed | 7556 | 0.5972737 | 0.5422181 | 0.5952885 | 0.5422181 | 0.5972737 | 0.05505558 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.7451093 | 0.7243959 | 0.735328 | 0.7243959 | 0.7451093 | 0.02071346 |
| meiji/cam1/observed/observed | 7999 | 0.7168396 | 0.6743343 | 0.6773347 | 0.6743343 | 0.7168396 | 0.04250531 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.6991474 | 0.7180268 | 0.7521315 | 0.6991474 | 0.7521315 | 0.05298417 |
| meiji/cam2/observed/observed | 7452 | 0.6957864 | 0.6296296 | 0.6387547 | 0.6296296 | 0.6957864 | 0.06615674 |
| meiji/evidence_gap/observed | 5077 | 0.7017924 | 0.710262 | 0.7362616 | 0.7017924 | 0.7362616 | 0.03446917 |
| meiji/observed/observed | 23007 | 0.6707524 | 0.6164646 | 0.6378928 | 0.6164646 | 0.6707524 | 0.05428783 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.7381242 | 0.7332521 | 0.7880633 | 0.7332521 | 0.7880633 | 0.05481121 |
| meiji/selection/cam0/observed/observed | 3604 | 0.6362375 | 0.5710322 | 0.6284684 | 0.5710322 | 0.6362375 | 0.06520533 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.816185 | 0.7976879 | 0.782659 | 0.782659 | 0.816185 | 0.03352601 |
| meiji/selection/cam1/observed/observed | 3828 | 0.7669801 | 0.7136886 | 0.7241379 | 0.7136886 | 0.7669801 | 0.05329154 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.7930607 | 0.755886 | 0.7930607 | 0.755886 | 0.7930607 | 0.03717472 |
| meiji/selection/cam2/observed/observed | 3507 | 0.7162817 | 0.6481323 | 0.6575421 | 0.6481323 | 0.7162817 | 0.06814942 |
| meiji/selection/evidence_gap/observed | 2493 | 0.7829924 | 0.7629362 | 0.7878059 | 0.7629362 | 0.7878059 | 0.02486963 |
| meiji/selection/observed/observed | 10939 | 0.7076515 | 0.6456715 | 0.6712679 | 0.6456715 | 0.7076515 | 0.06198007 |
| tracknet/evidence_gap/observed | 349 | 0.5816619 | 0.8853868 | 0.6618911 | 0.5816619 | 0.8853868 | 0.3037249 |
| tracknet/observed/observed | 1538 | 0.6911573 | 0.6495449 | 0.6710013 | 0.6495449 | 0.6911573 | 0.04161248 |

## 固定倍率適用後: coverage_0.9

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.9687084 | 0.9667111 | 0.9727031 | 0.9667111 | 0.9727031 | 0.005992011 |
| chat_annotation/observed/observed | 6622 | 0.9367261 | 0.9084869 | 0.9220779 | 0.9084869 | 0.9367261 | 0.0282392 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.8984018 | 0.8755708 | 0.8607306 | 0.8607306 | 0.8984018 | 0.03767123 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.8216093 | 0.7897267 | 0.8446356 | 0.7897267 | 0.8446356 | 0.05490891 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.8934708 | 0.8453608 | 0.8487973 | 0.8453608 | 0.8934708 | 0.04810997 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.8762887 | 0.8659794 | 0.8621434 | 0.8621434 | 0.8762887 | 0.01414529 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.8898204 | 0.8862275 | 0.9125749 | 0.8862275 | 0.9125749 | 0.02634731 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.8887199 | 0.8679341 | 0.8836502 | 0.8679341 | 0.8887199 | 0.0207858 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.8939628 | 0.868808 | 0.873452 | 0.868808 | 0.8939628 | 0.0251548 |
| meiji/calibration/observed/observed | 12068 | 0.8624461 | 0.8416473 | 0.8634405 | 0.8416473 | 0.8634405 | 0.02179317 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.9351797 | 0.9021803 | 0.9169122 | 0.9021803 | 0.9351797 | 0.03299941 |
| meiji/cam0/observed/observed | 7556 | 0.865405 | 0.8267602 | 0.8839333 | 0.8267602 | 0.8839333 | 0.05717311 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.9263521 | 0.8975834 | 0.901611 | 0.8975834 | 0.9263521 | 0.0287687 |
| meiji/cam1/observed/observed | 7999 | 0.9162395 | 0.9024878 | 0.9019877 | 0.9019877 | 0.9162395 | 0.01425178 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.9360536 | 0.9293544 | 0.9451888 | 0.9293544 | 0.9451888 | 0.01583435 |
| meiji/cam2/observed/observed | 7452 | 0.9146538 | 0.8972088 | 0.9072732 | 0.8972088 | 0.9146538 | 0.01744498 |
| meiji/evidence_gap/observed | 5077 | 0.9324404 | 0.9093953 | 0.9208194 | 0.9093953 | 0.9324404 | 0.02304511 |
| meiji/observed/observed | 23007 | 0.8990307 | 0.8759073 | 0.8977702 | 0.8759073 | 0.8990307 | 0.0231234 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9744214 | 0.9305725 | 0.9768575 | 0.9305725 | 0.9768575 | 0.04628502 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9134295 | 0.8673696 | 0.9270255 | 0.8673696 | 0.9270255 | 0.05965594 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9595376 | 0.950289 | 0.9549133 | 0.950289 | 0.9595376 | 0.009248555 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9597701 | 0.9422675 | 0.9454023 | 0.9422675 | 0.9597701 | 0.01750261 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.983891 | 0.9739777 | 0.9789343 | 0.9739777 | 0.983891 | 0.009913259 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9438266 | 0.9301397 | 0.9338466 | 0.9301397 | 0.9438266 | 0.01368691 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9723225 | 0.9514641 | 0.9699158 | 0.9514641 | 0.9723225 | 0.0208584 |
| meiji/selection/observed/observed | 10939 | 0.9393912 | 0.9137033 | 0.9356431 | 0.9137033 | 0.9393912 | 0.02568791 |
| tracknet/evidence_gap/observed | 349 | 0.9856734 | 1 | 0.9684814 | 0.9684814 | 1 | 0.03151862 |
| tracknet/observed/observed | 1538 | 0.9505852 | 0.9401821 | 0.9557867 | 0.9401821 | 0.9557867 | 0.01560468 |

## 固定倍率適用後: coverage_0.95

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 0.9840213 | 0.9786951 | 0.9793609 | 0.9786951 | 0.9840213 | 0.005326232 |
| chat_annotation/observed/observed | 6622 | 0.9565086 | 0.9376321 | 0.9503171 | 0.9376321 | 0.9565086 | 0.01887647 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 0.9360731 | 0.8961187 | 0.8949772 | 0.8949772 | 0.9360731 | 0.04109589 |
| meiji/calibration/cam0/observed/observed | 3952 | 0.8623482 | 0.8370445 | 0.8904352 | 0.8370445 | 0.8904352 | 0.05339069 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 0.9117984 | 0.8568156 | 0.8774341 | 0.8568156 | 0.9117984 | 0.05498282 |
| meiji/calibration/cam1/observed/observed | 4171 | 0.899065 | 0.893311 | 0.8901942 | 0.8901942 | 0.899065 | 0.008870774 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 0.908982 | 0.9221557 | 0.9221557 | 0.908982 | 0.9221557 | 0.01317365 |
| meiji/calibration/cam2/observed/observed | 3945 | 0.9102662 | 0.8912548 | 0.9084918 | 0.8912548 | 0.9102662 | 0.01901141 |
| meiji/calibration/evidence_gap/observed | 2584 | 0.9191176 | 0.8912539 | 0.8978328 | 0.8912539 | 0.9191176 | 0.02786378 |
| meiji/calibration/observed/observed | 12068 | 0.8907027 | 0.8742128 | 0.8962546 | 0.8742128 | 0.8962546 | 0.02204176 |
| meiji/cam0/evidence_gap/observed | 1697 | 0.9646435 | 0.9210371 | 0.9369476 | 0.9210371 | 0.9646435 | 0.04360636 |
| meiji/cam0/observed/observed | 7556 | 0.903388 | 0.8738751 | 0.9242986 | 0.8738751 | 0.9242986 | 0.0504235 |
| meiji/cam1/evidence_gap/observed | 1738 | 0.9447641 | 0.9125432 | 0.9228999 | 0.9125432 | 0.9447641 | 0.03222094 |
| meiji/cam1/observed/observed | 7999 | 0.9342418 | 0.9251156 | 0.9256157 | 0.9251156 | 0.9342418 | 0.009126141 |
| meiji/cam2/evidence_gap/observed | 1642 | 0.9470158 | 0.953715 | 0.9512789 | 0.9470158 | 0.953715 | 0.006699147 |
| meiji/cam2/observed/observed | 7452 | 0.9365271 | 0.9231079 | 0.9342458 | 0.9231079 | 0.9365271 | 0.01341922 |
| meiji/evidence_gap/observed | 5077 | 0.9521371 | 0.9286981 | 0.9367737 | 0.9286981 | 0.9521371 | 0.02343904 |
| meiji/observed/observed | 23007 | 0.924849 | 0.9076368 | 0.9279784 | 0.9076368 | 0.9279784 | 0.02034164 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 0.9951279 | 0.9476248 | 0.9817296 | 0.9476248 | 0.9951279 | 0.04750305 |
| meiji/selection/cam0/observed/observed | 3604 | 0.9483907 | 0.9142619 | 0.9614317 | 0.9142619 | 0.9614317 | 0.04716981 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 0.9780347 | 0.9687861 | 0.9687861 | 0.9687861 | 0.9780347 | 0.009248555 |
| meiji/selection/cam1/observed/observed | 3828 | 0.9725705 | 0.9597701 | 0.9642111 | 0.9597701 | 0.9725705 | 0.01280042 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 0.9863693 | 0.9863693 | 0.9814126 | 0.9814126 | 0.9863693 | 0.004956629 |
| meiji/selection/cam2/observed/observed | 3507 | 0.9660679 | 0.9589393 | 0.9632164 | 0.9589393 | 0.9660679 | 0.0071286 |
| meiji/selection/evidence_gap/observed | 2493 | 0.9863618 | 0.967509 | 0.977136 | 0.967509 | 0.9863618 | 0.01885279 |
| meiji/selection/observed/observed | 10939 | 0.9625194 | 0.9445105 | 0.9629765 | 0.9445105 | 0.9629765 | 0.01846604 |
| tracknet/evidence_gap/observed | 349 | 1 | 1 | 0.982808 | 0.982808 | 1 | 0.01719198 |
| tracknet/observed/observed | 1538 | 0.9720416 | 0.9629389 | 0.9811443 | 0.9629389 | 0.9811443 | 0.01820546 |

## 固定倍率適用後: area_px2_0.5

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 71490.31 | 59974.69 | 78307.26 | 59974.69 | 78307.26 | 18332.57 |
| chat_annotation/observed/observed | 6622 | 12405.48 | 7720.683 | 7413.672 | 7413.672 | 12405.48 | 4991.811 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 10321.61 | 6341.058 | 10595.39 | 6341.058 | 10595.39 | 4254.336 |
| meiji/calibration/cam0/observed/observed | 3952 | 1650.676 | 918.7905 | 1223.086 | 918.7905 | 1650.676 | 731.8852 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 15532.82 | 8587.629 | 12607.41 | 8587.629 | 15532.82 | 6945.186 |
| meiji/calibration/cam1/observed/observed | 4171 | 1834.712 | 1313.122 | 729.6017 | 729.6017 | 1834.712 | 1105.11 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 20777.65 | 12932.63 | 18845.19 | 12932.63 | 20777.65 | 7845.013 |
| meiji/calibration/cam2/observed/observed | 3945 | 4365.255 | 2620.376 | 2098.473 | 2098.473 | 4365.255 | 2266.782 |
| meiji/calibration/evidence_gap/observed | 2584 | 15461 | 9230.076 | 13941.01 | 9230.076 | 15461 | 6230.92 |
| meiji/calibration/observed/observed | 12068 | 2601.673 | 1611.325 | 1338.688 | 1338.688 | 2601.673 | 1262.985 |
| meiji/cam0/evidence_gap/observed | 1697 | 13157.55 | 8680.635 | 13010.3 | 8680.635 | 13157.55 | 4476.911 |
| meiji/cam0/observed/observed | 7556 | 1712.375 | 951.9069 | 1287.507 | 951.9069 | 1712.375 | 760.4678 |
| meiji/cam1/evidence_gap/observed | 1738 | 13954.91 | 8331.726 | 11905.86 | 8331.726 | 13954.91 | 5623.18 |
| meiji/cam1/observed/observed | 7999 | 1689.434 | 1299.077 | 971.9033 | 971.9033 | 1689.434 | 717.5307 |
| meiji/cam2/evidence_gap/observed | 1642 | 17425.7 | 12463.3 | 14944.91 | 12463.3 | 17425.7 | 4962.402 |
| meiji/cam2/observed/observed | 7452 | 3399.972 | 2129.31 | 1569.551 | 1569.551 | 3399.972 | 1830.421 |
| meiji/evidence_gap/observed | 5077 | 14810.91 | 9784.58 | 13257.91 | 9784.58 | 14810.91 | 5026.328 |
| meiji/observed/observed | 23007 | 2251.014 | 1453.973 | 1269.134 | 1269.134 | 2251.014 | 981.8804 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 16183.47 | 11176.94 | 15586.99 | 11176.94 | 16183.47 | 5006.522 |
| meiji/selection/cam0/observed/observed | 3604 | 1780.031 | 988.221 | 1358.149 | 988.221 | 1780.031 | 791.8103 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 12362.4 | 8073.456 | 11197.81 | 8073.456 | 12362.4 | 4288.947 |
| meiji/selection/cam1/observed/observed | 3828 | 1531.139 | 1283.774 | 1235.916 | 1235.916 | 1531.139 | 295.2231 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 13957.45 | 11977.68 | 10909.3 | 10909.3 | 13957.45 | 3048.156 |
| meiji/selection/cam2/observed/observed | 3507 | 2314.131 | 1576.914 | 974.5699 | 974.5699 | 2314.131 | 1339.561 |
| meiji/selection/evidence_gap/observed | 2493 | 14137.09 | 10359.32 | 12549.87 | 10359.32 | 14137.09 | 3777.767 |
| meiji/selection/observed/observed | 10939 | 1864.164 | 1280.38 | 1192.401 | 1192.401 | 1864.164 | 671.7633 |
| tracknet/evidence_gap/observed | 349 | 4689.34 | 5669.914 | 6458.132 | 4689.34 | 6458.132 | 1768.792 |
| tracknet/observed/observed | 1538 | 56.38787 | 49.66632 | 53.95938 | 49.66632 | 56.38787 | 6.721549 |

## 固定倍率適用後: area_px2_0.9

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 280822.7 | 237943.9 | 328727.4 | 237943.9 | 328727.4 | 90783.53 |
| chat_annotation/observed/observed | 6622 | 68135.14 | 40559.04 | 50043.47 | 40559.04 | 68135.14 | 27576.11 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 55321.3 | 29296.99 | 50347.28 | 29296.99 | 55321.3 | 26024.3 |
| meiji/calibration/cam0/observed/observed | 3952 | 12932.9 | 6724.238 | 11813.31 | 6724.238 | 12932.9 | 6208.666 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 72023.69 | 37652.51 | 58699.03 | 37652.51 | 72023.69 | 34371.19 |
| meiji/calibration/cam1/observed/observed | 4171 | 19550.15 | 11888.8 | 12660.22 | 11888.8 | 19550.15 | 7661.35 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 89593.58 | 60233.95 | 84667.3 | 60233.95 | 89593.58 | 29359.63 |
| meiji/calibration/cam2/observed/observed | 3945 | 26671.13 | 20707.84 | 21295.26 | 20707.84 | 26671.13 | 5963.287 |
| meiji/calibration/evidence_gap/observed | 2584 | 72039 | 42116.93 | 64259.16 | 42116.93 | 72039 | 29922.07 |
| meiji/calibration/observed/observed | 12068 | 19710.98 | 13080.44 | 15205.65 | 13080.44 | 19710.98 | 6630.536 |
| meiji/cam0/evidence_gap/observed | 1697 | 66370.67 | 37312.15 | 64018.23 | 37312.15 | 66370.67 | 29058.52 |
| meiji/cam0/observed/observed | 7556 | 15610.27 | 7327.473 | 13004.35 | 7327.473 | 15610.27 | 8282.799 |
| meiji/cam1/evidence_gap/observed | 1738 | 63251.35 | 36113.21 | 53675.41 | 36113.21 | 63251.35 | 27138.14 |
| meiji/cam1/observed/observed | 7999 | 16705.55 | 10169.45 | 11039.06 | 10169.45 | 16705.55 | 6536.099 |
| meiji/cam2/evidence_gap/observed | 1642 | 77337.94 | 55166.56 | 69465.29 | 55166.56 | 77337.94 | 22171.38 |
| meiji/cam2/observed/observed | 7452 | 22045.88 | 17693.01 | 15897.79 | 15897.79 | 22045.88 | 6148.088 |
| meiji/evidence_gap/observed | 5077 | 68849.87 | 42676.18 | 62239.28 | 42676.18 | 68849.87 | 26173.69 |
| meiji/observed/observed | 23007 | 18075.57 | 11672.97 | 13258.25 | 11672.97 | 18075.57 | 6402.601 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 78160.27 | 45864.25 | 78605.01 | 45864.25 | 78605.01 | 32740.75 |
| meiji/selection/cam0/observed/observed | 3604 | 18546.16 | 7988.956 | 14310.38 | 7988.956 | 18546.16 | 10557.21 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 54397.88 | 34559.68 | 48605.34 | 34559.68 | 54397.88 | 19838.2 |
| meiji/selection/cam1/observed/observed | 3828 | 13606.06 | 8296.039 | 9272.635 | 8296.039 | 13606.06 | 5310.023 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 64657.07 | 49923.35 | 53735.83 | 49923.35 | 64657.07 | 14733.72 |
| meiji/selection/cam2/observed/observed | 3507 | 16842.97 | 14301.66 | 9826.215 | 9826.215 | 16842.97 | 7016.753 |
| meiji/selection/evidence_gap/observed | 2493 | 65544.32 | 43255.84 | 60145.67 | 43255.84 | 65544.32 | 22288.48 |
| meiji/selection/observed/observed | 10939 | 16271.38 | 10120.24 | 11109.86 | 10120.24 | 16271.38 | 6151.141 |
| tracknet/evidence_gap/observed | 349 | 19244.97 | 21266.34 | 26618.65 | 19244.97 | 26618.65 | 7373.678 |
| tracknet/observed/observed | 1538 | 343.6316 | 231.3477 | 299.0504 | 231.3477 | 343.6316 | 112.2838 |

## 固定倍率適用後: area_px2_0.95

| source / half / camera / condition / teacher | n | seed42 | seed43 | seed44 | min | max | spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| chat_annotation/evidence_gap/observed | 1502 | 378294.7 | 321415.9 | 454358.2 | 321415.9 | 454358.2 | 132942.3 |
| chat_annotation/observed/observed | 6622 | 96029.34 | 57235.83 | 73390.79 | 57235.83 | 96029.34 | 38793.51 |
| meiji/calibration/cam0/evidence_gap/observed | 876 | 80835.8 | 41679.72 | 73414.18 | 41679.72 | 80835.8 | 39156.07 |
| meiji/calibration/cam0/observed/observed | 3952 | 19816.37 | 10314.11 | 18841.21 | 10314.11 | 19816.37 | 9502.26 |
| meiji/calibration/cam1/evidence_gap/observed | 873 | 101952.2 | 53340.8 | 84452.49 | 53340.8 | 101952.2 | 48611.38 |
| meiji/calibration/cam1/observed/observed | 4171 | 30212.67 | 18384.56 | 20970.01 | 18384.56 | 30212.67 | 11828.11 |
| meiji/calibration/cam2/evidence_gap/observed | 835 | 124144.5 | 84017.96 | 118751 | 84017.96 | 124144.5 | 40126.53 |
| meiji/calibration/cam2/observed/observed | 3945 | 38632.3 | 30287.44 | 32847.66 | 30287.44 | 38632.3 | 8344.865 |
| meiji/calibration/evidence_gap/observed | 2584 | 101964.8 | 59300.68 | 91793.69 | 59300.68 | 101964.8 | 42664.13 |
| meiji/calibration/observed/observed | 12068 | 29560.47 | 19632.69 | 24155.65 | 19632.69 | 29560.47 | 9927.781 |
| meiji/cam0/evidence_gap/observed | 1697 | 94225.41 | 51784.82 | 92011.11 | 51784.82 | 94225.41 | 42440.59 |
| meiji/cam0/observed/observed | 7556 | 23877.58 | 11178.12 | 20460.29 | 11178.12 | 23877.58 | 12699.46 |
| meiji/cam1/evidence_gap/observed | 1738 | 89751.92 | 50881.21 | 77118.16 | 50881.21 | 89751.92 | 38870.71 |
| meiji/cam1/observed/observed | 7999 | 25647.49 | 15437.34 | 17749.66 | 15437.34 | 25647.49 | 10210.15 |
| meiji/cam2/evidence_gap/observed | 1642 | 107472.2 | 76479.29 | 98058.12 | 76479.29 | 107472.2 | 30992.89 |
| meiji/cam2/observed/observed | 7452 | 32431.06 | 26333.78 | 25104.18 | 25104.18 | 32431.06 | 7326.873 |
| meiji/evidence_gap/observed | 5077 | 96978.27 | 59462.16 | 88868.55 | 59462.16 | 96978.27 | 37516.11 |
| meiji/observed/observed | 23007 | 27263.42 | 17567.89 | 21022.03 | 17567.89 | 27263.42 | 9695.529 |
| meiji/selection/cam0/evidence_gap/observed | 821 | 108512 | 62566.87 | 111853.9 | 62566.87 | 111853.9 | 49287.01 |
| meiji/selection/cam0/observed/observed | 3604 | 28330.93 | 12125.55 | 22235.71 | 12125.55 | 28330.93 | 16205.38 |
| meiji/selection/cam1/evidence_gap/observed | 865 | 77438.82 | 48398.86 | 69716.01 | 48398.86 | 77438.82 | 29039.95 |
| meiji/selection/cam1/observed/observed | 3828 | 20673.27 | 12226.04 | 14240.75 | 12226.04 | 20673.27 | 8447.229 |
| meiji/selection/cam2/evidence_gap/observed | 807 | 90221.39 | 68679.05 | 76647.31 | 68679.05 | 90221.39 | 21542.33 |
| meiji/selection/cam2/observed/observed | 3507 | 25455.31 | 21886.33 | 16393.6 | 16393.6 | 25455.31 | 9061.717 |
| meiji/selection/evidence_gap/observed | 2493 | 91809.71 | 59629.53 | 85836.65 | 59629.53 | 91809.71 | 32180.18 |
| meiji/selection/observed/observed | 10939 | 24729.29 | 15289.98 | 17564.99 | 15289.98 | 24729.29 | 9439.307 |
| tracknet/evidence_gap/observed | 349 | 27644.84 | 29572.54 | 40616.43 | 27644.84 | 40616.43 | 12971.59 |
| tracknet/observed/observed | 1538 | 515.5204 | 331.3873 | 548.5801 | 331.3873 | 548.5801 | 217.1929 |
