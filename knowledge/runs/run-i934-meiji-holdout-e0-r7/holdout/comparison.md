# Meiji holdout比較

主指標はoverall_observedとcamera行。visibility行は位置推定ラベルも含む参考値。
手首100.0pxは打球/飛行のproxy。pose欠落はunknownで、既存pose部分集合への選択バイアスがある。
score >= 0.5、一致距離 <= 20.0 source px。
accepted p95はscore閾値を通る全位置誤差（一致距離超も含む）。raw p95は低scoreも含むargmax誤差。
注釈位置不明は負例にせず、recallの分母から除外する。詳細条件・hashはprotocol.json。

| model | stratum | frames | references | missing | wrong | recall | top-K recall | accepted p95 px | raw p95 px |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ft_e13 | overall_observed | 28806 | 28806 | 13636 | 2354 | 0.4449 | 0.7159 | 346.0807 | 586.1035 |
| ft_e13 | camera/cam0 | 8979 | 8979 | 5940 | 567 | 0.2753 | 0.6365 | 279.9439 | 537.1133 |
| ft_e13 | camera_wrist/cam0/near_wrist | 922 | 922 | 595 | 94 | 0.2527 | 0.5607 | 372.8633 | 505.7823 |
| ft_e13 | camera_wrist/cam0/flight | 1388 | 1388 | 925 | 73 | 0.2810 | 0.7370 | 327.4515 | 512.5269 |
| ft_e13 | camera_wrist/cam0/unknown | 6669 | 6669 | 4420 | 400 | 0.2773 | 0.6260 | 218.3001 | 542.9107 |
| ft_e13 | camera/cam1 | 10135 | 10135 | 3610 | 941 | 0.5510 | 0.7797 | 480.8029 | 670.7596 |
| ft_e13 | camera_wrist/cam1/near_wrist | 1469 | 1469 | 697 | 190 | 0.3962 | 0.7107 | 495.0480 | 668.7692 |
| ft_e13 | camera_wrist/cam1/flight | 2750 | 2750 | 620 | 190 | 0.7055 | 0.8742 | 465.7842 | 623.5826 |
| ft_e13 | camera_wrist/cam1/unknown | 5916 | 5916 | 2293 | 561 | 0.5176 | 0.7529 | 474.2914 | 702.0572 |
| ft_e13 | camera/cam2 | 9692 | 9692 | 4086 | 846 | 0.4911 | 0.7228 | 285.9174 | 481.3556 |
| ft_e13 | camera_wrist/cam2/near_wrist | 2178 | 2178 | 1235 | 228 | 0.3283 | 0.5946 | 309.3874 | 389.2209 |
| ft_e13 | camera_wrist/cam2/flight | 1923 | 1923 | 482 | 70 | 0.7129 | 0.8513 | 18.8559 | 340.0498 |
| ft_e13 | camera_wrist/cam2/unknown | 5591 | 5591 | 2369 | 548 | 0.4783 | 0.7285 | 299.7123 | 514.5633 |
| ft_e13 | point_kind/observed | 28806 | 28806 | 13636 | 2354 | 0.4449 | 0.7159 | 346.0807 | 586.1035 |
| ft_e13 | point_kind/interpolated | 455 | 455 | 288 | 63 | 0.2286 | 0.5451 | 498.7900 | 575.4811 |
| ft_e13 | point_kind/occlusion_estimated | 138 | 138 | 105 | 18 | 0.1087 | 0.4058 | 301.9050 | 391.4500 |
| ft_e13 | point_kind/unresolved | 6607 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| ft_e13 | point_kind/out_of_frame | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| ft_e13 | point_kind/no_instance | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| ft_e13 | visibility/not_occluded | 29261 | 29261 | 13924 | 2417 | 0.4415 | 0.7132 | 350.7651 | 586.0941 |
| ft_e13 | visibility/occluded | 138 | 138 | 105 | 18 | 0.1087 | 0.4058 | 301.9050 | 391.4500 |
| ft_e13 | visibility/unlocated_or_unreviewed | 6607 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| ft_e13 | wrist/near_wrist | 4569 | 4569 | 2527 | 512 | 0.3349 | 0.6251 | 466.7400 | 574.2255 |
| ft_e13 | wrist/flight | 6061 | 6061 | 2027 | 333 | 0.6106 | 0.8355 | 254.8671 | 543.7384 |
| ft_e13 | wrist/unknown | 18176 | 18176 | 9082 | 1509 | 0.4173 | 0.6988 | 340.7611 | 596.6243 |
| mixed_ft | overall_observed | 28806 | 28806 | 4855 | 4362 | 0.6800 | 0.8511 | 386.2989 | 515.3416 |
| mixed_ft | camera/cam0 | 8979 | 8979 | 1675 | 1811 | 0.6118 | 0.8030 | 339.4206 | 443.1934 |
| mixed_ft | camera_wrist/cam0/near_wrist | 922 | 922 | 195 | 235 | 0.5336 | 0.7397 | 295.2915 | 412.4430 |
| mixed_ft | camera_wrist/cam0/flight | 1388 | 1388 | 201 | 243 | 0.6801 | 0.8948 | 307.9007 | 421.8831 |
| mixed_ft | camera_wrist/cam0/unknown | 6669 | 6669 | 1279 | 1333 | 0.6083 | 0.7926 | 345.4677 | 448.2037 |
| mixed_ft | camera/cam1 | 10135 | 10135 | 783 | 1277 | 0.7967 | 0.9090 | 335.6335 | 424.9326 |
| mixed_ft | camera_wrist/cam1/near_wrist | 1469 | 1469 | 190 | 211 | 0.7270 | 0.8822 | 353.9316 | 399.3706 |
| mixed_ft | camera_wrist/cam1/flight | 2750 | 2750 | 99 | 278 | 0.8629 | 0.9400 | 317.2333 | 434.3033 |
| mixed_ft | camera_wrist/cam1/unknown | 5916 | 5916 | 494 | 788 | 0.7833 | 0.9013 | 332.1470 | 443.7204 |
| mixed_ft | camera/cam2 | 9692 | 9692 | 2397 | 1274 | 0.6212 | 0.8352 | 537.9868 | 574.6882 |
| mixed_ft | camera_wrist/cam2/near_wrist | 2178 | 2178 | 802 | 365 | 0.4642 | 0.7782 | 324.1723 | 606.1884 |
| mixed_ft | camera_wrist/cam2/flight | 1923 | 1923 | 280 | 126 | 0.7889 | 0.9204 | 233.5112 | 648.0995 |
| mixed_ft | camera_wrist/cam2/unknown | 5591 | 5591 | 1315 | 783 | 0.6248 | 0.8281 | 550.9597 | 566.4639 |
| mixed_ft | point_kind/observed | 28806 | 28806 | 4855 | 4362 | 0.6800 | 0.8511 | 386.2989 | 515.3416 |
| mixed_ft | point_kind/interpolated | 455 | 455 | 100 | 97 | 0.5670 | 0.7824 | 530.0096 | 530.0096 |
| mixed_ft | point_kind/occlusion_estimated | 138 | 138 | 59 | 45 | 0.2464 | 0.4710 | 265.5807 | 543.6689 |
| mixed_ft | point_kind/unresolved | 6607 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| mixed_ft | point_kind/out_of_frame | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| mixed_ft | point_kind/no_instance | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| mixed_ft | visibility/not_occluded | 29261 | 29261 | 4955 | 4459 | 0.6783 | 0.8501 | 390.4400 | 516.1115 |
| mixed_ft | visibility/occluded | 138 | 138 | 59 | 45 | 0.2464 | 0.4710 | 265.5807 | 543.6689 |
| mixed_ft | visibility/unlocated_or_unreviewed | 6607 | 0 | 0 | 0 | N/A | N/A | N/A | N/A |
| mixed_ft | wrist/near_wrist | 4569 | 4569 | 1187 | 811 | 0.5627 | 0.8039 | 334.8549 | 478.7494 |
| mixed_ft | wrist/flight | 6061 | 6061 | 580 | 647 | 0.7976 | 0.9234 | 305.3423 | 474.8658 |
| mixed_ft | wrist/unknown | 18176 | 18176 | 3088 | 2904 | 0.6703 | 0.8389 | 452.2435 | 529.0820 |
