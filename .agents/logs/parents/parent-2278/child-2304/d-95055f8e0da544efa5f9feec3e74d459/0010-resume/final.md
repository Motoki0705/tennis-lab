DINOの5条件とMSDAの3方式を記録し、`codex/trt-dino` の `64dec6838` をpushしました。[ドラフトPR #2342](https://github.com/Motoki0705/tennis-lab/pull/2342) の対象HEADで `Python` CIの結果を待ちます。

| 受入条件 | 結果 |
|---|---|
| 固定入力とshape bucket | 評価64枚・PTQ64枚を固定。動画と元フレーム区間の重複は各0件で、再採取のdigestも一致。B1・750×1333の1 bucketで、TRTのmin・opt・maxは同一。[tensor lock](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/runs/run-i2304-dino-tensorrt/tensor-lock.json) |
| MSDA方式比較 | 3方式の変換・速度・精度・運用負担を記録。正式な条件2はGridSampleで、プラグイン版もモデル全体で実測。Phase 2の採否は #2278 に委ねます。[方式比較](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/runs/run-i2304-dino-tensorrt/README.md) |
| 5条件の実測 | 全条件の速度・VRAM・nsys・単体精度を記録。正式なncuは4条件で取得し、3aは親の判断に従い未採取と記録。[runノード](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/nodes/tennis_scene/000078-run-i2304-dino-tensorrt.md) |
| フォールバック防止 | 4つのTRT planの実行精度を再照合。範囲外shapeと精度照合失敗の非ゼロ終了、部分TRTのFP16射影からFP32 CUDA MSDAへの境界を確認。[試行記録](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/runs/run-i2304-dino-tensorrt/attempts.json) |

| 条件 | 中央値 / P95 ms | 枚[local path] | Peak GiB | DRAM R / W GB | kernels | 主要stall | Tensor Core active | IoU P05 / xyxy差P95 px / score差P95 | 欠落 / 追加 |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| 0 PyTorch FP32 | 268.15 / 272.50 | 3.72 | 2.51 | 26.36 / 12.50 | 1,862 | long scoreboard 28.36% | 0.00% | 1.0000 / 0 / 0 | 0 / 0 |
| 1 PyTorch FP16 | 154.09 / 169.25 | 6.41 | 2.52 | 21.21 / 8.19 | 2,370 | long scoreboard 45.64% | 14.58% | 0.9963 / 0.0371 / 0.0060 | 1 / 0 |
| 2 TRT FP16 GridSample | 56.98 / 57.32 | 17.54 | 1.12 | 6.48 / 2.62 | 561 | long scoreboard 52.69% | 26.20% | 0.9325 / 0.7074 / 0.0173 | 3 / 2 |
| 3a TRT INT8 | 45.14 / 45.51 | 22.14 | 1.00 | 未採取 | 585 | 未採取 | 未採取 | 0.9253 / 0.9105 / 0.0517 | 11 / 5 |
| 3b TRT FP8 | 52.55 / 52.84 | 19.02 | 0.90 | 5.37 / 2.22 | 608 | long scoreboard 44.59% | 16.74% | 0.9132 / 0.9177 / 0.0662 | 5 / 11 |

比較用のFP16プラグイン版は **41.47 / 41.82 ms、24.09枚[local path]、1.06 GiB、560 kernels**。GridSample版より中央値が27.2%短く、IoU P05は0.9310、xyxy差P95は0.7106 px、score差P95は0.0195、欠落・追加は各4件でした。条件1は #2238 と同じFP32 MSDAを使い単体許容差の草案を満たしました。TRT各条件とプラグイン版はIoUとscore差が草案の範囲を外れました。[全指標と精度分布](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/runs/run-i2304-dino-tensorrt/metrics.json)

3aのkernel replayはv6が45分、同一planのv7が12分で打ち切られ、判定時のカウンタ行は各0件でした。v6のCSVは判定後に出力され、v7のapplication replay診断は585カーネルでnsysと一致しました。v7はkernel replayの失敗を示す終了値3を返し、診断値は試行記録に保存しました。[v6・v7の記録](https://github.com/Motoki0705/tennis-lab/blob/64dec6838e07df59b8f00885ce94bdc4eeb068b1/knowledge/runs/run-i2304-dino-tensorrt/attempts.json)

#2327・#2334を含むmainを取り込み、関連CPUテストは **133件成功、main比11件増、6.77秒**でした。Ruffとknowledge検証は成功し、pre-commitの2 hookは設定対象パスに変更がなくskipされました。

3a・3bは計測時点の #2297 設定です。#2334のViT向けPTQ設定をDINOに適用した結果、プラグイン版のPTQ、E2Eは未確認です。E2Eではbox差が追跡ID、カメラ間対応、3Dの欠測・誤差へ伝わる範囲と、横断batch時の速度を確認する必要があります。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2342,"head":"64dec6838e07df59b8f00885ce94bdc4eeb068b1","resume_on_success":true,"next_step":"ドラフト解除と最終報告"}}