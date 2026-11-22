ViTPose-HとHMR2の5条件、およびCLIP-ReIDの新しい3a・3bを実測し、[ドラフトPR #2334](https://github.com/Motoki0705/tennis-lab/pull/2334)を作成しました。ブランチは `codex/trt-vit-pose-hmr2`、HEADは `45045468e8d320b675b8632f00cc9a34169b5a43` です。Python CIの結果を受け取った後に、対象HEADを照合してドラフトを解除します。

**ViTPose-HのE2E候補はTensorRT FP16です。** flip test込みで中央値54.423 msから8.468 msに短縮し、座標差の最大値は0.302 pxでした。INT8とFP8は最大5 pxの基準を超えました。HMR2はFP8が最速の5.399 msで、cosineと相対L2の単体基準を満たしました。

### 5条件の実測

VRAMはTorch allocatorの割当最大[local path]。GPUカウンタは代表1 cropを使用しました。精度欄はViTPose-Hが座標差P95／最大、HMR2がcosine P05／相対L2 P95です。

| モデル | 条件 | 中央値[local path] ms | crop/s | VRAM MiB | DRAM R／W bytes | kernel数 | 最大stall／Tensor Core active | 単体精度 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| ViTPose-H | 0 PyTorch FP32 | 54.423／55.147 | 18.31 | 2,549.40／3,048.1 | 3,966,944,768／145,381,632 | 634 | `mio_throttle` 18.44%[local path] | 0／0 px |
| ViTPose-H | 1 PyTorch FP16 | 27.197／28.322 | 36.57 | 2,553.93／3,050.1 | 4,701,833,472／40,244,224 | 994 | `long_scoreboard` 40.72%[local path] | 0.00768／0.10426 px |
| ViTPose-H | 2 TensorRT FP16 | 8.468／8.608 | 117.36 | 1.42／1,682.1 | 1,478,578,688／6,400 | 238 | `long_scoreboard` 4.14%[local path] | 0.02471／0.30197 px |
| ViTPose-H | 3a TensorRT INT8 | 8.691／8.799 | 114.88 | 2.85／1,642.1 | 1,368,989,184／0 | 300 | `long_scoreboard` 2.83%[local path] | 0.07941／23.35357 px |
| ViTPose-H | 3b TensorRT FP8 | 7.732／8.075 | 125.62 | 2.85／1,092.1 | 767,607,040／0 | 270 | `long_scoreboard` 3.70%[local path] | 0.28388／29.05916 px |
| HMR2 | 0 PyTorch FP32 | 36.564／39.089 | 27.06 | 2,669.67／3,172.1 | 3,376,803,584／22,684,928 | 705 | `mio_throttle` 15.74%[local path] | 1.00000000／0 |
| HMR2 | 1 PyTorch FP16 | 25.058／28.535 | 39.06 | 2,677.02／3,172.1 | 4,487,879,936／39,337,728 | 1,131 | `long_scoreboard` 52.27%[local path] | 0.99999838／0.001800 |
| HMR2 | 2 TensorRT FP16 | 6.538／6.960 | 150.86 | 0.38／1,710.1 | 1,336,494,848／0 | 309 | `long_scoreboard` 4.88%[local path] | 0.99999198／0.004006 |
| HMR2 | 3a TensorRT INT8 | 6.573／6.966 | 150.73 | 0.75／1,680.1 | 1,286,221,568／0 | 341 | `long_scoreboard` 4.10%[local path] | 0.99990905／0.013965 |
| HMR2 | 3b TensorRT FP8 | 5.399／5.953 | 182.09 | 0.75／1,132.1 | 709,049,600／0 | 341 | `long_scoreboard` 5.58%[local path] | 0.99965411／0.026229 |

IQR・MAD、全stall、confidence差、外れ値のcropと関節も記録しました。[runノード](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/nodes/tennis_scene/000077-run-i2303-vit-tensorrt.md)

### 受入条件と証拠

- **固定入力は達成。** 評価 `clip_000/037` と校正 `clip_027` の媒体・元フレーム区間の重複は0件。独立に再展開したテンソルはlockと一致しました。SHA-256は `757984765eb0a9f069154dc3e143121b9ede97f6223806e10cef971a081a4b8e` です。[tensor lock](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/runs/run-i2303-vit-tensorrt/tensor-lock.json)
- **5条件の計測は達成。** タイムアウトしたINT8カウンタは同一planで追測し、ViTPose-Hの300件、HMR2の341件がNsight Systemsと一致しました。CLIP-ReIDを含む全12条件のSQLiteとCSVから集計値を再確認しました。[metrics.json](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/runs/run-i2303-vit-tensorrt/metrics.json)
- **INT8劣化の切り分けは達成。** 旧CLIP-ReIDのQ/DQで活性化181辺を元に戻すと、12 cropのcosine P05が0.51539から0.99974に回復しました。LayerNorm入力を戻すと0.74920、LayerNorm・QuickGELU・残差Addをまとめて戻すと0.84776でした。同じ重み付き線形層のmax／entropyは0.84918／0.85612で、量子化範囲の影響が大きいことを確認しました。[層別・校正比較](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/runs/run-i2303-vit-tensorrt/int8-diagnosis.json)
- **3モデルの再計測は達成。** 3aはattention出力投影だけのINT8 max、3bは重み付き線形層4種のFP8 maxです。CLIP-ReIDのINT8はcosine P05が0.489896から0.999602、FP32とのID一致が96/236から236/236に回復しました。FP8は0.995429、ID一致236/236でした。同じ設定でHMR2は両条件とも基準内、ViTPose-HはINT8で1点、FP8で6点が5 pxを超えました。[条件定義](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/runs/run-i2297-clipreid-tensorrt/PROTOCOL.md%23vit-3%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AEptq%E8%A8%AD%E5%AE%9A)
- **実行精度と失敗時の停止を確認。** TensorRT全8条件でEngine Inspectorの結果が変換時・推論時の精度記録と一致しました。変換・精度照合・プロファイル取得に失敗した際の非ゼロ終了も記録しました。[validation.json](https://github.com/Motoki0705/tennis-lab/blob/45045468e8d320b675b8632f00cc9a34169b5a43/knowledge/runs/run-i2303-vit-tensorrt/validation.json)

関連pytestは**72件成功、6.09秒**で、増分は19件です。pre-commitを実行し、hook対象外の実験Python 8ファイルはRuffで直接検証しました。knowledge検証はエラー・警告0件でした。

E2Eでは、B=1単体計測から変わる段間転送と複数モデル常駐時のVRAM、HMR2 FP8の特徴差が時系列復元に及ぼす影響を確認する必要があります。CLIP-ReIDのPTQ範囲を選んだ12 cropは、再計測の236 cropにも含まれます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2334,"head":"45045468e8d320b675b8632f00cc9a34169b5a43","resume_on_success":true,"next_step":"対象HEADのPython CI成功を照合し、PR #2334のドラフトを解除して最終結果を報告する"}}