#2303 の再試行と追加診断を、共有 training queue に11件登録しました。`codex/trt-vit-pose-hmr2` のコミットは `a48ad429a` です。PR作成とCIは実測結果の取り込み後に行います。

- **固定入力**は ViTPose-H が評価236件・PTQ254件、HMR2 が評価40件・PTQ32件です。再展開した入力はロックと完全一致し、評価・PTQ間で媒体と元動画フレーム区間の重複は0件でした。[入力ロック](knowledge/runs/run-i2303-vit-tensorrt/tensor-lock.json)
- **初回の失敗8件**はすべて終了値1でした。両モデルの条件0では精度検査が PyTorch 2.13 の複合演算を捕捉できず、後続6件はFP32の結果が存在しなかったため停止しました。検査を修正し、pytest 46件、Ruff、knowledge検証を通しました。[失敗記録](knowledge/runs/run-i2303-vit-tensorrt/validation.json)
- **CLIP-ReIDのINT8**では、旧グラフでLayerNorm・QuickGELUの積・残差Addの活性化Q/DQを外すと、cosine P05が **0.515 → 0.848** に改善しました。同じ32校正cropで重み付き線形層だけに絞った比較では、maxが **0.849**、entropyが **0.856** でした。暫定基準は0.995です。初回の全辺除去には活性化181辺と定数49辺が含まれていたため、今回の再試行では両者の寄与を個別に測ります。[INT8診断](knowledge/runs/run-i2303-vit-tensorrt/int8-diagnosis.json)

**投入したジョブ**

| 対象 | ジョブ名 |
|---|---|
| ViTPose-H | `i2303-vitpose-0-r2`、`i2303-vitpose-1-r2`、`i2303-vitpose-2-r2`、`i2303-vitpose-3b-r2` |
| HMR2 | `i2303-hmr2-0-r2`、`i2303-hmr2-1-r2`、`i2303-hmr2-2-r2`、`i2303-hmr2-3b-r2` |
| CLIP-ReID | `i2303-clip-ablation-r2`、`i2303-clip-sensitivity-r2`、`i2303-clip-3b-r2` |

[キュー登録記録](outputs/i2303/round2-queue.txt)

再開後は各ジョブの終了値、計測結果、TensorRTの実行精度を照合します。CLIP-ReIDの投影層別の結果から3aの候補を決め、3モデルのINT8計測を登録します。その結果を `PROTOCOL.md` とknowledgeのrunノードに反映し、PRの対象HEADでPython CIを確認します。