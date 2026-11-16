## 担当範囲
#1750（C7）を完了させる。DINO人物検出の正式な複数入力APIの追加と、batch size 1と複数での一致確認まで。

## 資源と制約
- GPUでの確認は共有training queue（`TRAINING_QUEUE_DIR=[local path]`）経由。WSL2・16GB GPU。
- 実画像での一致確認には、書き出し済みの `data/tennis_multiview/processed/meiji_4cam/dataset/` の若い番号のclipを読み取り専用で使ってよい（書き出し中なので、未完了のclipは使わない。datasetへ書き込まない）。
- worktreeの `data`・`ckpt`・`outputs` はmainの実体へのsymlink。`third_party/DINO` のsubmoduleは未初期化の可能性があるので、必要なら確認して対処し、方法を報告する。CUDA拡張が読めない既知の問題（#1088のplayer予測）に当たったら、原因と対処を報告する。

## 報告
PR URL、APIの形、batch sizeごとの一致結果（許容差と最大差）、速度とpeak VRAM。
