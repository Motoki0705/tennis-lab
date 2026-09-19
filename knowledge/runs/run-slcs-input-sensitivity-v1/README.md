# 固定train窓の入力時間感度診断

結論・介入の定義・限界は[run node](../../nodes/run-slcs-input-sensitivity-v1.md)を参照。

- `probe.py`: CPU再現入口。既存勾配probeのcheckpoint/入力identity・失敗保存関数を再利用するため、repo rootから実行する。
- `test_probe.py`: permutation、mask/target不変、距離単位の小さな決定的テスト。
- `result/results.json`: 実行command・環境・入力hash・source frame ID・介入mapping・metric。
- `result/{broadcast,meiji}.npz`: baseline/介入の正規化3D出力、教師、教師mask、ball UV/visibility、DINO以外の入力。mへ変換するscaleは結果JSONにある。
- `result/resolved_training_config.yaml`: 元学習設定の解決済みsnapshot。
- `result/executed_probe.py.txt`: 実行時の正確なソース。結果JSONのprobe source SHAと一致する。`probe.py`は実行後にruff整形、型明示・config型検査・baseline非None検査・NPZのallow_pickle=False明示を加えた。

DINOは大量コピーせず、前runの保存batchファイルと各tensorのhashを記録した。再現には`outputs/slcs/analyze/ball_gradient_probe/s42-001/*_inputs.npz`と選定checkpointが必要。生動画やdatasetを現在の版から再生成する入口ではない。

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -B knowledge/runs/run-slcs-input-sensitivity-v1/probe.py \
  --output-dir knowledge/runs/run-slcs-input-sensitivity-v1/REPRO_NEW_RUN_ID
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/python -B -m pytest -o addopts='' -q \
  knowledge/runs/run-slcs-input-sensitivity-v1/test_probe.py
```
