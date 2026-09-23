# Association入力のCPUベンチマーク

`association_input.py`はPLCS/BLCSの実V2データから、各taskのtrain 8例・val 4例を
作る時間と全tensor fieldのSHA-256を記録します。seed42、CPU thread1、worker0、
T512で、DataModule初期化・1例のwarmup・ハッシュ計算は計測区間から除きます。
valは異なるindexを一度ずつ読むため、結果キャッシュのhitを測りません。
GPUでのforward/backwardや新しいデータセット生成は行いません。

比較するcheckoutをCWDにし、改善branchにあるscriptの絶対pathを使います。
`PYTHONPATH=.`により、CWDの実装・設定を読み込みます。

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /path/to/repo/.venv/bin/python \
  /path/to/improved-worktree/tests/benchmarks/association_input.py \
  --data-root /path/to/repo/data --output /tmp/baseline_a.json
```

baseline→candidate→candidate→baselineの順に別processで実行し、
`(task, stage, index)`ごとの`sample_sha256`が全runで一致することを確認します。
各versionの`seconds`をtask/split別に集計して比較してください。
CPU負荷・cache状態の揺れを含むため、この倍率を学習全体やGPU利用率の倍率として
扱わないでください。通常のCIでは時間の閾値を設けず、別途unit testで順序・欠測・
overflow・gradient・追跡結果の同値性を検証します。
