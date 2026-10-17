# Ball detection diagnostics

GPUを使うコマンドはすべて元repo共通training queueの`resource=all`で実行する。通常のpytest CPU suiteとは別に、実データを明示して起動する。

| module / script | 責任 |
|---|---|
| `training.py` | 同じ窓でcompute/pipeline/prepared、BS、precision、compile、decoder、先読みを計測 |
| `input_pipeline.py` | CPUの検証・JPEG・配置・collate/IPCを分解。CUDA_VISIBLE_DEVICES=''が必要 |
| `jpeg_decode_probe.py` | 実JPEGのOpenCV/nvJPEG画素・MDD差分を計測 |
| `jpeg_prefetch_check.py` | 同期/非同期復号の画素・順序・教師・例外/終了処理を確認 |
| `native_rgb_correctness.py` | FP32 MDD、native/compile出力と勾配の数値確認 |
| `checkpoint_replay.py` | 明示checkpointから上限付き失敗prefixを再生。学習重みは保存しない |
| `architecture.py` | CNN候補のparameter/MAC数をmeta deviceで比較 |
| `pretraining_comparison.py` | 完了済みのresidual/ConvNeXt V2/FasterNetの3runを同一recipe・予算で照合し、Common誤差で順位をJSONへ保存。学習やsweepは起動しない |
| `detection_holdout.py` | Meiji holdoutの固定checkpoint比較 |
| `detection_evidence.py` | sceneへの検出証拠出力とload-only再開の契約確認 |

各scriptの`--help`を参照する。例：

```bash
PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_detection/training.py --help
```

固定BS一覧を順番に実行するshell、特定campaignのcase sweep、学習CLIを複製したsmoke入口は保守しない。
保存・再開・freeze/unfreezeは `tests/integration/tasks/ball_detection/` の本番CLIテストが担当する。
過去の試行の正本はknowledgeのrun ID・commit・再現bundle。記録済みのコマンドは当時のcommitで実行する。
ベンチマークの少数窓の速度・loss低下を精度や長時間安定性の証拠と解釈しない。
