# Colab train scripts

train スクリプトは作成日ごとの `YYYY-MM-DD/` に保存する。同日に作成した複数のスクリプトは同じディレクトリに置く。

各スクリプトは必要な setup を内部で実行するため、Colab 側では Drive のマウント後に train スクリプトだけを実行する。

- `2026-07-02/`: 既存の7学習スクリプト。
- `2026-08-22/`: BLCS / PLCS の base-size track-query、および synthetic Court v2 KP の学習スクリプト。
- `2026-08-24/`: Issue #790 encoder scaling のColab shard。`colab0`はseed 43、`colab1`はseed 44を担当し、それぞれdepth `1/2/4/8`のtrainingと対応するcapacity profileを専用training queueで直列実行する。local GPUはseed 42のtraining/profileを担当する。

Issue #790の2本は、Driveの`tennis_lab/data/court_query_issue790_v3.tar.zst`を`DATA_ROOT`へ展開する。このarchiveはroot-relativeに次の2 subtreeを含む必要がある。

- `issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes/B00/datasets/court/`
- `court_detection/derived_targets_issue790_v3/`

branchをcheckoutしてDriveをmountした後は、各runtimeで対応するシェルだけを実行する。

```bash
# Colab 0: seed 43
bash scripts/colab/train/2026-08-24/train_court_query_consistency_encoder_colab0.sh

# Colab 1: seed 44
bash scripts/colab/train/2026-08-24/train_court_query_consistency_encoder_colab1.sh
```

結果・checkpointは共通`OUTPUT_ROOT`、queue/repro bundleは`DRIVE_QUEUE_ROOT/issue-790/{colab-0,colab-1}`へ保存される。canonical manifestの学習条件は変更せず、実行環境ごとにPython executableとdata/external/output/checkpoint role rootだけをshard planへ記録して置換する。
