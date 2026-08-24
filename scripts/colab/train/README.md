# Colab train scripts

train スクリプトは作成日ごとの `YYYY-MM-DD/` に保存する。同日に作成した複数のスクリプトは同じディレクトリに置く。

各スクリプトは必要な setup を内部で実行するため、Colab 側では Drive のマウント後に train スクリプトだけを実行する。

- `2026-07-02/`: 既存の7学習スクリプト。
- `2026-08-22/`: BLCS / PLCS の base-size track-query、および synthetic Court v2 KP の学習スクリプト。
- `2026-08-25/`: Issue #790 の入力長辺 `256/384/512` × encoder depth `1/8` × DPT decoder `tiny/small/base/large` gridをColab 1/2へ12条件ずつ分割し、各queueで直列実行する。seedは42固定で、seed安定性は測定しない。

Issue #790の2本は、Driveの`tennis_lab/data/court_query_issue790_v3.tar.zst`を`DATA_ROOT`へ展開する。このarchiveはroot-relativeに次の2 subtreeを含む必要がある。

- `issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes/B00/datasets/court/`
- `court_detection/derived_targets_issue790_v3/`

branchをcheckoutしてDriveをmountした後は、各runtimeで対応するシェルだけを実行する。

```bash
# Colab 1: grid conditions 0..11
bash scripts/colab/train/2026-08-25/train_court_query_scaling_grid_colab1.sh

# Colab 2: grid conditions 12..23
bash scripts/colab/train/2026-08-25/train_court_query_scaling_grid_colab2.sh
```

結果・checkpointは共通`OUTPUT_ROOT`、queue/repro bundleは`DRIVE_QUEUE_ROOT/issue-790/{colab-0,colab-1}`へ保存される。canonical manifestの学習条件は変更せず、実行環境ごとにPython executableとdata/external/output/checkpoint role rootだけをshard planへ記録して置換する。
