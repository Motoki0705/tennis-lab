# 前半T64の12k終端と4条件比較

科学的な条件・結果・限界は[runノード](../../nodes/ball_detection/000093-run-i1899-convnext-1query-t64-t64-s42-12k.md)、比較の解釈は[group](../../nodes/ball_detection/000094-group-i1899-controlled-query-ablation-s42.md)。全4条件の数値は[比較表](comparison.md)。

## 証拠の配置

- `stage1/2-validation.json`: 各6kの元評価JSON。
- `stage1/2-publish-receipt.json` と `stage1/2-readback-receipt.json`: 完成世代のDrive保存と独立復元。
- `stage1/2-SNAPSHOT.json`: 今回Driveから再取得し、過去receiptとの内容一致を確認したmarker。
- `completion-verification.json`: 親がmarker・全payload・full state・予測集計・評価ID・終了状態を再確認した結果。
- `configuration.json`: 実行source、学習条件とGPU移管の履歴。

checkpoint、全frame予測、全更新ログと専用envは実行worktreeの `outputs/ball_detection/train/query_controls/desktop-resume-20261112/` に残る。Driveは `outputs/ball_detection/train/query_controls/s42-20261111-main/desktop-attempts/20261112-a1/`。このGit bundleは重みや予測streamを含まない。

## 完了照合の再実行

`SOURCE` を固定実行版のcheckout、`ADAPTER` を移管adapterのdirectory、`OUTPUT` を上記実行出力、`EVIDENCE` をこのdirectoryの絶対pathへ設定する。`PY` は実行出力の `env/bin/python`、`QUEUE` は元repoの `.training_queue`。`SOURCE` 内をcwdにし、次を実行する。学習や推論は再実行しない。

```bash
"$PY" -B "$EVIDENCE/verify_completed.py" \
  --run-root "$OUTPUT" --source "$SOURCE" --adapter "$ADAPTER"
```

元の本番commandは実行出力の `production-registration.json` と共有queueのrepro bundleに保存されている。実行済みattemptへの再投入を避ける。起動・失敗時の新attempt復旧は[固定adapterの手順](https://github.com/Motoki0705/tennis-lab/blob/f06f1bda6610838313e4b306a92a4db5e42ed579/knowledge/runs/run-ball-query-desktop-20261112/recovery.md)を使う。
