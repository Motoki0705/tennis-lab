第3回レビューの修正を `codex/person-stage-speed` の `83d57c76c` に反映し、PR本文に検証結果を追記しました。 [PR #2240](https://github.com/Motoki0705/tennis-lab/pull/2240)

固定8画像・初回全尺・再開の各入口で、固定したソースとツールチェーンからDINO拡張を再生成し、SHA-256を照合します。CPUでの新規ビルドは約155秒で、両方の入力pinと完全一致しました。不一致時に停止することも確認しました。GPU再計測は行っていません。 [ビルド環境と再現手順](https://github.com/Motoki0705/tennis-lab/blob/83d57c76c/knowledge/runs/run-person-stage-precision-meiji-20261118/README.md)

対象HEADの手動CIは `Python` を含む全14ジョブが成功し、Knowledge libraryも通過しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44307884471)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44307884480)

最新mainの変更と `definition.py`・`orchestrator.py` で衝突しています。衝突の解消と、統合後のPR CI・FP32キャッシュ互換の確認が残っています。 [main側の変更 #2244](https://github.com/Motoki0705/tennis-lab/pull/2244)