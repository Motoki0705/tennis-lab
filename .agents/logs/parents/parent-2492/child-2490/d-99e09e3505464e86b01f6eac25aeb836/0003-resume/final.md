公式重み1B・5Bの SHA-256 が一致しました。公式モデルと重みのテンソル名・shapeも全件一致し、BBox の crop と左右反転を確認しました。結果と環境の再現手順を `67cccd7be2d727824c985f4ea08f4eb3bc6cbc0c` に記録しました。 [検証記録](https://github.com/Motoki0705/tennis-lab/blob/67cccd7be2d727824c985f4ea08f4eb3bc6cbc0c/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2.md)

#2495・#2496 の最新 HEAD を取り込み済みで、採点器の修正を含むテスト47件が通過しました。初回の精度評価は更新済み採点器で行います。

依存パッケージの取得タイムアウトを受け、必要な依存に絞った Python 3.12 環境の構築を再開しました。同じジョブで環境の確認後に1Bの ONNX 変換を試します。GPUでの精度評価と5B FP32のCUDAロード試行は未実施です。速度計測用の固定入力 manifest は #2497 で公開待ちです。 [#2497 進捗](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T132058Z_94a174870c92_sapiens2-runtime-onnx-1b"]}}