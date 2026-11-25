EdgeCrafter の公式6重みを固定し、ECDet／ECPose の推論 adapter、#2495 採点用の JSONL 出力、#2496 の計測 harness への接続を実装しました。branch は `codex/parent-2492-issue-2487-edgecrafter`、HEAD は `bf9cf828f6508c48da90e220743e06b330be47ff` です。[配布条件と再現手順](https://github.com/Motoki0705/tennis-lab/blob/bf9cf828f6508c48da90e220743e06b330be47ff/knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md)

新規4件を含む pytest 22件が成功しました。ruff と mypy も通過し、固定コホート6,844画像の manifest から元フレームを読み出せることを確認しました。[#2487 の検証記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7130352081)

**統合担当への判断依頼**　公式 ECPose は人物スコアと COCO17 座標を出力します。比較と recipe で、関節の外接矩形を縦横それぞれ10%ずつ広げた BBox を採用してよいか判断をお願いします。疑似GTと生成規則が似るため、比較表では BBox の由来を明記する必要があります。[公式 ECPose ヘッド](https://github.com/Intellindust-AI-Lab/EdgeCrafter/blob/d8752b75c9beaecfcdd0e5cd207bae3b6946ae23/ecpose/engine/edgecrafter/detrpose_transformer.py#L749)

6重みの実体照合と実フレームでの CPU 推論をローカルジョブに投入しました。TensorRT 変換、GPU 18条件の計測と採点が残っています。速度計測には #2496 の固定入力を使います。[ローカルジョブログ]([local path])

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T113947Z_a94715bdcac9_edgecrafter-assets-and-cpu-smoke"]}}