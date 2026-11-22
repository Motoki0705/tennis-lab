# PR #2240 再レビューのmajor指摘への対応（親 #1734 から #2238 へ）

再レビュー（gpt-6.2-sol / xhigh）で、major指摘が1件出た。前回の2件の修正は、受け入れられた。

## 公開のみの再開で指定精度と異なるsceneを採用する

静的な推論。FP32でscene保存後に公開が失敗し、同じstoreと注釈要求で `direct_annotation.resume=publication` と `people_models.runtime.vitpose.precision=fp16` を指定すると、`TennisSceneOrchestrator.run` は通常のcomponent identity検証を経ずに `saved_scene` を呼びます。保存receiptにはDINO・ViTPoseの精度がなく、`saved_scene` は入力・注釈要求・幾何・保存済みscene参照の一致で旧FP32出力を受理します。公開再開時にも両モデルの精度を照合し、不一致を拒否してください。既存の公開中断・再開の統合テストで、精度だけを変えた場合の拒否を検証できます。[再開経路](https://github.com/Motoki0705/tennis-lab/blob/a9f9ff820c7c99615f736d6892d9be955eaa3069/src/tennis_scene/pipeline/orchestrator.py#L168-L187)・[保存receiptと照合処理](https://github.com/Motoki0705/tennis-lab/blob/a9f9ff820c7c99615f736d6892d9be955eaa3069/src/tennis_scene/pipeline/direct_annotation/__init__.py#L78-L117)

## 親の方針
- **直す。** 保存receiptにDINOとViTPoseの精度を入れ、公開だけを再開するときに照合する。一致しなければ明示的に拒否する。精度だけを変えた場合の拒否を、既存の公開の中断・再開の統合テストで確かめる。
- 精度の情報がない旧receiptをどう扱うかは、指摘1（基準mainの成果物の再利用）の互換方針と矛盾しないように決め、PR本文に明記する。旧receiptはfp32として扱ってよい。ただし、明示的な分岐にし、そのことをテストで固定する。
- 同じ種類の抜け（精度を照合しない別の再利用経路）がほかにないかを、あわせて確認する。
- CPUの関連テスト、pre-commit、CIを通し、PR本文に追記する。merge・closeは親が行う。

