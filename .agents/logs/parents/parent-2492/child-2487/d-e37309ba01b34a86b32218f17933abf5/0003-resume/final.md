2026-11-24のユーザー判断に沿い、RF-DETR-2XLの3条件をライセンス上の理由で対象外とし、同意フラグと専用の実行経路を削除しました。比較対象はApache-2.0のRF-DETR-Lです。[除外理由と再現手順](https://github.com/Motoki0705/tennis-lab/blob/1d5c4982428be19175394927d2d7267f88be6bbe/knowledge/runs/run-meiji-player-model-comparison/rfdetr.md)

疑似BBoxの各辺余白を5%・10%・15%とすると、PT32のRecall@0.75は13.94%・42.06%・56.87%でした。この感度を踏まえ、検出器比較の主指標には固定余白でのRecall@0.50を使うと記録しました。

`codex/parent-2492-issue-2487-rfdetr` の `1d5c4982428be19175394927d2d7267f88be6bbe` にコミット済みです。#2495・#2496の最新headを取り込み済みと確認し、対象テスト39件、pre-commit、knowledge検査が通過しました。既存のLのTensorRT planとPT32予測6,844行も読み戻せました。[#2487への記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7131685953)

Lの速度・VRAM計測は、#2496の固定速度入力の公開待ちです。[PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}