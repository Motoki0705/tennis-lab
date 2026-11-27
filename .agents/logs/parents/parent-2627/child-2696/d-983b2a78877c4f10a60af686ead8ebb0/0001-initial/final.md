固定3クリップの C1〜C4 を、共有 GPU キューの `all` 枠へ1件で登録しました。計測コードは `codex/parent-2627-issue-2696-primary` の `2dbd8c355faa2d530b8b0e17164f691e60a57d07` に固定しました。

#2363 と同じ12映像、manifest、共有コート、drift を照合しました。同梱テスト6件と既存テスト45件、ruff が通過しました。C1〜C3 の CLIP-ReID・HMR2・GVHMR には forward 限定の CUDA autocast FP16 を適用します。GVHMR の clip 内バッチ上限は全条件で4に揃えました。 [計測設計と見積もり](https://github.com/Motoki0705/tennis-lab/issues/2696#issuecomment-7219784149)

GPU 計測結果、出力整合、比較図、記録専用 PR はキュー完了後に確認・作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204212722975544204_2026-11-27T13-47-57-024455795Z_973716_i2696-ablation-v1.job"]}}