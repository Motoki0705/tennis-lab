#2600の再実験10件が完了しました。レビュー修正後の重み9本とtest予測110ファイルを別出力に保存し、初回の成果物も保管しています。

| 方式 | v3全イベント macro-F1 | v3ポイント内 | 独立6本の全イベント |
| --- | ---: | ---: | ---: |
| 球のみ | 0.6688 | 0.7259 | 0.5011 |
| Shared | 0.6728 | 0.7326 | 0.5696 |
| Per Player | 0.6367 | 0.6903 | 0.5060 |
| Set Attention | **0.6795** | **0.7508** | 0.4868 |

追加学習の全9 runでv3の打球F1が改善しました。Set Attentionはv3で最高、Sharedは独立6本で最高でした。valのmacro-F1は同じrun内で0.113〜0.231動き、選択epochは1〜9に分かれました。Set Attentionをv3での追加検証候補とし、既定モデルの置換と全層fine-tuningは見送りました。 [比較と再現手順](https://github.com/Motoki0705/tennis-lab/blob/35aa2bc958e94286f225405195f2a30ef4ffed7a/knowledge/nodes/ball_refiner_3d/000027-group-i2600-motion-fusion.md)

- 実装ブランチ `codex/parent-1722-issue-2600-primary` の `3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f` で対象36テストが通過しました。配布版の `clip_057` で3方式の初期出力が球のみとビット一致し、手首192点のメッシュ基準との差は中央値2.35 cmでした。 [実装PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613)
- 記録ブランチ `codex/parent-1722-issue-2600-knowledge` の `35aa2bc958e94286f225405195f2a30ef4ffed7a` をpushしました。再採点で表・CSV・図など5点が掲載版とバイト一致し、knowledge 783ノードと再現参照601件の検査も通過しました。両PRの実行証拠を更新済みです。 [記録PR #2621](https://github.com/Motoki0705/tennis-lab/pull/2621)

#2613のCIは、`origin/main` と同一の共有キュー停止テスト1件がタイムアウトしたため、同じHEADで失敗ジョブを再実行中です。#2621のKnowledge CIも進行中です。 [実装CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45814820124) · [記録CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45880218730)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2613,"head":"3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f","resume_on_success":true,"next_step":"記録PR #2621 の HEAD 35aa2bc958e94286f225405195f2a30ef4ffed7a の Knowledge library CI の成否を照合する。"}}