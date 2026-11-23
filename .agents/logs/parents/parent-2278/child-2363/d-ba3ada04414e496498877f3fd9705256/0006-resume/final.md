正式レビューのmajor 3件を修正し、`codex/parent-2278-issue-2363-primary` のHEAD `a42c9f6e7` をpushしました。修正内容と検証結果をPR本文に追記しました。 [PR #2433](https://github.com/Motoki0705/tennis-lab/pull/2433)

- **FP32の互換性** `a42c9f6e7` — ソース差分30ファイルの完全一致を条件に、固定mainのidentity `6022ecaa…` を復元します。既定・明示FP32でのキャッシュ再利用と、公開済みannotationの `overwrite=false` での受理を確認しました。#2379のHEADとマージ時点でも同じidentityが維持されていました。
- **先読み中断** `d09216b12` — 満杯キューで予測が失敗した場合もproducerを停止し、待機spanを中断として記録します。停止に5秒を超えた場合や未完了spanが残った場合は、記録を失敗扱いにします。
- **公開identity** `0ab0ccb7b` — 同じ入力と設定で記録なし・ありを別々に公開し、`publication_identity_sha256` とsceneのNPZ・metadataのバイト列が一致することを確認しました。

動画バッチ・公開・計測の81件、設定監査・GVHMRの405件、pre-commitのruff・mypyが通りました。全件E2Eで停止した14件の扱いは親の判断待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2433,"head":"a42c9f6e75ea982af391010280161ef353cd9dd5","resume_on_success":true,"next_step":"最終報告"}}