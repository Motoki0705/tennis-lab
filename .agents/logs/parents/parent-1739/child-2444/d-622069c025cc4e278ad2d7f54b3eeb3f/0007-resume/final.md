#2442の疑似正解は品質を理由に棄却されたため、採点コード・集計JSON・knowledgeの指標・PR本文の表を削除しました。ダブルスの精度は #2459 の新旧横並び比較で定性評価します。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449) [比較ページ #2459](https://github.com/Motoki0705/tennis-lab/issues/2459)

比較用の保存済み69 clipを読み戻し、46,228フレームすべてに候補があり、DINO行の割当率が85.90%であることを確認しました。処理時間、Meiji 3camの人手評価、未使用視点への再投影は同じ実測記録を参照できます。 [比較用の保存済み候補](https://github.com/Motoki0705/tennis-lab/blob/590c2377810762fdea0d5485d10307e612de51eb/outputs/i2444/refine/timing/meiji4) [実測記録](https://github.com/Motoki0705/tennis-lab/blob/590c2377810762fdea0d5485d10307e612de51eb/knowledge/runs/run-multiview-pose-link-recall-20261124/README.md)

予約3 clipは**GPUロックのACL復旧待ち**です。4ファイルすべてで `mask::---` を確認し、ViTPoseのqueue投入・実データの採点はともに0回です。復旧後に共有queueへ投入し、完了後に全3 clipと補完clipを除く2 clipを固定設定で一括採点します。

pytest 34件が23.52秒で成功し、ruff・mypy・knowledge検査も通過しました。`codex/parent-1739-issue-2444-primary` の完全HEAD `590c2377810762fdea0d5485d10307e612de51eb` をpushし、PR #2449の `Closes #2444` を維持しました。 [実行中のCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45319660691)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"590c2377810762fdea0d5485d10307e612de51eb","resume_on_success":false}}