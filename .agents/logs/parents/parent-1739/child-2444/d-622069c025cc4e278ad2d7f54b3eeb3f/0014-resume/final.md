正式レビューのmajor 2件に対応し、`codex/parent-1739-issue-2444-primary` にpushしました。完全HEADは `9d6630baeaace774f8eba582c20c184e048ea78b` です。PR本文を更新し、`Closes #2444` を維持しました。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

- **人物観測の入力検査**。全カメラでDINOとViTPoseの保存設定に `enabled=true` を要求します。無効ならモデルとカメラを示して停止し、有効な空観測は候補ゼロとして受理します。
- **未見評価の再現手順**。実際に採点した `590c2377810762fdea0d5485d10307e612de51eb` と固定入力をREADME・knowledgeノードに記載しました。採点版では固定ソルバーとの `src` 照合が通り、現行HEADではその保護が働くことを確認しました。 [固定版の再現手順](https://github.com/Motoki0705/tennis-lab/blob/9d6630baeaace774f8eba582c20c184e048ea78b/knowledge/runs/run-multiview-pose-link-recall-20261124/README.md#未見評価の再現)

回帰3ケースを含む関連テスト23件が22.74秒で成功しました。FP32・FastHMRの互換テスト、pre-commit、knowledge検査も通過しました。 [実行中のCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45785995459)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"9d6630baeaace774f8eba582c20c184e048ea78b","resume_on_success":false}}