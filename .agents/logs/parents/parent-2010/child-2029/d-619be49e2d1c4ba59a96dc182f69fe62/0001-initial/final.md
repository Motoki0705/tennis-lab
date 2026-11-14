[PR #2038](https://github.com/Motoki0705/tennis-lab/pull/2038) を作成しました。`Closes #2029` 付きです。

branch: `codex/registry-txn-client`  
HEAD: `fe1593005`（push済み、worktree clean）

transaction用／ロック外clientを分離し、review 6入口などのI/O解放・再照合、競合時の終了値2、spawn前の通信なしrollbackを実装しました。

- 関連e2e 927件、review等226件、最終ロック・競合20件、unit等1237件が成功（一部重複）。
- ruff・pre-commitのmypy成功。
- 対象HEADと完全一致する [Python CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43595987440/job/132455307909) を確認済み。

CIで判明したdsh fixtureのumask依存もテスト側で修正しました。対象外のローカルexplain_html 4失敗は基準branchでも再現し、PRに記録しています。validatorは未指定のため試行0／完了0です。

残作業は親による統合レビュー・merge・mainのCI確認です。