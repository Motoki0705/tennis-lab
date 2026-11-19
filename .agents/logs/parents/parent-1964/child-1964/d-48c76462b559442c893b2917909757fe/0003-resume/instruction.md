# PR #1971 と main の衝突解消

統合レビューの指摘2件は解消済みで、親が fixes_verified を記録した（対象 HEAD 188e1fca4）。
merge gateに進もうとしたところ、PR #1971 が main と衝突していた（CONFLICTING）。
origin/main を branch `codex/issue-proposal-label` へ merge（rebase・force push はしない）し、衝突を解消すること。

- 衝突は `tests/e2e/agents_ops/test_ops_control.py`。main では `.agents/ops/ops_control/`（README・targets.py）、`parent/README.md`、`parent/session.py` も自動mergeされる。main 側の変更（30 commit）の意図を読み、両方の契約を保つ。
- 新しい `ops_control` の対象（targets.py）が parent CLI の入口・timer・unit を列挙する仕組みなら、今回追加した `improvement-expire` の timer/service や新コマンドを、そこへ登録する必要があるかを確認する。必要なら対応する。
- merge後に関連テスト（ops_control・improvement・parent session・dispatcher）を実行し、pushして対象 HEAD の CI `Python` を確認する。
- PR本文の閉じる記述は `Closes #1964` のまま維持する（`Refs` に戻さない）。証拠節を更新する場合も、この行を変えない。

最終報告に含めるもの:
- merge commit・HEAD
- 衝突の解消内容と、main側の変更で影響を受けたファイル
- ops_control への登録要否の判断
- 実行したテストと結果
- CI結果
