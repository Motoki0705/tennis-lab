## 結果

- reviewのWeb検索を明示的に無効化し、初回・resumeの引数を固定しました。
- シグナル終了を `cli_signal` と終了シグナル番号で記録します。SIGTERM/SIGKILL、stop・timeout・検査後cleanupとの区別を検証しました。
- 対象264件成功（53.25秒）、pre-commit成功。Codex CLI 0.168.0の短いread-only runも成功しました。
- 実検索を伴う実機再現・OSの送信元監査は未確認です。既存の単独テストのimport衝突はmainでも再現し、PRに記載しました。
- validatorは未指定、試行・完了とも0回です。

## branch・commit・PR・CI

- branch: `codex/review-cli-signal-web-search`
- HEAD: `d3b78c032`（worktree clean）
- PR: [#2000](https://github.com/Motoki0705/tennis-lab/pull/2000)
- CI: [最終HEADのPythonを含め全工程成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43570891568)

## issueへの報告案

過去2runは、重複IDの検査エラー後にrunnerがCLIを回収した流れと整合する。現行mainでは保存された検索イベントを受理できた。PR #2000で検索設定の固定とシグナル終了の独立記録を追加し、対象264件・CIの成功を確認した。

## 親への判断依頼

なし。統合レビュー・merge・merge後のmain CI確認は親担当です。