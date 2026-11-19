# 中断したターンの続き（branch の扱いを変更）

前のターン（「#2032: 判断を待たずに進められる2PRの実装」）は、メモリ不足とその後の WSL の再起動で途中で終了しました。WSL のメモリ上限は 24GB から 40GB に上がっています。

## 親が行ったこと

あなたは2本目のために worktree を新しい branch `codex/board-display-discovery`（main から作成）へ切り替え、未commitの変更を置いていました。しかし、この委任は branch `codex/board-project-io` に結び付いているため、別 branch のままでは再開も成果の記録もできません。そこで親が次を行いました。

- 未commitの変更（`board/README.md`・`board/display.py`・`board/receivers.py`・新規 `tests/e2e/agents_ops/test_board_io.py`）を、local branch `codex/board-display-discovery` に WIP commit `150113057` として保存した（hooks は通していない。push していない）。
- worktree を `codex/board-project-io`（`143c835f5`、PR #2084 の HEAD）へ戻した。

## 依頼（計画の変更）

- **2本目は別PRにせず、PR #2084 に含めてください。** WIP commit `150113057` の変更を `codex/board-project-io` へ取り込み（cherry-pick など）、残りの実装とテストを仕上げ、ruff/mypy（pre-commit）を通して正式な commit にし、push してください。PR #2084 の題名と本文を、Project I/O と board 表示・Codex sessions 探索の両方を含む内容に更新してください（`References #2032` のまま。`Closes` は fence のPRに付けます）。
- worktree の branch は今後も `codex/board-project-io` から変えないでください。
- 取り込みが終わったら、local の `codex/board-display-discovery` は削除してよい。
- fence はこのターンでも実装しないでください。
- 前のターンの他の指示（テスト設計、routes のロック順を変えない、など）と完了条件はそのままです。対象HEADで CI（Python）の成功を確認してください。時間の上限は、このターンから5時間です。
