# #1878 issue:standalone ラベル導入の実装

担当Issue: https://github.com/Motoki0705/tennis-lab/issues/1878 （親自身を作業対象とする未分割Issue。子Issueは作らない）
worktree: `.`（branch `codex/issue-standalone-label`、origin/main起点）

#1878本文の「全体方針・合意事項」と「全体の完了条件」が要件の正本。まず本文を全部読むこと。
今回のworkerの担当は **コード・規約・テンプレート・テストの変更とPR作成まで**。
以下は親が行うので、あなたは行わない: GitHub上のラベル作成（`sync_github_labels.py --apply`）、既存11件へのラベル付与、Issueのclose・コメント投稿、merge。

## 親が決めた設計（この方針で実装する。不都合があれば理由を最終報告で返す）

1. **ラベル定義**: `.github/labels.yml` の `hierarchy` に `issue:standalone` を追加。色は親子のオレンジ系と区別できる `7DD3FC`。
   説明は「親にも子にも属さず、単独の作業単位として扱うIssue（.agents/ops/issues/README.md）」。
   manifestを読む既存コード（`scripts/sync_github_labels.py`、`.agents/ops/ops-report/opsreport/triage.py` の `LABEL_MANIFEST` 等）が新ラベルで壊れないか確認する。
2. **共通定数**: `.agents/ops/issues/format.py` に `STANDALONE_LABEL = "issue:standalone"` を置き、`issues/__init__.py` から公開する。文字列直書きを新たに増やさない。
3. **生成時の排他**: `issues/cli.py` の `render_spec()` は任意 `labels` に `STANDALONE_LABEL` が含まれる場合も、親・子ラベルと同じく拒否する（親・子specへの混入防止）。`render` に standalone role は追加しない（新しい単独専用CLIは増設しない）。
4. **整合検査**: `issues/consistency.py` の `check_consistency()` に、standaloneラベルを持つIssueが親・子の役割（親子ラベル、正式なparent/children、子一覧節、`親issue:` 参照、親prefix付きタイトル等の既存の推定根拠）と矛盾する場合の問題コード（例 `label.standalone`）を追加する。
   - 有効な単独（ラベルのみ・関係なし）、子0件の未分割親、親兼子、移行前の役割ラベルなしIssueは**新たな問題を出さない**（役割ラベルなしを単独と断定しない）。
   - standaloneのIssue本文には親子の必須節を要求しない（人間テンプレートの自由書式を許す）。
   - 取得失敗を「関係なし」に読み替えない既存契約を維持する。
5. **単独用テンプレート**: `.github/ISSUE_TEMPLATE/01_bug.md`・`02_feature.md`・`03_research.md` の `labels` に `issue:standalone` を追加し、冒頭コメントの案内文も実態に合わせる。04〜07には付けない。
6. **自動起票**: `.agents/ops/ops-report/opsreport/organization.py` の `render_node()` で role=standalone のときだけ `labels` に `STANDALONE_LABEL` を加える。parent・child・reuse（既存issueのラベル保持）とレポート容器には付けない。
   **保存済みtriage計画・journalとの互換を必ず確認する**: `organization_apply.py` の `_guard` 等は、作成済みissueのラベルが保存draftと一致することを検査している。コード更新前に保存されたdraft（standaloneなし）で再開・復旧したとき、再renderした結果との差で停止したり、保存draftと異なるラベルで再POSTしたりしないこと。既存の振る舞い（保存draftを正とする等）を読み、必要ならテストで固定する。
7. **単独→親・子への移行**:
   - 親CLI `.agents/ops/parent/session.py` の `create_child()`: 親Issueに `STANDALONE_LABEL` が残っていれば「移行時にstandaloneを外す」旨で拒否する（既存の「migrate the parent using issues before splitting」と同じ扱い）。子のラベル合成（既存ラベル ∪ draftラベル）からは `STANDALONE_LABEL` を除き、ラベルが既に一致していればPATCHしない（再実行で余分なラベル変更を出さない）。
   - ops-reportで既存の親へ子を加える経路（role=child + 既存親番号）で、親側にstandaloneが残る矛盾を既存のfamily検証で拒否できるか確認する。
   - 昇格（単独→親）と編入（単独→既存親の子）の手動手順を `.agents/ops/issues/README.md` の段階移行節に明記する: 現状の再取得、本文・タイトルの整形、`gh issue edit --add-label issue:parent|issue:child --remove-label issue:standalone` を1回の編集で行うこと、編入時の sub-issue POST（既にあれば再追加しない）、親一覧の更新、`issues check` で0になるまで照合、途中失敗時は実状態から不足分だけ揃えること。
8. **規約・検索**: `.agents/ops/issues/README.md` の役割表・テンプレート表・ラベル軸表を更新し（「単独タスクには親子ラベルを付けない」等の旧記述を置換）、意味・排他・付与しない場合（子0件の親、ops-report容器、PR、参照/依存だけを理由にした判定）・ラベルが実行承認や担当予約を生まないことを書く。#1878本文の3つの検索例を同READMEに載せ、3つ目が確認候補の抽出であり自動付与の根拠でないことを書く。同じ事柄を他のREADMEへ重複させず、必要ならリンクで参照する。
9. 他に `PARENT_LABEL`[local path]`CHILD_LABEL` で役割を判定している箇所（`parent/session.py` の `task_title`、`board/service.py`、`dispatcher/judgement.py` の route、`project_progress/progress.py`、`issue_supplement`）を確認し、standaloneが付いても既存動作が変わらないことを確かめる。挙動を変える必要がなければ変更しない。

## テスト

[tests README](tests/README.md#テスト作成方針) に従って設計する。既存の `tests/e2e/agents_ops/test_issue_format.py`、`test_ops_report_triage.py`、`test_adopted_proposal_issues.py`、親セッションの `create_child` テストなどを再利用し、少なくとも次を検証する:
- ラベルmanifestとテンプレートの対応（01〜03にstandalone、04〜07に無し、manifestにある）
- render_specがstandalone混入を拒否
- check_consistency: 有効な単独／子なし親／親兼子[local path] の各ケース
- triage render_node の role別ラベルと、保存draftでの再開互換
- create_child: 親にstandaloneがあると拒否、子ラベル合成でstandaloneが落ち、再実行で余分なPATCHが出ない（既存fake GitHubで）

関連テストを実行し（例: `.venv/bin/python -m pytest tests/e2e/agents_ops/test_issue_format.py tests/e2e/agents_ops/test_ops_report_triage.py tests/e2e/agents_ops/test_adopted_proposal_issues.py -n 4 -q` と、create_childを含む親CLIテスト）、失敗が変更と無関係なら origin/main で同じテストを比べて示す。

## 成果

- コミット（pre-commit通過）とPR。PR本文は日本語で、`Closes #1878` は**付けない**（11件の移行とGitHub反映が残るため）。代わりに `Refs #1878` とする。
- PRにはpushごとのCI（対象HEADの `Python`）結果を確認して最終報告へ含める。
- 作業時間の目安は最大3時間。上記9項目と テストが揃うまで止めない。止めてよいのは、要件どうしの矛盾・既存契約と両立しない設計点が見つかった場合だけで、そのときは根拠と選択肢を最終報告で返す。

最終報告: PR番号・HEAD・変更ファイル要約・各完了条件(#1878の1〜4,6のうちコードで担保した部分)に対する証拠・実行したテストと結果・CI結果・未確認事項。
