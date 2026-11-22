親Issue #2384 配下の子Issue #2388（PR #2394、run `parent-2384-1e05481e8f5846c09fcad9ca93a0e278`）の正式レビューで検出された major 指摘 3 件をすべて修正し、検証・commit・push まで行ってください。

## 修正対象の major 指摘 3 件（すべて採用・修正必須）

### Major 1: 復旧判定後に加わった未追跡ファイルを強制削除できる
- 指摘詳細:
  `publication.py:325-359` で Git の状態取得直後に未追跡ファイルが追加され lock 理由が `manual` に変わっても、`commit()` が `git worktree remove --force --force` を呼ぶ。journal lock は手動修復や別 Git プロセスを排他せず、二重 `--force` は汚れと lock の両方を無視するため、確認時点の後に加わったデータを消してしまう。
- 修正要件:
  - `git worktree remove --force --force` のように未知の内容や別 lock を無条件に破棄する手順をやめ、削除・再作成の境界で未知ファイル・変更・別 lock（`manual` 等）を絶対に消さない安全な復旧手順へ変更する（例: worktree ディレクトリを空確認しつつ個別に安全撤去・`locked (initializing)` 確認後の unlock + `--force` なし削除／あるいは原子的な検証付き手順で、未知ファイルや `manual` lock へ変わっていれば失敗して `worktree_dirty` 等で保護する）。
  - 状態確認と削除の間で未追跡ファイル追加や lock 理由変更が起きた場合にファイルを削除せず安全に拒否する回帰テストを追加する。

### Major 2: `journal_unavailable` が未検証のディレクトリ名を出力する
- 指摘詳細:
  `reconciliation.py:166-209` で、turn を識別できない壊れた名前空間でもパス中の 32 桁 16 進文字列が `public_key` にそのまま入り、一括照合と `remaining_publications` の両方に出る。非公開の turn ID も同じ形式のため、正規の公開キーと検証できない値を形式だけで公開するリスクがある。
- 修正要件:
  - 読み取れなかった（検証できない）名前空間の `journal_unavailable` 行には、パスのダイジェスト（SHA-256）、固定理由、判明した親番号のみを使い、未検証の 32 桁 16 進ディレクトリ名を `public_key` 等に出力しない（`public_key: None` 等にする）。
  - 壊れた名前空間のパスに 32 桁 16 進文字列が含まれていても、一括照合と `remaining_publications` にその生文字列が出ないことをテストで検証する。

### Major 3: ディレクトリを読めない名前空間を残件ゼロと誤判定する
- 指摘詳細:
  `reconciliation.py:74-110` で、正規の issue ディレクトリ位置や正規の親ディレクトリ位置に通常ファイル（ディレクトリではない型不一致）があると、`_journal_paths()` が再帰呼び出しに `namespace=True` を渡さないため黙って飛ばし、一括照合・`remaining_publications` が空、終了 checkpoint も `remaining_complete=true`・`error=null` になる。
- 修正要件:
  - 正規の親ディレクトリ・issue ディレクトリ等の名前空間位置に型不一致（通常ファイル・symlink・読取不能等）がある場合に黙って飛ばさず `journal_unavailable` として残す。
  - 一括照合・`remaining_publications`・終了 checkpoint のすべてで残件ゼロ（`remaining_complete=true`）にならず、該当名前空間が報告される回帰テストを追加する。

## 検証・push・終了ルール
- 関連テスト（新規回帰テスト＋既存の `worker_logs` / `decision_logs` ユニット・E2E テスト）を実行し全通過を確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2394 本文（`Closes #2388` を維持）に major 3件ごとの修正 commit と検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2394,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
