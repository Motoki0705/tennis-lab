親Issue #2384 配下の子Issue #2019（PR #2399、run `parent-2384-b84612c0e63848859a37ff288fd97c24`）の正式レビューで検出された major 指摘 5 件をすべて修正し、検証・commit・push まで行ってください。

## 修正対象の major 指摘 5 件（すべて採用・修正必須）

### Major 1: 通常の CLI が状態ディレクトリの symlink を検査前に辿る
- 指摘詳細:
  `parent/cli_impl.py:508` は `--state-dir` と既定パスを `.resolve()` してから `Store` を作るため、`register` や `release` は状態ディレクトリが symlink でもリンク先が同一所有者ディレクトリなら辿って lock 取得・registry 更新してしまう。また `dispatcher/judgement.py:292-293` も run ディレクトリを先に `.resolve()` するため、`judgements` が symlink の場合にリンク先へ `input.json` や agent-input を書いてしまう。
- 修正要件:
  - 状態ディレクトリや run ディレクトリを検査前に `.resolve()` で辿らず、解決前のパスを共通ヘルパーへ渡し、ディレクトリ作成も検査済みディレクトリから行う。
  - 通常の CLI 入口（`parent register` / `release` および `dispatcher` 判定経路）から symlink な状態ディレクトリ[local path]`judgements` ディレクトリを渡したときにリンク先を変更せず拒否する回帰テストを追加する。

### Major 2: 状態ディレクトリの差し替えで registry lock と書込み先が分離する
- 指摘詳細:
  `lib/state_files.py:68-98` は直上ディレクトリだけの所有者・書込み権限を検査している。変更可能な上位ディレクトリを経由すると、直上 fd を固定していても次にパスを開き直す際に別ディレクトリを指し得る。`parent/state.py:240-245` で lock 取得後の読込み・保存がパスを開き直すため、旧ディレクトリの lock を保持したまま別ディレクトリへ保存でき排他が分離する。
- 修正要件:
  - 状態ルート（および親ディレクトリ経路）の信頼境界（symlink 拒否・所有者・権限）を一貫して検査し、トランザクション中は lock・読込み・書込み（および `parents/` 配下）を同じ検証済みディレクトリ fd（`dir_fd` ベース等）に結び付ける。
  - トランザクション中のディレクトリ差し替えや信頼境界違反を検出し排他分離を防ぐテストを追加する。

### Major 3: v2 registry は一時ファイルの拒否で部分更新される
- 指摘詳細:
  `parent/state.py:390-407` は親レコード（`parents/<digest>.json`）を先に置換し、索引 `registry.json` を後で書いている。担当解除などで `status` が変わるとき、索引側の危険な `.tmp` を `lib/state_files.py:157-164` が拒否すると親レコードだけが新しくなり、以後は `parent/state.py:349-354` の整合性検査で registry を開けず、一時ファイルを退避しても再試行できなくなる。
- 修正要件:
  - v2 registry の保存で索引書込みの事前検証（および/または書込み失敗時に従前の整合した状態へ戻せる・破損を残さない保存順序／ロールバック）を行い、索引側の拒否や書込み失敗後も従前の整合した状態を読め、原因除去後に再試行できるようにする。
  - 既存の `test_parent_storage_split.py` を拡張し、v2 保存で索引側が拒否・失敗した後の再読込み整合性と再試行成功を検証する。

### Major 4: supervisor の Git エラーと traceback が絶対パスを journal に出す
- 指摘詳細:
  `dispatcher/cli_impl.py:215` は保存済み `config.cwd` を `checkout_barrier()` に渡すが、設定先が移動・削除されていると `lib/checkout_lock.py:341-353` の Git が stderr を継承しているため `report_failure()` の前に絶対パスが journal に出る。また `Schedule._validate()` が `candidates` の各値を検証せず、`dispatcher/state.py:95-98` の `.get()` が壊れた値で `AttributeError` を起こすと CLI の捕捉を抜け、traceback に絶対パスが出る。
- 修正要件:
  - `checkout_barrier()` / `checkout_lock.py` 内の外部 Git コマンドの stderr を捕捉し、そのまま継承出力して絶対パスを漏らさないようにする。
  - `Schedule._validate()` で `candidates` の各エントリ型を厳格に検証し、壊れた値で `AttributeError` が抜けないようにして固定の診断へ変換する。
  - 既存の supervisor 起動テストを拡張し、削除済み `config.cwd` および壊れた `candidates` の両ケースで stdout・stderr の双方に私的な絶対パスが出ず、固定診断が出力されることを検証する。

### Major 5: 壊れた registry の原因と復旧手順が journal で分からない
- 指摘詳細:
  `parent/state.py:245-247, 333-355` の JSON 破損・形式不正・v2 索引とレコードの不一致が `DispatchStore._require_registry()` の診断より先に汎用 `ValueError` 等になり、`dispatcher/diagnostics.py:94-104` が `unclassified_error` と `Inspect private records.` にまとめてしまうため、運用者が journal から対象や復旧手順を判断できない。
- 修正要件:
  - registry 読込み境界で JSON 破損・形式不正・v2 索引とレコードの不一致を型付きの失敗（対象識別子付き）にし、`dispatcher/diagnostics.py` で固定の対象名と README 復旧手順へマッピングする。
  - 公開 CLI の出力で原因・復旧案内・秘匿（私的パスや内容の非露出）を確認する回帰テストを追加する。

## 検証・push・終了ルール
- 関連テスト（新規回帰テスト＋`test_private_state_writes.py`, `test_parent_storage_split.py`, `test_dispatcher_supervisor.py`, `test_execution_basis.py` 等の関連テスト）を実行し全通過を確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2399 本文（`Closes #2019` を維持）に major 5件ごとの修正 commit と検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
