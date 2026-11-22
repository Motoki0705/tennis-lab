PR #2399（子Issue #2019）のマージ前確認で、実機（Ubuntu / WSL2 の既定環境）で致命的な運用ブロッカーとなる上位ディレクトリ権限検査を1点修正してください。

## 修正内容（User Private Group 上位ディレクトリの `0775` 対応）
- 現象:
  前回の報告どおり、Ubuntu / Debian / WSL2 の既定 umask `0002`（User Private Group）では `[local path]` および `[local path]` が `0775`（例: `kamimura:kamimura`, gid=1000, 他メンバー0人）で作られています。
  現在の `.agents/ops/lib/state_files.py` `_check_directory()` は非 leaf の上位ディレクトリに対しても `info.st_mode & 0o022` を一律拒否するため、このままマージすると実機のすべての `parent` / `dispatcher` コマンドが `[local path]` で即座に失敗します。
- 修正方針（fail-closed を保ちつつ単一ユーザー専用グループのみ許可）:
  - **leaf ディレクトリ（状態ディレクトリ本体）と全状態ファイル**: 従来どおり `st_mode & 0o022`（`0o020` も `0o002` も）を厳格に拒否する。
  - **非 leaf の上位ディレクトリ**:
    - world-writable（`st_mode & 0o002`）は非 sticky の場合、従来どおり厳格に拒否する。
    - group-writable（`st_mode & 0o020`）かつ `st_mode & 0o002 == 0` の場合は、以下の**すべて**を POSIX の `pwd` / `grp` から確実に検証できた場合（単一ユーザー専用の User Private Group であり、実効的に他ユーザーからの書込みが不可能である場合）のみ許可する：
      1. `info.st_uid == owner`（現在の実効ユーザー自身が所有者）
      2. `pwd.getpwuid(owner)` が取得でき、`pw_entry.pw_gid == info.st_gid`
      3. `grp.getgrgid(info.st_gid)` が取得でき、`gr_mem` に `pw_entry.pw_name` 以外のユーザーが含まれない
      4. `pwd.getpwall()` 内に `pw_uid != owner` かつ `pw_gid == info.st_gid` となる別ユーザーが存在しない
    - 上記いずれか 1 つでも満たさない場合や `KeyError` / lookup 失敗時は fail-closed で `StateFileError(..., "shared_write", ...)` を送出する。
  - 併せて `.agents/ops/dispatcher/README.md` の記述も実態に合わせて簡潔に整合させる。

## 追加テスト
`tests/unit/agents_ops/test_private_state_writes.py` に以下を追加すること：
1. 非 leaf の上位ディレクトリが `0775` かつ所有者の単一ユーザー専用グループ（UPG）の場合は許可されるテスト。
2. 非 leaf の上位ディレクトリが `0775` でも、(a) グループに別ユーザーが含まれる (`gr_mem`)、(b) `pwd.getpwall()` に同一 `pw_gid` の別ユーザーがいる、(c) 所有者が root 等の別人、(d) グループ lookup が失敗する、の各ケースで `shared_write` で拒否されるテスト。
3. 非 leaf の上位ディレクトリが non-sticky `0777` の場合は `shared_write` で拒否されるテスト。
4. leaf 状態ディレクトリ自体が `0775` の場合は UPG であっても引き続き `shared_write` で拒否されるテスト。

## 検証・push・終了ルール
- `tests/unit/agents_ops/test_private_state_writes.py` および関連テストを実行し全通過を確認する。
- 実機の `[local path]` 相当の権限配置（`0750` → `0775` → `0775` → `0755` → `0700`）で `open_state_directory` が通ることも確認する。
- pre-commit（ruff・mypy）を通して commit・push し、PR #2399 本文（`Closes #2019` を維持）に本修正コミットと検証結果を追記する。
- push が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<40桁HEAD SHA>` は push した実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
