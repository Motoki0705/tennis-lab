# PR #2717 正式レビュー major 指摘3件の修正（Turn 3）

正式レビュー（run `parent-2690-25d70f92df35458b83a99864637d924e`, reviewed HEAD `3060eaa20705e5ad0dfa6aca8e4ea609fc4ffc0d`）で、以下の major 指摘 3 件が返りました。すべて採用して修正し、回帰テスト・文書・ローカル検証を通して commit / push してください。

## Major 指摘 3 件と対処方針

### 1. 不正な setup 結果と欠落した call 計測を成功として受理する（`scripts/ci_metrics/report.py`）
- **指摘内容**:
  計測元 run の shard 8 を読み込み、成功ケース 1 件の setup outcome を未定義値に変更して call 計測を除くと、元の JUnit に対して `successful_test_baseline=true`、`issues=[]` のまま、短くなった計測値が profile 生成に渡る。`report.py` は setup が `passed` の場合だけ call を要求し、未知の outcome を `passed` に分類している。集計件数と inventory が一致するため `update_profile.py` の coverage 再照合でも検出できない。
- **修正要求**:
  - `scripts/ci_metrics/report.py` の phase 判定で、正当な setup skip / xfail / error / failure 等の既存契約は維持しつつ、**未定義・未知の phase outcome**（`passed` / `failed` / `skipped` / `xfailed` / `xpassed` / `error` 等、実際の pytest/plugin が記録する正当な値以外）および **不完全な phase**（call 欠落等）を明示的に検出して `issues` に記録し、`successful_test_baseline=False` として拒否すること。
  - 上記の改ざん・不完全観測（setup outcome を未定義値にして call 計測を除いたケース）に対して、`report.summarize` / `inspect_shards` / `build_profile` / `update_profile` のすべてで拒否され、既存 `cpu-file-times.json` が保全される回帰テストを追加すること。

### 2. `--latest-main` が不正な plan を持つ候補で探索を打ち切る（`scripts/ci_metrics/update_profile.py`）
- **指摘内容**:
  新しい候補の `plan_id` が不正で、次に適格な候補がある入力を与えると、`read_plan` の `ValueError` が伝播し 2 件目に到達しない。`_select_run` が候補を拒否して続行するのは `IneligibleRun` の場合だけで、plan の構造不正や coverage JSON の解析エラーは候補の拒否理由に変換されない。
- **修正要求**:
  - `scripts/ci_metrics/update_profile.py` で、候補 run 固有の証拠データ不正（`read_plan` の `ValueError`、`json.JSONDecodeError`、`KeyError`、`TypeError`、`zipfile.BadZipFile`、coverage/artifact/shard 構造不正など）を `IneligibleRun` として扱い、run ID と具体的な拒否理由を記録して次候補の探索を続行すること。
  - 一方、`--run-id` で明示指定された run の検証失敗、および GitHub API 通信障害（`subprocess.SubprocessError` 等）や出力先などのローカル I/O 障害（`OSError` 等）は探索続行せず、出力 profile を保全したままエラー終了すること。
  - 以下を回帰テストで検証すること:
    1. `--latest-main` で先頭候補の `plan_id` 不正や coverage JSON 解析エラーがあっても拒否理由を記録して 2 件目の適格候補へ到達し成功すること。
    2. `--run-id` 明示指定で不正 plan / 不正 JSON の場合に失敗し、既存出力 profile が保全されること。
    3. 通信障害やローカル I/O 障害では探索を誤魔化さず失敗し、既存出力 profile が保全されること。

### 3. 最新の全件 inventory に対する A/B/C の割当比較（特に B）が欠けている（`sharding*.md` / `sharding-comparison.json`）
- **指摘内容**:
  3 言語のオフライン A/B/C 表と `_compare` の入力は計測元 run (`45955929416`) の 26,189 件に固定されている。後続の全件 Actions で確認した 26,208 件について、`sharding-comparison.json` にあるのは A/C の結果で、B の割当・負荷・未計測件数が欠けている。最新の全件 nodeid 集合を固定して 3 条件を比較し、増分ケースが割当グループと負荷推定にどう影響するかを記録すること。
- **修正要求**:
  - まず Major 1・2 の回帰テスト追加まで完了した **この PR 最終コードの最新全件 CPU inventory**（実コレクションまたは全件 nodeid 集合）および、実 Actions 比較で用いた 26,208 件集合の両方（あるいは最新全件 inventory を固定して、26,189 → 26,208 → 最終 HEAD の増分を明示）について、**A（旧重み）/ B（新重み・2 split file）/ C（新重み・4 split file）の 3 条件** を同一 nodeid 集合上で厳密に比較し、各 shard 0〜9 の推定負荷・min/max/mean・max/mean 比・未計測 file/split case 数、さらに増分ケース（`test_ci_shards.py` / `test_ci_growth.py` 等への追加ケース）がどの割当グループ（ファイル単位なのでファイル数・推定負荷への影響、未計測件数への影響）にどう作用するかを計算して記録する。
  - `scripts/ci_metrics/sharding-comparison.json` と 3 言語文書（`scripts/ci_metrics/sharding.md`, `sharding.ja.md`, `sharding.zh-CN.md`）を更新し、計測元 run（26,189件）での比較と、最新全件 inventory（増分反映後）での A/B/C 3条件比較および増分影響の説明を明確・整合的に記載する。
  - PR #2717 本文も、最新 HEAD と修正内容・最新全件 A/B/C 比較に合わせて更新する（ローカル絶対パスは載せない）。

## ローカル検証と引き渡し
- worktree 内で以下をすべて実行して通すこと：
  - `.venv/bin/python -m pytest tests/e2e/development/test_ci_artifacts.py tests/e2e/development/test_ci_growth.py tests/e2e/development/test_ci_metrics.py tests/e2e/development/test_ci_shards.py tests/e2e/development/test_ci_test_lanes.py tests/e2e/development/test_readme_layout.py`
  - `.venv/bin/python -m ruff check scripts/ci_metrics tests/e2e/development`
  - `.venv/bin/python -m mypy --follow-imports=skip scripts/ci_metrics`
  - 変更対象ファイルの pre-commit 確認
- commit・push して PR #2717 を更新し、**PR CI の完了前に** 修正 commit の完全な HEAD SHA・各指摘の修正内容・追加回帰テストの実行結果を報告して `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返してください。
