# 共有memoryの公開CLI

`workflow`、`decision`、`environment`、`gotcha` のエントリを最新の remote `main` に反映する。エントリを選び、根拠を確かめる手順は [memory skill](../../skills/memory/SKILL.md) を参照。

## コマンド

リポジトリのルートから実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory add --spec add.json --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory update --spec update.json
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory delete --layer gotcha --slug example --reason '参照先の仕様へ統合済み'
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory batch --spec - < operations.json
```

| コマンド | JSON と適用条件 |
|---|---|
| `add` | `layer`、`slug`、`title`、`summary`、`applies_to`、`source`、`evidence`、`citations`、`body` が必須。`created` と `last_verified` は省略時に実行日の値を使う。`reason` は任意。既存のエントリは拒否する。 |
| `update` | `layer`、`slug`、非空の `reason`、変更するフィールドを指定する。省略したフィールドは既存値を引き継ぐ。エントリが無い場合や、指定した内容が同じ場合は拒否する。 |
| `delete` | `--layer`、`--slug`、非空の `--reason` を指定する。代わりに `--spec` で `layer`、`slug`、`reason` を渡せる。エントリが無い場合は拒否する。 |
| `batch` | 各オブジェクトに `op` を加えた JSON 配列。空配列や同一 `(layer, slug)` の複数操作は拒否する。全操作と生成文書を1コミットにまとめる。 |

`--spec` は UTF-8 の JSON ファイルか `-`（標準入力）を指定する。`--repo` は作業ツリーのパスで、既定はカレントディレクトリ。`--remote` は既定で `origin` を使う。Git の実効 fetch・push URL が同じ GitHub リポジトリを指していることを照合する。公開には Git の push 権限と `gh` の認証が必要。

単独の `add` では次のオブジェクトを指定する。

```json
{
  "layer": "workflow",
  "slug": "example",
  "title": "確認した作業上の観察",
  "summary": "SUMMARY.md に載せる1行の要点。",
  "applies_to": "all",
  "source": "観察した issue・PR・会話の日付",
  "evidence": "観察が今も成り立つかを確かめる方法",
  "citations": [{"path": "AGENTS.md", "symbol": "#開発ルール"}],
  "body": "観察の詳細と、次の作業でどう使うかを書く。"
}
```

`slug` は kebab-case で `index` は予約名。`applies_to` は `all`、`claude`、`codex`、`gemini` のいずれか。日付は `YYYY-MM-DD`。`citations` の `path` はリポジトリ相対パス。`symbol` に本文の文字列や `#` 付きの Markdown アンカー、`note` に補足を指定できる。該当する参照が無ければ `[]` を指定する。上限は [schema.py](schema.py) に定義される。

## 限定公開例外

1. 一時 worktree の `FETCH_HEAD` と index に最新 `main` を取得する。Git が記録したツリーを読み、変更を適用する。呼び出し元の作業ツリー、ブランチ、index を保全するため、対象コミットをファイルとして checkout せず、Git のオブジェクトから検査する。
2. `SUMMARY.md` と4層の `index.md` をエントリから生成する。全共有memoryの形式、構造化引用と Markdown アンカー、相対リンクの symlink 境界、上限、機密やローカル参照、生成文書のバイト一致を検査する。コミットの親、実差分のパス・モード・内容を照合する。空差分や rename は拒否する。
3. 検査を通過したら `shared-memory/<UTC時刻>-<短いID>` へ push し、変更一覧・操作理由・検査結果を記した日本語の PR を作る。GitHub の PR 状態、base・head のリポジトリと ref、SHA、差分全ページを照合して、`sha` を指定して即マージする。
4. マージが HTTP 405 または 409 で拒否されたとき、または直前の照合で base が動いたときは、Git の `main` が先に進んだことを確認する。同じ操作を新しい base で検査し直し、旧 SHA を lease に指定して同じブランチに push する。最大3回載せ直す。通信断、曖昧な応答、検査の失敗はエラーとして扱う。マージ API は head を固定し、PUT の直前に GitHub の `refs/heads/main` を直接照合する。[GitHub REST API のマージ仕様](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)
5. マージ成功を確認したら、lease 付きで専用ブランチの削除を試みる。一時 worktree は各回の検査終了時に取り除く。

[CI workflow](../../../.github/workflows/ci.yml) の除外は共有4層の直下の Markdown と共有 `SUMMARY.md` に適用する。専用の `ops-report` 領域と、CLI・検査・workflowを含む実装PRには通常の条件を適用する。

## 結果と検査

成功時は標準出力へ JSON を返し、終了コードは `0`。`base`、`head`、`files`（パスと `added` / `modified` / `removed` の対応）、`operations`、`validation` を含む。公開後は `pr`、`url`、`merge_sha`、`retries` を返す。マージ後のブランチ削除に失敗した場合は `branch_cleanup_error` に理由を返す。

`--dry-run` では fetch、適用、生成、コミット・差分検査までを実行して同じ形式で報告する。`dry_run` は `true`、`pr`・`url`・`merge_sha` は `null`。GitHub への変更操作は発生しない。

検査や公開の失敗は標準エラー出力に理由を出して `1`、引数の誤りは `2`、[checkout lock](../lib/README.md#対話cliの保護) の取得失敗は `75` を返す。push 後に止まった場合はエラーにブランチ名と、取得済みなら PR 番号を含める。応答が不確かだった場合は、その PR とブランチの状態を確認してから再実行する。

次のコマンドで手元の共有memoryを点検できる。`audit` は壊れた引用、鮮度、上限の使用量を JSON で返す。期限切れや検査違反があれば終了コードは `1`。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit
.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py \
  tests/e2e/agents_ops/test_shared_memory_publish.py \
  tests/e2e/agents_ops/test_agents_docs.py \
  tests/e2e/agents_ops/test_interactive_checkout.py \
  tests/e2e/development/test_ci_test_lanes.py::test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs \
  -n 2 -q
```

## Python API

`shared_memory.operations.apply_operations(source, operations, *, today=None)` は checkout の `Path` または `Tree` に対して、`batch` と同形式の操作配列を検査する。返り値はリポジトリ相対パスから書き込む `bytes` または削除を表す `None` への対応で、変更が必要な生成文書も含む。入力ツリーへの書き込みは発生しない。

`shared_memory.publish.publish_operations(repo, operations, *, remote="origin", dry_run=False, today=None)` は、最新 `main` の fetch から公開までを実行し、JSON 出力と同じ項目を持つ `Publication` を返す。`today` を指定した場合は、日付既定値と検査に同じ日を使う。入力・検査・公開の拒否や曖昧な応答では例外を送出する。AutoDream の Prune & Index はこの API に操作配列を渡す（#2326で実装中）。
