# 共有memoryの公開CLI

`workflow`、`decision`、`environment`、`gotcha` のエントリを最新の remote `main` に反映する。エントリを選び、根拠を確かめる手順は [memory skill](../../skills/memory/SKILL.md) を参照。

## コマンド

リポジトリのルートから実行する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory add --spec add.json --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory update --spec update.json
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory delete --layer gotcha --slug example --reason '参照先の仕様へ統合済み'
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory batch --spec - < operations.json
PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory repair-index --reason '生成文書のずれを修復' --dry-run
```

| コマンド | JSON と適用条件 |
|---|---|
| `add` | `layer`、`slug`、`title`、`summary`、`applies_to`、`source`、`evidence`、`citations`、`body` が必須。`created` と `last_verified` は省略時に実行日の値を使う。`reason` は任意。既存のエントリは拒否する。 |
| `update` | `layer`、`slug`、非空の `reason`、変更するフィールドを指定する。省略したフィールドは既存値を引き継ぐ。エントリが無い場合や、指定した内容が同じ場合は拒否する。 |
| `delete` | `--layer`、`--slug`、非空の `--reason` を指定する。代わりに `--spec` で `layer`、`slug`、`reason` を渡せる。エントリが無い場合は拒否する。 |
| `batch` | 各オブジェクトに `op` を加えた JSON 配列。空配列や同一 `(layer, slug)` の複数操作は拒否する。全操作と生成文書を1コミットにまとめる。 |
| `repair-index` | `--reason` を添えて `SUMMARY.md` と各層の `index.md` を再生成して公開する。生成文書に実差分があるときだけ適用する。 |

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

`update` と `delete` の spec では、判断に使ったエントリの生バイトを SHA-256（小文字16進64桁）にして `expected_sha256` に渡せる。`delete` の引数形式では `--expected-sha256` を使う。初回の適用と載せ直しのたびに最新の `main` のエントリと照合する。エントリが更新・削除されていた場合は再判定が必要なことを報告し、公開を中止する。

## 限定公開例外

1. remote の `refs/heads/main` から取得対象の SHA を確定し、`--no-write-fetch-head` で取得する。Git オブジェクトから変更候補を構築する。一時 worktree、index、参照の登録が発生しないため、SIGTERM で中断した場合も回収対象は増えない。
2. `SUMMARY.md` と4層の `index.md` をエントリから生成する。全共有memoryの形式、構造化引用と Markdown アンカー、相対リンクの symlink 境界、上限、機密やローカル参照、生成文書のバイト一致を検査する。コミットの親、実差分のパス・モード・内容を照合する。空差分や rename は拒否する。
3. 検査を通過したら `shared-memory/<UTC時刻>-<短いID>` へ push し、変更一覧・操作理由・検査結果を記した日本語の PR を作る。PR 状態、両ブランチのリポジトリ・ref・SHA、差分全ページを照合する。GitHub の `updateRefs` は両 ref の `beforeOid` を一つのトランザクションで検査し、`main` を検査済みのコミットへ fast-forward する。[GitHub GraphQL の `updateRefs`](https://docs.github.com/en/graphql/reference/git#updaterefs)
4. 照合中に `main` が動いた場合や、トランザクションが明示的に拒否された場合は、Git の `main` が先に進んだことを確認する。同じ操作を新しい base で検査し直し、旧 SHA を lease に指定して同じブランチへ push する。載せ直しは最大3回。通信断、曖昧な応答、検査の失敗では停止する。
5. 更新の応答を受けたら、検査済みのコミットが remote `main` に含まれ、同じ head の PR がマージ済みになったことを照合する。照合できた場合は lease 付きで専用ブランチの削除を試みる。確定できない場合は PR とブランチを特定するエラーを返す。[GitHub の間接マージ](https://docs.github.com/en/pull-requests/reference/pull-request-merges#indirect-merges)

共有 `SUMMARY.md` と4層の直下にある kebab-case の Markdown がフルCIの除外対象。`paths-ignore` は共有 `SUMMARY.md` と英小文字・数字だけの直下の `.md` を除外する。Actions の文字クラスは英数字を対象とするため、ハイフンを含む名前は `scope` ジョブがイベントの Git 差分を [共通のパス規則](layout.py) で照合する。名前違反、ネスト、非Markdown、`ops-report` や実装ファイルを含む差分は通常のCIで扱う。[CI workflow](../../../.github/workflows/ci.yml) · [GitHub Actions のパターン仕様](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#filter-pattern-cheat-sheet)

## 結果と検査

成功時は標準出力へ JSON を返し、終了コードは `0`。`base`、`head`、`files`（パスと `added` / `modified` / `removed` の対応）、`operations`、`validation` を含む。生成文書の修復では `index_repair_reason` を返す。公開後は `pr`、`url`、`merge_sha`（fast-forward した `head`）、`retries` を返す。マージ後のブランチ削除に失敗した場合は `branch_cleanup_error` に理由を返す。

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
  tests/e2e/development/test_ci_test_lanes.py \
  -n 2 -q
```

## Python API

`shared_memory.operations.apply_operations(source, operations, *, today=None)` は checkout の `Path` または `Tree` に対して、`batch` と同形式の操作配列を検査する。返り値はリポジトリ相対パスから書き込む `bytes` または削除を表す `None` への対応で、変更が必要な生成文書も含む。入力ツリーへの書き込みは発生しない。

`shared_memory.publish.publish_operations(repo, operations, *, remote="origin", dry_run=False, today=None, require_fresh=False, guard=None)` は、最新 `main` の fetch から公開までを実行し、JSON 出力と同じ項目を持つ `Publication` を返す。`today` を指定した場合は、日付既定値と検査に同じ日を使う。入力・検査・公開の拒否や曖昧な応答では例外を送出する。

`shared_memory.publish.publish_index_repair(repo, *, reason, remote="origin", dry_run=False, today=None, require_fresh=False, guard=None)` は、エントリ操作が無い場合に生成文書だけを検査・公開する。生成差分が無ければ無変更としてエラーを返す。AutoDream は両 API の `require_fresh` と `PublicationGuard` を使い、候補の鮮度、観測した SHA の一致、公開インテントの記録を必須にする。

## AutoDream

### 発火条件

前回統合完了から **168時間以上**経過したうえで、前回確認した `main` 以降の件数が **main のコミット10件以上**または **共有memory更新3件以上**に達すると起動する。コミットは `main` の first-parent 上で1件ずつ数える。共有memory更新は、同じ区間で `SUMMARY.md` または共有4層直下の Markdown に first parent から差分のあるコミットを数える。初回は経過時間の条件を満たした扱いとし、first-parent 全履歴に件数条件を適用する。

引用の破損、上限超過、最終確認から90日を超えたエントリ、生成文書のバイト差分は直ちに発火する。引用の破損には `citations` 欄の欠落、型・要素の不正、参照先の破損を含む。手動の `--force` も発火する。条件未達なら計測値と `skipped_trigger` を記録して終了する。

### 実行の流れ

1. **Orient** は最新の remote `main` の SHA を固定し、共有4層のエントリの生バイトと SHA-256、生成文書を取得する。
2. **Gather Signal** は `audit` と `plan_sync` で問題を点検する。上限の80%以上に達した文書、題名・要約の一致による重複候補、履歴上で変更・削除された引用先、規則や skill への昇格候補を示す。
3. **Consolidate** は専用の `ops-report` memory を除いた Git 管理ファイルを展開し、[隔離した runner](../lib/README.md#autodreamのcodex隔離) に渡す。`workflow`・`decision`・`environment`・`gotcha` を同条件で扱い、根拠のある修正・削除・重複統合・相対日時の絶対日付化を JSON 計画にする。各更新・削除に観測時の SHA-256 を付け、適用後の形式・引用・鮮度などを機械検査する。
4. **Prune & Index** は検査済みの一括操作を公開 API に渡し、`SUMMARY.md` と4層の索引を再生成して即時マージする。生成文書だけにずれがあれば修復 API を使う。両経路とも、観測した `main` の SHA が変わっていたら拒否して再計画を求める。変更も未修復の問題もなければ `noop` を記録する。

### 手動実行と状態

メインcheckoutから [有限jobのlauncher](../lib/README.md#既存cliと新規jobの接続) 経由で実行する。

```bash
.venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- \
  .venv/bin/python .agents/ops/shared_memory/autodream_job.py run --force --dry-run
```

`--dry-run` はトリガー、シグナル、発火時の計画と公開前検査を JSON で返す。PR とマージは発生せず、統合の基準を進めない。件数や間隔は `--min-main-commits`、`--min-memory-commits`、`--min-days` で変更できる。runner は既定1800秒で停止し、外側の監視にも15秒の回収猶予込みで上限を設ける。手動起動では `--runner-timeout`（秒）を指定できる。

状態と排他 lock は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/shared-memory/` に置く。試験時は `--state-dir` で差し替えられる。`autodream_state.json`（schema version 2）に最終起動、結果、トリガーの計測、直近の確定した PR とマージの SHA、失敗詳細を記録する。成功時の `last_completed_at` は統合または無変更の確認時刻、`last_consolidated_sha` はマージ確定の SHA または無変更を確認した `main` の SHA。

runner・検証・公開の失敗は `failed` と詳細を記録して非ゼロで終了する。runner の詳細は状態ディレクトリの `agent-runs/` に残る。排他競合は標準エラー出力に理由を出す。launcher を経由しない起動は終了コード `75` で拒否する。

### 中断時の復旧

- push 前に `pending_publication` へ観測した base、検査済みの head、専用ブランチ、操作種別を保存する。PR 作成後に番号を追加し、ref 更新前に `advancing` を保存する。確定した公開拒否はインテントを消し、統合の基準を進めず失敗を記録する。
- 応答喪失や中断で結果が不明ならインテントを保持する。再起動は通常の計画に先立ち remote `main` に head が含まれることと PR のマージを照合する。確定すれば専用ブランチの後片付けを試み、統合基準を確定した head に進める。照合できなければ失敗を記録して終了し、計画や再公開を保留する。運用者が PR と専用ブランチを確認してから状態を復旧する。
- 統合確定後に専用ブランチの削除に失敗した場合は `merged` と基準 SHA を保存し、`branch_cleanup_error` へ理由を記録する。pending の `--dry-run` は照合だけを行い、ブランチや統合基準を更新しない。
- 状態ファイルの読込・schema 検査に失敗した場合は原本を同じディレクトリに保全し、`last_outcome: "failed"`、`last_error`、`state_backup` を保存する。保全済みの記録がある間は公開処理を開始しない。運用者が原本と公開先を確認して状態を復旧する。

### systemd への導入

merge 後に人間がメインcheckoutで導入・有効化する。日次 timer は毎日05:00 Asia/Tokyoに点検する。service は起動45分・停止60秒を上限とし、子孫プロセスも停止する。`run` で単発実行でき、`status` の `last_run` から状態 JSON を確認できる。

```bash
spin ops install memory-autodream
spin ops on memory-autodream
spin ops status memory-autodream
spin ops run memory-autodream
```
