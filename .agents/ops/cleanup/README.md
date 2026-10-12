# cleanup: branch・worktree の掃除（#1060）

ローカルbranchとgit worktreeを **自動削除 / 承認待ち / 保護** に分類する。日次 timer（`apply-auto`）が
消すのは **worktree を持たない merge 済みのローカルbranch の ref だけ** で、worktree は決して消さない。
worktree は人間が承認した `delete worktree:<path>` でのみ、しかも **完全に clean なときだけ** 消す。
承認待ちは `--report-json` に理由つきで出力し、週次レポート（#1057）が掲載する。

## なぜこの範囲か

当初は worktree も自動削除し、未commit変更は `--discard-changes` で patch に退避する設計だった。
しかし PR #1065 のクロスレビューで、使用中の worktree を見落とす経路が毎回新たに見つかった
（rebase 中の branch、`printf %q` された queue job、相対パスの argv、`bash -c` 内の symlink、
シンボリック参照、削除途中の内容変更など）。複数の Claude/Codex セッションが同じ repo で並行稼働する
環境では「使用中か」を外から完全に判定する手段がなく、worktree の自動削除の攻撃面は際限がない。
そこでユーザー判断により、構造的に安全な範囲へ絞った。

- 自動で消すのは branch ref だけ。worktree を持たない branch には作業ファイルが無く、消しても
  `deleted.jsonl` の tip から `git branch <name> <tip>` で完全に戻せる。
- worktree の削除は承認制で、tool は作業物を **一切破棄しない**。`git status --ignored` に
  `.venv` symlink 以外が1つでも出る worktree は消さず、理由を出すだけにする（掃除は人間が行う）。
  使用中判定（プロセス・queue・lock など）は追加の防壁として残す。

| ファイル | 役割 |
|---|---|
| `cleanup.py` | CLI（`scan` / `apply-auto` / `delete`）、削除・ログ・ロック・レポート |
| `inventory.py` | 事実収集（git, `gh pr list`, `/proc`, training queue, `du`） |
| `classify.py` | 事実→分類の純粋関数。理由コードの一覧（`REASON_KINDS`）の正本 |
| `config.toml` | 方針パラメータ（全キー必須。未知キーはエラー） |
| `systemd/` | 日次 timer（`apply-auto`）。`.agents/ops/install_units.sh` で導入 |

## 使い方

```bash
PY=.venv/bin/python; C=.agents/ops/cleanup/cleanup.py
$PY $C scan --report-json /tmp/cleanup.json        # dry-run（何も消さない）。--no-disk で du を省略
$PY $C apply-auto --report-json FILE               # auto_delete の branch だけ削除（timer と同じ）
$PY $C delete 'branch:feat/x@0123abcd4567'         # 承認済み branch を削除（tip が変わっていたら拒否）
$PY $C delete 'worktree:/abs/path@0123abcd4567'    # 承認済み worktree を削除（clean なときだけ）
```

- `--repo`（既定: cwd のrepoのメインcheckout）、`--config`（既定: `config.toml`）、
  `--state-dir`（既定: `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/cleanup`）。
- ID は `branch:<name>`、`worktree:<abs path>`、または素のbranch名。末尾の `@<sha先頭7文字以上>` で
  承認時の tip（worktree なら HEAD）を固定できる（推奨）。
- worktree を消しても branch は残る。merge 済みなら次の `apply-auto` で branch が消える（2段階）。
- 終了コード: 0 成功 / 1 調査・削除の失敗または拒否 / 2 引数エラー / 3 別の削除系実行がロック中。
  gh・git・queue・設定の失敗は黙って続行せず 1 で止まる（レポートも書かない）。

## 分類

エントリは2種類（レポートの `kind`）。理由に `protect` 系が1つでもあれば **protected**。

- **branch**（`branch:<name>`、全ローカルbranch）: `block`・`approval` 系が無く merge 済みの証拠があれば
  **auto_delete**、それ以外は **approval_required**。worktree が checkout していれば
  `checked_out_in_worktree`（block）で、先に worktree を消すまで tool は消さない。
- **worktree**（`worktree:<path>`、メインcheckout以外の全 worktree）: 決して auto_delete にならず、
  保護されなければ **approval_required**（`worktree_needs_approval`）。`block` 系の理由があるうちは
  承認しても `delete` が拒否する。

`delete` は protected と block のある entry を拒否する。

### merge 判定（branch）

`gh pr list --state all` の PR（fork 由来は除外）を headRefName で突き合わせる。
`headRefOid` は merge 時点の PR head なので、通常merge・squash merge のどちらでも同じ判定になる。

| 状態 | 結果 |
|---|---|
| tip == merged PR の head | `merged_pr`（auto） |
| tip が `default_branch_refs` に含まれる（PR なし） | `ancestor_of_default_branch`（auto。ただし `stale_days` 以上無更新が必要） |
| tip が merged PR の head の祖先 | `behind_merged_pr`（承認待ち） |
| merged PR の head より後に commit がある | `commits_after_merge`（未push commit。承認待ち） |
| merged PR の head と分岐 | `diverged_from_merged_pr`（承認待ち） |
| merged PR の head がローカルに無い | `undetermined`（保護。`git fetch` 後に再判定） |
| open PR がある | `open_pr`（保護） |
| closed（未merge）PR のみ | `pr_closed_unmerged`（承認待ち） |
| PR なし・default branch にも無い | `no_pr`（承認待ち） |

PR による merge は `min_idle_hours`、祖先関係のみは `stale_days` だけ無更新でなければ
`recently_active`（承認待ち）になる。前者は merge 直後に作業継続中のセッション、後者は main から
切ったばかりの空branchを守るため。無更新時間は tip の commit 日時、branch と worktree HEAD の
reflog 最新エントリ、worktree の管理ディレクトリ（`.git/worktrees/<id>`）の `HEAD`/`index` と
worktree ディレクトリの mtime の最大値。どれも取れなければ `undetermined`（保護）で `last_activity` は null。

### worktree の削除条件

`delete worktree:<path>` が消すのは、次をすべて満たす worktree だけ。満たさなければ理由つきで拒否する。

- `git status --porcelain --ignored --ignore-submodules=none` に何も出ない。例外は最上位の `.venv`
  が symlink の場合だけ（削除時に symlink だけを外し、参照先は触らない）。`__pycache__` や `outputs/`、
  他の symlink も作業物として扱う。submodule は展開されていれば `submodule_populated` で拒否する
  （展開済みなら拒否なので submodule の中は調べない）。
- `assume-unchanged` / `skip-worktree` の付いた tracked ファイルが無い（`git ls-files -v` の小文字タグ・`S`）。
  これらの変更は `git status` に出ないため、付いていれば `hidden_index_flags` で拒否する。
- HEAD が push 済み: `refs/remotes/*` のどれかに含まれる、または branch の PR の head と一致・その祖先。
  ローカルの main に含まれるだけでは push 済みとみなさない。
- 使用中でない（下表の protect 系が無い）。
- 削除は `--force` なしの `git worktree remove`（git 自身も未commit・untracked を再確認する）。

### 理由コード

| code | kind | 対象 | 意味 |
|---|---|---|---|
| `protected_branch` | protect | branch | `protected_branches` / `protected_branch_globs`（既定 `main`, `ai-env/base`） |
| `current_checkout` | protect | 両方 | メインcheckoutでcheckout中のbranch、または実行時cwdを含むworktree |
| `symbolic_ref` | protect | branch | シンボリック参照（`refs/heads/alias` → `refs/heads/main` など）、またはその参照先 |
| `branch_in_use` | protect | branch | どこかの worktree の rebase（`rebase-{merge,apply}/head-name`、`rebase-merge/update-refs`）・bisect（`BISECT_START`）が保持中。HEAD が detached なので `git worktree list` には出ない（git の `branch.c` と同じ判定） |
| `open_pr` | protect | branch | open PR がある |
| `protected_worktree` | protect | worktree | `protected_worktree_globs` に一致 |
| `worktree_locked` | protect | worktree | `git worktree lock` 済み（理由を表示） |
| `worktree_missing` | protect | worktree | worktree ディレクトリが無い（手で `git worktree prune`） |
| `operation_in_progress` | protect | worktree | rebase / am / bisect / merge / cherry-pick / revert / sequencer が進行中 |
| `process_using_worktree` | protect | worktree | `/proc/*/{cwd,root,fd/*}` が配下を指す、または argv が worktree を指すプロセスがいる（絶対パス、プロセスの cwd 基準で配下に解決される相対パス、worktree のディレクトリ名をパス要素として含むもの） |
| `queue_job_references` | protect | worktree | training queue の `jobs/`・`running/` の job が worktree を参照する。job は `printf %q` 形式をシェルの単語に復号し、相対パスは job 内の全 `cd`/`pushd` 先を基準に解決、絶対パスは symlink も解決して照合する。ディレクトリ名をパス要素として含む job も該当。復号できない job、最初の絶対 `cd` より前に相対の単語がある job、展開が必要な `cd` 先を持つ job があれば実行全体を 1 で止める |
| `queue_job_unparseable` | protect | worktree | 入れ子のシェルコード（`-c`/`-lc`、`eval`、展開を含む `source`/`.`、`$(...)`、バッククォート）を含む job、またはメインcheckout・全 worktree の外へ `cd` する job がある間は、すべての worktree を保護する（`warnings` にも列挙） |
| `undetermined` | protect | 両方 | 判定不能（理由を表示） |
| `checked_out_in_worktree` | block | branch | worktree が checkout している。先に worktree を消す |
| `uncommitted_changes` | block | worktree | tracked ファイルの変更（submodule 含む） |
| `untracked_files` | block | worktree | untracked（`.venv` symlink 以外） |
| `ignored_files` | block | worktree | gitignore 対象（`outputs/`、`__pycache__` など。`.venv` symlink 以外） |
| `submodule_populated` | block | worktree | submodule が展開済み |
| `hidden_index_flags` | block | worktree | `assume-unchanged` / `skip-worktree` の tracked ファイルがある（変更が `git status` に出ない） |
| `unpushed_commits` | block | worktree | HEAD がどの `refs/remotes/*` にも PR head にも含まれない |
| `worktree_needs_approval` | approval | worktree | worktree は承認した `delete` でのみ消す（push 済みの根拠を表示） |
| `no_pr` | approval | branch | PR が無く default branch にも含まれない |
| `pr_closed_unmerged` | approval | branch | close された未merge PR のみ |
| `behind_merged_pr` | approval | branch | tip が merged PR の head の祖先（一致ではない） |
| `commits_after_merge` | approval | branch | merge 済み PR head 以降の commit（未push） |
| `diverged_from_merged_pr` | approval | branch | merge 済み PR head と分岐 |
| `recently_active` | approval | branch | merge 済みだが無更新時間が閾値未満 |
| `merged_pr` | auto | branch | tip == merge 済み PR の head |
| `ancestor_of_default_branch` | auto | branch | default branch に含まれる |
| `stale` | info | 両方 | `stale_days` 以上無更新（分類は変えない） |

### 安全機構

- `/proc` で読めないプロセス（他ユーザー、non-dumpable）は `warnings` に列挙する。
- git は `--no-optional-locks` で実行し、他セッションの index を書き換えない。
- `apply-auto` と `delete` は削除直前に1件ずつ事実を取り直して再分類する。`apply-auto` は
  auto_delete でなくなっていれば `skipped`。`delete` は実行冒頭の収集から tip・分類・理由コード
  （info 以外）・worktree の状態のどれかが変われば `refused`。
- branch は削除直前にシンボリック参照でないことと tip を確認し、`git update-ref --no-deref -d <ref> <tip>`
  で消す（原子的に tip を再確認し、参照先は決して消さない）。worktree は HEAD を確認してから消す。
- 削除系（`apply-auto`, `delete`）は `<state-dir>/cleanup.lock` を flock し、多重起動は終了コード 3。

## ログと復元

削除前に `<state-dir>/deleted.jsonl` へ `phase: "intent"`、完了後に `"done"`（失敗時 `"failed"` と
`error`）を1行ずつ追記する。各行は `ts, id, repo, branch, tip, worktree, category, approved, reasons,
branch_config, phase`（worktree の削除では `tip` は HEAD、`branch` は checkout していた branch）。
`branch_config` は branch 削除時に一緒に消す `branch.<name>.*` 設定（upstream・description など）の
`[key, value]` の列で、worktree の削除では空。

```bash
jq -c 'select(.phase=="done")' ~/.local/state/tennis-lab-agents/cleanup/deleted.jsonl
git branch <branch> <tip>                # branch の復元（tip は gc されるまで残る）
git config --add <key> <value>           # branch_config の各組を順に戻す（upstream・description など）
git worktree add <worktree> <branch>     # worktree の復元（detached なら --detach <worktree> <tip>）
```

削除済み branch の commit は unreachable になるため、`gc.reflogExpireUnreachable` / `gc.pruneExpire`
（既定 30日 / 2週間）を過ぎると復元できない。merge 済み PR の head は GitHub 側にも残る。
worktree は clean なものしか消さないので、復元は `git worktree add` と `.venv` symlink の再作成で足りる。

## レポート JSON（schema_version 2）

週次レポート（#1057）はこの形式を読む。互換性のない変更では `schema_version` を上げる
（v2: worktree を独立 entry にし `kind` を追加、reason kind に `block` を追加、
`worktree_state` の disposable を1つに統合）。

```jsonc
{
  "schema_version": 2,
  "generated_at": "2026-10-10T12:00:00+00:00",  // UTC ISO8601
  "host": "...", "repo": "/abs/main/checkout",
  "mode": "dry-run" | "apply-auto" | "delete",
  "config": { /* config.toml の有効値 */ },
  "summary": {
    "branches": 489, "worktrees": 71,
    "categories": {"auto_delete": 0, "approval_required": 0, "protected": 0},
    "reason_counts": {"<category>": {"<code>": 0}},      // 1件内の重複は1回
    "disk_bytes_by_category": {"<category>": 0},
    "disk_bytes_total": 0,
    "actions": {"deleted": 0, "would_delete": 0, "failed": 0, "skipped": 0, "refused": 0}  // 0件のキーは省略
  },
  "worktrees_by_size": [  // 大きい順。メインcheckoutは計測しない
    {"worktree": "/abs", "id": "worktree:/abs", "category": "...", "disk_bytes": 0, "disk_error": null}
  ],
  "warnings": ["process scan could not inspect pid ..."],
  "entries": [{
    "id": "branch:<name>" | "worktree:<abs path>",
    "kind": "branch" | "worktree",
    "category": "auto_delete" | "approval_required" | "protected",  // worktree は auto_delete にならない
    "branch": "name" | null, "tip": "<40 hex>",   // worktree では checkout 中の branch と HEAD
    "worktree": "/abs" | null,                    // branch entry では常に null
    "reasons": [{"code": "<理由コード>", "kind": "protect|block|approval|auto|info", "detail": "..."}],
    "prs": [{"number": 1, "state": "OPEN|CLOSED|MERGED", "head_oid": "...", "merged_at": "..." | null, "base_ref": "main"}],
    "last_activity": "ISO8601" | null, "activity_source": "tip commit date|branch reflog|...|none",
    "idle_days": 1.5,
    "worktree_state": null | {"dirty_tracked": [], "untracked": [], "ignored": [],
                              "disposable": [".venv"], "submodule_populated": [],
                              "hidden_index_flags": []},
    "disk_bytes": 0 | null, "disk_error": null | "du: ...",
    "action": "none" | "would_delete" | "deleted" | "skipped" | "failed" | "refused",
    "action_error": null | "..."
  }]
}
```

承認依頼には `delete '<id>@<tip先頭12桁>'` を提示すると、承認後に tip / HEAD が動いていた場合に
削除を拒否できる。

## systemd（日次）

`systemd/tennis-lab-cleanup.{service,timer}`：毎日 04:30（±15分）にメインcheckoutのコードで
`apply-auto --report-json %S/tennis-lab-agents/cleanup/report-latest.json` を実行する
（`%S` = `~/.local/state`）。`Persistent=true` で停止中の分は起動時に実行。導入・有効化はPR merge後に
人間が `.agents/ops/install_units.sh --enable` で行う。

## テスト

`tests/e2e/agents_ops/test_cleanup.py`：一時repo＋worktree＋fake `gh` で、通常/squash merge 判定と
tip 一致の要件、`apply-auto` が worktree を一切消さず worktree の無い merge 済み branch だけを消すこと、
worktree の削除条件（dirty・untracked・ignored・`__pycache__`・`.venv` 以外の symlink・submodule・
assume-unchanged / skip-worktree・未push の各拒否と、clean な worktree の削除で branch が残ること）、各保護条件（cwd・open fd・argv・
相対 argv・queue・lock・保護branch/worktree・current checkout・rebase/bisect 中の branch・シンボリック参照・
解決しきれない queue job）、削除途中の変化による拒否、削除ログと復元（branch 設定を含む）、tip ガード、ロック、
失敗時の非ゼロ終了、レポートスキーマ（理由コードがこのREADMEに載っていること）を確認する。

```bash
.venv/bin/python -m pytest -n 0 -q tests/e2e/agents_ops/test_cleanup.py
```
