# cleanup: branch・worktree の自動掃除（#1060）

ローカルbranchとgit worktreeを **自動削除 / 承認待ち / 保護** に分類し、自動削除対象だけを日次で消す。
承認待ちは `--report-json` に理由つきで出力し、週次レポート（#1057）が掲載する。人間が承認したものだけを
`delete` サブコマンドで消す。複数のClaude/Codexセッションが同じrepoで並行稼働しているため、
**使用中のworktreeを壊さないこと** を最優先にしている。判定できないものは保護側に倒し、理由を出す。

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
$PY $C scan --report-json /tmp/cleanup.json      # dry-run（何も消さない）。--no-disk で du を省略
$PY $C apply-auto --report-json FILE             # auto_delete だけ削除（timer と同じ）
$PY $C delete 'branch:feat/x@0123abcd4567'       # 承認済み候補を削除（tip が変わっていたら拒否）
$PY $C delete 'worktree:/abs/path' --discard-changes   # 未commit変更ごと消す（patch を退避）
```

- `--repo`（既定: cwd のrepoのメインcheckout）、`--config`（既定: `config.toml`）、
  `--state-dir`（既定: `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/cleanup`）。
- ID は `branch:<name>`（`@<sha先頭7文字以上>` で承認時の tip を固定できる。推奨）、
  `worktree:<abs path>`（detached worktree）、または素のbranch名。
- 終了コード: 0 成功 / 1 調査・削除の失敗 / 2 引数エラー / 3 別の削除系実行がロック中。
  gh・git・queue・設定の失敗は黙って続行せず 1 で止まる（レポートも書かない）。

## 分類

1件 = ローカルbranch 1本（checkoutしているworktreeがあればそれと一体）、または detached worktree 1個。
理由に `protect` 系が1つでもあれば **protected**、なければ `approval` 系か未mergeなら
**approval_required**、それ以外（merge済みの証拠あり）が **auto_delete**。

### merge 判定

`gh pr list --state all` の PR（fork 由来は除外）を headRefName で突き合わせる。
`headRefOid` は merge 時点の PR head なので、通常merge・squash merge のどちらでも同じ判定になる。

| 状態 | 結果 |
|---|---|
| tip == merged PR の head、または tip が head の祖先 | `merged_pr`（auto） |
| tip が `default_branch_refs` に含まれる（PR なし） | `ancestor_of_default_branch`（auto。ただし `stale_days` 以上無更新が必要） |
| merged PR の head より後に commit がある | `commits_after_merge`（未push commit。承認待ち） |
| merged PR の head と分岐 | `diverged_from_merged_pr`（承認待ち） |
| merged PR の head がローカルに無い | `undetermined`（保護。`git fetch` 後に再判定） |
| open PR がある | `open_pr`（保護） |
| closed（未merge）PR のみ | `pr_closed_unmerged`（承認待ち） |
| PR なし・default branch にも無い | `no_pr`（承認待ち） / detached なら `detached_head` |

PR による merge は `min_idle_hours`、祖先関係のみは `stale_days` だけ無更新でなければ
`recently_active`（承認待ち）になる。前者は merge 直後に作業継続中のセッション、後者は main から
切ったばかりの空branchを守るため。無更新時間は tip の commit 日時、branch と worktree HEAD の
reflog 最新エントリ、worktree の `HEAD`/`index` と worktree ディレクトリの mtime の最大値。

### 理由コード

| code | kind | 意味 |
|---|---|---|
| `main_worktree` | protect | メインcheckout |
| `protected_branch` | protect | `protected_branches` / `protected_branch_globs`（既定 `main`, `ai-env/base`） |
| `current_checkout` | protect | メインcheckoutでcheckout中のbranch、または実行時cwdを含むworktree |
| `protected_worktree` | protect | `protected_worktree_globs` に一致 |
| `worktree_locked` | protect | `git worktree lock` 済み（理由を表示） |
| `worktree_missing` | protect | worktree ディレクトリが無い（手で `git worktree prune`） |
| `process_using_worktree` | protect | `/proc/*/{cwd,root,fd/*}` が配下を指す、または argv に worktree パスを含むプロセスがいる |
| `queue_job_references` | protect | training queue の `jobs/`（待機中）・`running/`（実行中）の job ファイルが worktree パスを含む |
| `open_pr` | protect | open PR がある |
| `undetermined` | protect | 判定不能（理由を表示） |
| `no_pr` | approval | PR が無く default branch にも含まれない |
| `pr_closed_unmerged` | approval | close された未merge PR のみ |
| `commits_after_merge` | approval | merge 済み PR head 以降の commit（未push） |
| `diverged_from_merged_pr` | approval | merge 済み PR head と分岐 |
| `detached_head` | approval | detached worktree の HEAD が default branch に無い |
| `uncommitted_changes` | approval | tracked ファイルの変更（submodule 含む） |
| `untracked_files` | approval | disposable 以外の untracked |
| `ignored_files` | approval | disposable 以外の gitignore 対象（`outputs/` など。worktree 削除で消えるため） |
| `submodule_populated` | approval | submodule が展開済み（削除に `--force` が要る） |
| `recently_active` | approval | merge 済みだが無更新時間が閾値未満 |
| `merged_pr` | auto | merge 済み PR の head に含まれる |
| `ancestor_of_default_branch` | auto | default branch に含まれる |
| `stale` | info | `stale_days` 以上無更新（分類は変えない） |

symlink（`.venv`, `.training_queue` など）と `disposable_untracked` / `disposable_ignored` に一致する
エントリは作業物とみなさない。worktree 削除は symlink の先を辿らない。

### 安全機構

- `/proc` で読めないプロセス（他ユーザー、non-dumpable）は `warnings` に列挙する（保護判定には使えない）。
  WSL の Windows 側プロセス（Codex Desktop 等）は見えないので、無更新時間の閾値が最後の防壁になる。
- git は `--no-optional-locks` で実行し、他セッションの index を書き換えない。
- `apply-auto` は削除直前に1件ずつ tip・worktree状態・プロセス・queue を取り直して再分類し、
  auto_delete でなくなっていれば `skipped`。branch は `git update-ref -d <ref> <tip>` で消すため、
  分類後に tip が動いていれば失敗する。worktree は auto では `--force` なしの `git worktree remove`
  （git 自身も未commit・untracked を再確認する）。
- `delete` は protected を承認があっても拒否する。未commit・untracked・ignored ファイルがある worktree は
  `--discard-changes` が必要で、tracked の差分は `backups/*.patch` に退避する（untracked は名前のみ記録）。
- 削除系（`apply-auto`, `delete`）は `<state-dir>/cleanup.lock` を flock し、多重起動は終了コード 3。

## ログと復元

削除前に `<state-dir>/deleted.jsonl` へ `phase: "intent"`、完了後に `"done"`（失敗時 `"failed"` と
`error`）を1行ずつ追記する。各行は `ts, id, repo, branch, tip, worktree, category, approved, reasons,
discarded_patch, discarded_untracked, phase`。

```bash
jq -c 'select(.phase=="done")' ~/.local/state/tennis-lab-agents/cleanup/deleted.jsonl
git branch <branch> <tip>                     # branch の復元（tip は gc されるまで残る）
git worktree add <worktree> <branch>          # worktree の復元（detached なら --detach <worktree> <tip>）
git -C <worktree> apply <discarded_patch>     # --discard-changes で退避した差分の復元
```

削除済み branch の commit は unreachable になるため、`gc.reflogExpireUnreachable` / `gc.pruneExpire`
（既定 30日 / 2週間）を過ぎると復元できない。merge 済み PR の head は GitHub 側にも残る。

## レポート JSON（schema_version 1）

週次レポート（#1057）はこの形式を読む。互換性のない変更では `schema_version` を上げる。

```jsonc
{
  "schema_version": 1,
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
    {"worktree": "/abs", "id": "branch:x", "category": "...", "disk_bytes": 0, "disk_error": null}
  ],
  "warnings": ["process scan could not inspect pid ..."],
  "entries": [{
    "id": "branch:<name>" | "worktree:<abs path>",
    "category": "auto_delete" | "approval_required" | "protected",
    "branch": "name" | null, "tip": "<40 hex>", "worktree": "/abs" | null,
    "reasons": [{"code": "<理由コード>", "kind": "protect|approval|auto|info", "detail": "..."}],
    "prs": [{"number": 1, "state": "OPEN|CLOSED|MERGED", "head_oid": "...", "merged_at": "..." | null, "base_ref": "main"}],
    "last_activity": "ISO8601", "activity_source": "tip commit date|branch reflog|...",
    "idle_days": 1.5,
    "worktree_state": null | {"dirty_tracked": [], "untracked": [], "ignored": [],
                              "disposable_untracked": [], "disposable_ignored": [], "submodule_populated": []},
    "disk_bytes": 0 | null, "disk_error": null | "du: ...",
    "action": "none" | "would_delete" | "deleted" | "skipped" | "failed" | "refused",
    "action_error": null | "..."
  }]
}
```

承認依頼には `id` と `tip` を組み合わせた `delete '<id>@<tip先頭12桁>'` を提示すると、承認後に
branch が進んでいた場合に削除を拒否できる。

## systemd（日次）

`systemd/tennis-lab-cleanup.{service,timer}`：毎日 04:30（±15分）にメインcheckoutのコードで
`apply-auto --report-json %S/tennis-lab-agents/cleanup/report-latest.json` を実行する
（`%S` = `~/.local/state`）。`Persistent=true` で停止中の分は起動時に実行。導入・有効化はPR merge後に
人間が `.agents/ops/install_units.sh --enable` で行う。

## テスト

`tests/e2e/agents_ops/test_cleanup.py`：一時repo＋worktree＋fake `gh` で、通常/squash merge 判定、
未push検出、各保護条件（cwd・open fd・argv・queue 実行中/待機中・lock・保護branch/worktree・current
checkout）、worktree の clean 判定、削除ログと復元、承認削除の tip ガード、ロック、失敗時の非ゼロ終了、
レポートスキーマ（理由コードがこのREADMEに載っていること）を確認する。

```bash
.venv/bin/python -m pytest -n 0 -q tests/e2e/agents_ops/test_cleanup.py
```
