# クロスレビューと merge gate（`.agents/ops/review/`）

open PR を定期的に poll し、PR を作成した agent と**逆の** agent でレビューして結果を PR コメントに残し、
条件を満たす PR だけを merge する（自動 merge は既定で無効）。#1058 で導入。
この README が下記規約（作成 agent の判別・レビュー・merge gate）の正本。

| ファイル | 役割 |
|---|---|
| `cross_review.py` | poll 本体（gh / git / agent_exec の呼び出し、コメント・ラベル・merge） |
| `review_policy.py` | I/O を持たない判定ロジック（判別・パス照合・出力パース・gate） |
| `prompt_template.md` | レビュー agent へのプロンプト（`string.Template`） |
| `shared_paths.txt` | 共有基盤パス一覧 |
| `required_checks.txt` | merge に必須の CI check 名 |
| `systemd/` | 10 分おきの systemd --user service / timer |

テスト: `tests/e2e/agents_ops/test_review_policy.py`（判定ロジック）、
`tests/e2e/agents_ops/test_review_poll.py`（fake gh・fake agent_exec・実 git での end-to-end）。

## 作成 agent の判別規約

PR を作る agent は branch 名に自分の prefix を付ける。

| 作成 agent | branch prefix（主） | ラベル（補助） | レビュー agent |
|---|---|---|---|
| Claude | `claude/` | `agent:claude` | Codex |
| Codex | `codex/` | `agent:codex` | Claude |

- branch prefix が主。prefix を付けられなかった PR は `agent:<name>` ラベルで作成 agent を示せる。
- prefix とラベルが食い違う、または両方のラベルが付いている PR はエラー（poll は非ゼロ終了し、その PR には何もしない）。
  黙ってどちらかに寄せない。
- どちらの手がかりも無い PR（人間が作った PR、`fix/` `refactor/` 等の旧来 branch）は**レビュー対象外**。
  ログに `skip (作成agentを判別できない ...)` と記録するだけで、コメントもラベルも付けない。
  自動レビューを受けたいときは `agent:<name>` ラベルを付ける。
- draft PR と fork からの PR も対象外（ready for review になった時点で対象になる）。

## レビュー

1. base ブランチと `refs/pull/<N>/head` を fetch し、PR の差分（`base...head`、rename は削除と追加に分解）、
   base 版の `AGENTS.md`、タイトル・本文、CI 状態をプロンプトに埋め込む。diff が 20 万文字を超える場合は
   先頭だけを埋め込み、切ったことをプロンプトとレビューコメントに明記する。
2. PR head を detached で checkout したレビュー用 worktree（`.claude/worktrees/review-pr<N>-<sha12>`）を作り、
   `../lib/agent_exec.sh --mode read-only` で逆の agent を起動する。worktree はレビュー後に必ず削除する。
3. agent の最終メッセージは JSON 1 つ（`prompt_template.md` のスキーマ。` ```json ` フェンスは可）。
   `verdict` は `approve` / `request_changes` / `comment`。キーの過不足、未知の値、
   `request_changes` なのに blocker/major の指摘が無い、`approve` なのに blocker がある、はすべてパース失敗で、
   その PR には何も投稿せず poll は非ゼロ終了する（次回の poll で再レビューされる）。
4. 結果を PR コメントとして投稿する。コメント先頭の不可視マーカー
   `<!-- agent-cross-review:v1 sha=<head> reviewer=<agent> verdict=<verdict> -->` が状態の正本。

**二重レビューの防止**: poll を実行している gh アカウントが投稿したコメントに、現在の head SHA と
期待するレビュー agent のマーカーがあれば再レビューしない。他アカウントが書いたマーカーは無視する。
新しい commit が push されると head SHA が変わるので再レビューされる。

## 共有基盤パス

`shared_paths.txt` に 1 行 1 パターンで書く（`#` 以降はコメント）。パターンは repo 相対の POSIX glob で、
`*` は `/` を跨がず、`**` は跨ぐ。パターンは**そのパス自身とその配下**に一致する（`src/utils` は
`src/utils/io.py` にも一致）。変更ファイル（rename の旧パスを含む）が 1 つでも一致すると共有基盤に触れたとみなす。

一覧は運用で自由に増減してよい（固めすぎない。週次レポートで見直す）。テストは「代表的な共有基盤パスが一致し、
タスク内のパスが一致しないこと」と「各パターンが repo 内の実在パスを指すこと」だけを確認する。

## merge gate

レビュー後（または既存レビューがあれば毎 poll）、最新 head について次を判定する。

| 判定 | 条件 | 動作 |
|---|---|---|
| `merge` | 下の阻害要因が無く、共有基盤にも触れない | `AGENT_AUTO_MERGE=1` なら `gh pr merge --merge --match-head-commit <head>`（repo 慣例の merge commit）。無効時は「merge してよい状態」とコメントするだけ |
| `needs_human` | 阻害要因は無いが共有基盤に触れる | `needs-human` ラベル＋該当ファイルを列挙したコメント。merge しない |
| `blocked` | 阻害要因が 1 つ以上ある | merge しない |

阻害要因（理由コード）:

- `draft` / `fork`
- `conflict`（mergeable=CONFLICTING）、`mergeable_unknown`（GitHub が計算中）
- `ci_failed` / `ci_pending`: `statusCheckRollup` を check ごと（workflow 名＋check 名）に最新 run へ畳み、
  1 つでも失敗・未完了があれば該当。成功扱いは SUCCESS / NEUTRAL / SKIPPED のみ。
- `ci_missing_required`: `required_checks.txt` の check が rollup に無い（CI 起動前に labeler 等だけが
  成功して merge される競合を防ぐ。main に branch protection の required checks は無いのでこのファイルで持つ）。
- `review_missing` / `review_request_changes` / `review_not_approved`（`comment`）: 最新 head への
  逆 agent の approve が無い。

gate コメントにもマーカー `<!-- agent-merge-gate:v1 sha=... decision=... reasons=... -->` を付け、
同じ head・判定・理由の組では再投稿しない。`blocked` は `ci_failed` か `conflict` を含むときだけコメントする
（他の理由は次の poll で解消するか、レビューコメント自体が理由を伝えている）。

### 自動 merge フラグ

環境変数 `AGENT_AUTO_MERGE`: 未設定または `0` で無効（既定。判定をコメントするだけ）、`1` で有効。
それ以外の値はエラー。systemd unit は `AGENT_AUTO_MERGE=0` を明示しているので、有効化は unit を書き換えて行う。

## request_changes の後（v1）

v1 では `agent-review:changes-requested` ラベルとレビューコメントを付けるところまで。作成 agent や人間が
修正を push すると新しい head を再レビューし、`approve` / `comment` ならラベルを外す。

将来の拡張点: ラベルの付いた PR を作成 agent に差し戻す（`agent_exec.sh --mode write` で PR branch の
worktree を作り、レビューコメントを入力に修正 commit を push させる）。差し戻し回数の上限と、
上限到達時の `needs-human` 化を同時に決めること。

## 実行

```bash
# 1 本を dry-run（コメント・ラベル・merge を一切しない。レビューと gate 判定を stdout に出す）
.venv/bin/python .agents/ops/review/cross_review.py --pr 1055 --dry-run

# 全 open PR を処理（timer と同じ）
.venv/bin/python .agents/ops/review/cross_review.py
```

- 必要なラベル: `needs-human`、`agent-review:changes-requested`（判別用に `agent:claude`、`agent:codex` も）。
  書き込みを伴う実行は前 2 つが repo に無いと開始前にエラーになる。
  `gh label create agent-review:changes-requested --color FBCA04 --description "クロスレビューが修正を要求"` 等で作る。
- ログ・成果物: `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/review/runs/<UTC時刻>-pr<N>-<sha12>/`
  に `prompt.md`、`agent_output.md`（生出力）、`review.json`、`review_comment.md`、agent_exec のログを残す。
- 多重起動防止: state dir の `poll.lock` を flock で取り、取れなければ終了コード 75（unit は成功扱い）。
- 終了コード: 0 = 全 PR 正常、1 = 1 つ以上の PR でエラー（他の PR は処理を続ける）、2 = 設定エラー、75 = 実行中。
- systemd: `../install_units.sh` で `systemd/` の unit を link する。timer の有効化（`--enable`）は人間が行う。
  unit はメイン checkout `%h/projects/tennis-lab` の `.venv` とスクリプトを使い、`--repo` も同 checkout を指す。
