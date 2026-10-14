# 追補 issue の消化（`.agents/ops/followup/`）

クロスレビュー（[`../review/`](../review/README.md)）は、merge を止めない minor/nit 指摘を
`[追補] PR #<N> のレビュー軽微指摘` issue として起票する。この job は 2 日に 1 回それらの issue を処理する。
指摘ごとに対応するかを agent に判断させ、issue を close するか、実装して PR を作る。#1086 で導入。
この README が下記規約（対象・ラウンド・agent の出力形式・GitHub への反映）の正本。

| ファイル | 役割 |
|---|---|
| `resolve_followups.py` | 本体（対象の選定、worktree、agent 起動、出力と branch の検証、push・PR・issue 操作） |
| `followup_policy.py` | I/O を持たない判定ロジック（元 PR の状態、ラウンド、出力のパース、処置から動作の導出） |
| `prompt_template.md` | agent へのプロンプト（`string.Template`）。判断の基準はここにある |
| `systemd/` | 2 日に 1 回（奇数日 02:00 JST）の systemd --user service / timer |

テスト: `tests/e2e/agents_ops/test_followup_policy.py`（判定ロジック）、
`tests/e2e/agents_ops/test_followup_resolver.py`（fake gh・fake agent_exec・実 git での end-to-end）。

## 分担

- **スクリプト**: 対象の選定、worktree と branch の作成、push、PR 作成、issue へのコメント・ラベル付け・close。
- **agent**（write モード）: worktree の中で指摘を判断し、実装して commit し、JSON を 1 つ返す。
  push・`gh`・issue 操作はしない。プロンプトで禁じたうえで、環境からも塞ぐ（`GH_TOKEN` 等を除き、
  `GH_CONFIG_DIR` を空ディレクトリに、`GIT_SSH_COMMAND=false` に、`SSH_AUTH_SOCK` を除く）。
  write モードの agent は sandbox の外で動くので、これは多層防御のひとつで、完全な隔離ではない。
- **model**: [`../lib/agents.conf`](../lib/agents.conf) の `followup` プロファイル
  （`FOLLOWUP_CODEX_MODEL` / `_EFFORT` / `_SERVICE_TIER`、現在 `gpt-6.1-sol` / `max` / `default`=非 fast）。
  `agent_exec.sh --agent codex --profile followup --mode write` で起動する。プロファイルの値が欠けていればエラーになる。

## 対象

poll を実行する gh アカウントが起票した open issue のうち、`ai-proposed` ラベルを持ち、本文 1 行目が
`<!-- agent-review-followup:v1 pr=<N> sha=<head> -->` のもの（`../review/` の追補 issue）を issue 番号順に処理する。
1 回の件数に上限は無い。ほかの `ai-proposed` issue（週次レポート由来など）は対象外。

一覧取得後、各 issue の着手時に状態・ラベル・本文・作成者を再取得する。公開直前にも再取得し、
対象外、`needs-human` 付き、本文やラベル・元 PR・信頼できるコメント・直近ラウンドの PR 状態の変更があれば、
push・PR 作成・issue 操作を行わず次回に回す。agent の出力と差分は run dir に残す。
agent を使わない元 PR の却下処理・人間への引き継ぎも、書き込み直前に同じ確認を行う。

issue ごとに、次の順で判定する。

1. `needs-human` ラベルがあれば skip（人間の判断待ち）。
2. 元 PR（マーカーの `pr=`）の状態で分ける。
   - **open**: 保留する。指摘は未 merge の head に対するもので、main にはまだそのコードが無い。元 PR の branch は書き換えない。
   - **merge されずに close**: 指摘は対象のコードとともに無効なので、ラウンドを記録して issue を close する（not planned）。agent は起動しない。
   - **merge 済み**: merge commit が処理対象の base（既定 main）に含まれることを確認して次へ。
     親 branch にだけ merge された PR など、base に未取り込みなら保留する。merge commit が取得できなければエラーにする。
3. 直近のラウンドに PR があれば、その状態で分ける。
   - **open**: レビュー中なので skip。
   - **merge 済み**: 新しいラウンドを始める（issue が open のままなのは、未処理の指摘が残っている場合だけ）。
   - **merge されずに close**: 初回は `needs-human` ラベルを付けてから「人間の判断待ち」コメント（マーカー
     `<!-- agent-followup-rejected:v1 issue=<N> round=<k> pr=<PR> -->`）を記録して skip する。
     人間がコメントで方針を書いてラベルを外すと、次回は新しいラウンドを始める。
4. 新しいラウンドの branch `codex/followup-i<N>-r<k>` が remote に既にあれば、エラーにする。前回の実行が push 後に
   中断した跡なので、下記「公開途中の失敗からの復旧」に従って PR とラウンド記録を揃えるまで止める。

## ラウンド

1 回の処理を **ラウンド** と呼ぶ。ラウンドは issue コメントの 1 行目のマーカー
`<!-- agent-followup-round:v1 issue=<N> round=<k> base=<base SHA> pr=<PR番号|none> -->` で記録する。
数えるのは poll アカウントのコメントだけで、番号が 1 から連番になっていなければエラーになる。
branch 名にラウンド番号を含めるので、前回の PR が merge 済みでも close 済みでも、次のラウンドの branch とは衝突しない。

## agent の作業と出力

1. `git fetch <remote> main` した commit から、`.claude/worktrees/codex-followup-i<N>-r<k>` に新しい branch の worktree を作る
   （同名の worktree か local branch が残っていればエラー。再利用しない）。
2. プロンプトに入れるもの: 指摘（番号付き）、元 PR（本文を含む）、これまでのラウンド、issue のコメント。コメントは
   `author_association` が OWNER / MEMBER / COLLABORATOR のものだけを入れる（repo は public）。
   元 PR 自体が追補 PR（`codex/followup-i*`）なら、指摘の連鎖を止めるため、正しさに関わるものだけを直すよう指示する。
3. agent の最終メッセージは JSON 1 つ（` ```json ` フェンスは可）。

   ```json
   {"summary": "...", "findings": [{"index": 1, "disposition": "fixed|declined|already_resolved|needs_human", "reason": "..."}],
    "pr_title": "... | null", "pr_body": "... | null", "verification": "..."}
   ```

   次のどれかに当たればパース失敗で、GitHub には何も書かず、その issue はエラーになる（他の issue は処理を続ける）。
   - キーに過不足がある。
   - 指摘 1〜n の処置がそれぞれちょうど 1 回になっていない。
   - `fixed` があるのに `pr_title` / `pr_body` が無い、または `fixed` が無いのにある。
   - PR を作る場合、タイトル・本文・要約・理由・確認内容に issue の閉鎖指定がある（`Fixes #...` など）。
4. branch を検証する。次のどれかに当たればエラーになる。
   - 指定の branch に居ない。
   - 未 commit の変更や追跡外のファイルがある。
   - base の子孫になっていない。
   - `fixed` があるのに commit が無い、または `fixed` が無いのに commit がある。
   - 追加された commit メッセージに issue の閉鎖指定がある。閉鎖指定はスクリプトだけが付ける。
5. 検証が終わったら、成否に関わらず worktree と local branch を削除する。

## GitHub への反映

処置から動作が決まる（agent は動作を選ばない）。

| 処置 | 動作 |
|---|---|
| `fixed` あり、`needs_human` なし | push して PR を作る（本文末尾に `Closes #<N>`）。merge で issue が close される |
| `fixed` あり、`needs_human` あり | PR を作る（`Refs #<N>`）。issue に `needs-human` を付ける |
| `fixed` なし、`needs_human` あり | issue に `needs-human` を付ける |
| どちらもなし | issue を close する。全部が `already_resolved` なら completed、それ以外は not planned |

どの場合も、ラウンドのマーカーと指摘ごとの処置の表を issue にコメントする。PR 本文にも同じ表と確認内容を付け、
1 行目に `<!-- agent-followup-pr:v1 issue=<N> -->` を書く。PR は branch prefix `codex/` なので、
Claude がクロスレビューし、通常の merge gate を通る。`needs_human` の処置がある場合は、push・PR 作成・
ラウンド記録より先に `needs-human` を付ける。ラベル付与に失敗しても、人間がラベルを外して再開した状態とは
区別できる。ラベル付与後のコメントに失敗した場合はラベルが残り、人間の判断待ちを維持する。
PR 作成後にコメントが失敗した場合は、
remote に branch と PR が残り、ラウンド記録が欠ける。次回の実行では上記 4 によりエラーとして表に出る。

### 公開途中の失敗からの復旧

remote branch の存在だけでは、対応する PR や人間の判断を確定できないため、自動では記録を補完しない。
PR を作るだけでは復旧は完了せず、同じラウンドを記録する必要がある。

1. 対象 run dir の `prompt.md`（issue番号 N・ラウンド k・当時の base の40桁SHA）、`resolution.json`、
   `changes.diff` と remote branch `codex/followup-i<N>-r<k>` の差分を照合する。
2. `gh pr list --state all --head codex/followup-i<N>-r<k>` で PR の有無を確認する。無ければ検証済みの branch から
   PR を作る。既存 PR を使う場合も head branch・base branch と本文の処置が保存済み結果と一致することを確認する。
   人間待ちがある場合は本文を `Refs #<N>`、残件が無ければ `Closes #<N>` とする。
3. `needs_human` の処置があれば、記録より先に `gh issue edit <N> --add-label needs-human` を実行する。
4. 実行 job と同じ gh アカウントで、issue に k 番の記録がまだ無く、既存記録が 1〜k-1 であることを確認する。
   次の内容をファイルに保存し、`gh issue comment <N> --body-file <ファイル>` で投稿する。1行目の各値は実値に置き換え、
   base は現在の main ではなく手順1のSHA、PR番号は手順2で確認した番号を使う。

   ```text
   <!-- agent-followup-round:v1 issue=<N> round=<k> base=<40桁SHA> pr=<PR番号> -->
   公開途中の失敗から復旧。保存済みの処置・差分とPRの一致を確認しました。
   （指摘ごとの処置、確認内容、PRリンクを記載）
   ```

次回は記録済みの PR 状態に従う（openならレビュー待ち）。人間待ちのラベルは、方針が決まるまで残す。

## 実行

```bash
# dry-run: agent は実行するが、push・PR・issue 操作はしない
.venv/bin/python .agents/ops/followup/resolve_followups.py --dry-run --issue 1080

# 全対象を処理（timer と同じ）
.venv/bin/python .agents/ops/followup/resolve_followups.py
```

- 必要なラベル: `ai-proposed`、`needs-human`。無ければ開始前にエラーになる（終了コード 2）。
- ログ・成果物: `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/followup/runs/<UTC時刻>-i<N>-r<k>/` に
  次を残す。
  - `prompt.md`
  - `agent_output.md`
  - `resolution.json`
  - `changes.diff`（commit があるとき）
  - `commit_messages.txt`（commit があるとき。閉鎖指定の検証対象）
  - agent_exec のログ
- 多重起動防止: state dir の `run.lock` を flock で取る。取れなければ終了コード 75 で終わる（unit はこれを成功扱いにする）。
- 終了コード: 0 = 全 issue 正常、1 = 1 つ以上の issue でエラー、2 = 設定エラー、75 = 実行中。
- タイムアウト: issue ごとに `--agent-timeout`（既定 3 時間）。unit 全体は 36 時間で止める。
- systemd: `../install_units.sh` で link する。timer の有効化は人間が行う
  （`systemctl --user enable --now tennis-lab-followup.timer`）。
  unit はメイン checkout `%h/projects/tennis-lab` を使う。codex が unit 環境で対話シェルと同じ設定
  （認証・接続先）を読むことを、初回に `systemctl --user start tennis-lab-followup.service` で確認する。

## 改訂の仕方

- 判断の基準（何を `declined` にするか等）は `prompt_template.md` を直す。
- 処置と動作の対応、ラウンドの扱いは `followup_policy.py` とこの README を一緒に直し、テストを更新する。
- model を変えるときは `agents.conf` の `FOLLOWUP_CODEX_*` を変える。
