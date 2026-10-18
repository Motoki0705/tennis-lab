# ops-report: 運用レポートと逆提案（#1057）

プロジェクト本体とAI制御（AGENTS.md・`.agents/`・skill・memory・GitHub運用・この自動化自体）を
共同で改善し続けるため、3日おきに「運用レポート」issueを作り、人間がチェックした提案を `ai-proposed` 子issueにする。
設計原則は [../../README.md](../../README.md)、今回の合意事項は親issue #1216 を参照する。

## 構成

| パス | 役割 |
|---|---|
| `ops_report.py` | CLI入口（`report` / `triage` / `schedule init` / `schedule status`） |
| `opsreport/collect.py` | 決定的な収集。gh（open PR/issue、merge済みPR、CI run、過去レポート）、git（期間内commit、行数の多いファイル、TODO系マーカー、worktree/branch数）、掃除モジュール出力、共有memory、`knowledge/`、AI制御ファイル |
| `opsreport/agent.py` | agent選択、プロンプト組み立て、`../lib/agent_exec.sh` による read-only 実行、出力JSONの厳格な検証 |
| `opsreport/proposals.py` | 提案のID付与・描画・本文からのチェック状態の解析・子issue番号の追記 |
| `opsreport/schedule.py` | 初回基準・3日周期の予定計算と、成功/失敗・スキップのマシンローカル状態 |
| `opsreport/report.py` | issue本文（日本語）の組み立て |
| `opsreport/triage.py` | チェック済み提案 → 子issue化（重複防止） |
| `prompt.md` | agentへの依頼文（分析観点・品質基準・出力JSON形式） |
| `systemd/` | `report --if-due`（毎日 08:00 JST に予定判定）と `triage`（毎日 09:00 JST）の service / timer |

分担: 収集はスクリプト（同じrepo状態なら同じ事実）、コードベース読解と提案文の生成はagent（read-only）、
issue作成・編集はスクリプト。agentはJSONだけを返し、IDとMarkdownはスクリプトが付ける。
read-only のagentは git / gh / ネットワークを使えないため、GitHub・git履歴の事実（放置PRの差分規模・mergeable・CI、
全open issue、checkoutと基準refで内容が異なるファイル一覧など）はすべて `collect.py` が集めてプロンプトに埋め込む。
model / effort は [`../lib/agents.conf`](../lib/agents.conf) で固定され、このツールからは変更しない。

## 提案ID・チェック・triage

- レポートのタイトルは `運用レポート YYYY-MM-DD`（実行日ではなく予定回の日付）。提案IDは
  `R-<予定回YYYYMMDD>-<連番>`（例 `R-20261019-03`）。同じISO週にも複数回発行できる。
  重複判定は本文1行目の `<!-- ops-report:v2 slot=YYYYMMDD since=… until=… proposals=… -->` にある
  `slot` が正本。タイトルの変更・close・再試行でも同じslotのissueを再公開しない。
- **旧形式の自動移行・後方互換はない。** `ops-report:v1 week=` 本文はreport/triageとも明示エラー。
  reportはopen/closedの履歴全件を検査し、triageは旧形式のopenレポートが1件でもあれば新形式を含め書き込み前に停止する。
  切替時点の実データは0件。旧形式が発見された場合は、勝手に新IDへ読み替えず、レポートと子issueの対応を人間が確認して
  `ops-report` 対象から外すまで新周期との併用を禁止する。
- 本文のうちtriageが解析するのは `<!-- ops-report:proposals:begin -->` 〜 `end -->` の間だけで、
  そこにあるIDもメタデータのslot・提案数の範囲内でなければエラーになる。agentが書いた文章（概要・提案本文・選択肢など）は
  チェックボックス記法とHTMLコメントを無害化してから埋め込むので、選択可能な項目やマーカーを偽造できない。
- 通常の提案は行頭のチェックボックスで選ぶ。選択肢つきの提案（放置PRの「継続 / rebaseして仕上げ / close」など）は
  **選択肢を1つだけ**チェックする。2つ以上、または見出しだけのチェックはエラーとしてtriageが非ゼロ終了する（他の提案は処理する）。
- `triage` は open な `ops-report` issueをすべて走査し、選ばれた提案ごとに
  タイトル `[<ID>] <提案タイトル>`・ラベル `ai-proposed` の子issueを作り、レポートの該当行末に `→ #番号` を追記する。
- 重複防止: 作成前に全issue（open/closed）のタイトル先頭 `[<ID>]` を走査する。作成後の本文編集が失敗しても、
  再実行時は既存の子issueを再利用して追記だけ行う。再利用するのは子issue本文1行目の
  `<!-- ops-report-proposal: <ID> report=#<レポート番号> option=<選択肢(%エンコード)> -->` が
  ID・レポート・現在選ばれている選択肢のすべてに一致する場合だけで、一致しなければその提案はエラーにする
  （例: 子issue作成後に本文追記が失敗し、その間に選択肢が close → 継続 に変わった場合）。
  `→ #番号` の追記直前にも最新の本文を読み直して同じ照合をし、triage実行中に選択が変わっていれば追記せずエラーにする。
  追記済みの提案も毎回この照合を行うので、後から選択を変えた場合もエラーとして表に出る。同じIDの子issueが2件見つかった場合は停止する。
- レポートをcloseするとtriage対象外になる。不要な提案は放置でよい。
- 同時実行（timerと手動実行など）: `report` の「同slotチェック → issue作成」と `triage` の「子issue走査 → 作成 → 本文追記」は
  マシン共通のロックファイル（flock）で排他する。`report` は agent 実行前にも同slotチェックをし、ロック内で再確認する。
- 行末の `→ #番号` はtriageだけが付ける。PRタイトル等に同じ形があれば `→ \#番号` にエスケープして描画する。
  title走査で見つかった子issueと一致しない `→ #番号` は、子issue本文のマーカーで検証し、不一致ならエラーにする。

## 予定と状態

- `schedule init` を導入完了時に1回だけ実行する。その時刻より**後**に初めて迎える08:00 JSTがanchor。
  08:00ちょうどの初期化は翌日が初回となる。anchorから72時間（3日）ずつの08:00が予定回（slot）となり、月・年・閏日をまたいでも変わらない。
- 状態は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report/schedule.json`。
  未初期化・JSON破損・不正なanchor等はエラーで止め、今日を基準に作り直さない。`init` は既存ファイルを上書きしない。
  破損時はtimerを止め、ログと公開issueを照合して状態を明示修復する。
- timerは毎日08:00 JSTに `report --if-due` を起動し、`Persistent=true` で停止中の起動を回収する。
  初回前は何もしない。現在以前の最新slotが未公開なら1回だけ生成し、逃した古い回は範囲と件数で `skipped` に記録する。
  失敗したslotは同じslotで再試行できる。新しいslotを迎えた場合は古い失敗もスキップし、最新の1回にまとめる。
  成功済みslotはagentを起動しない。開始の厳密な精度は保証せず、timerには最大10分の遅延を許容する。
- `schedule status` はanchor、`next_slot`（次の予定時刻）、`pending_slot`（未成功の最新回）、
  `last_attempt`（running / failed / succeededとエラー）、`last_success`、`skipped` をJSONで表示する。
  実行中でも確認できる。中断して `running` が残ったら再実行し、GitHubに公開済みなら成功状態を回復して二重公開を防ぐ。
- `schedule.lock` でreportの収集から公開までを直列化する。初期化も同じロックを使い、triageとissue公開は別の共通ロックで排他する。
  状態は一時ファイルからのatomic replaceで保存する。ログ先・checkoutを変えても予定状態はマシン共通。
- 手動の `report` も最新slotを使う。初回前と成功済みslotはエラー（`--dry-run` は成功済みslotのプレビューが可能）。
  任意日付の追加公開は提供しない。`--now` はtimezone付きISO日時を使うテスト専用の時計指定。

## 手動実行

```bash
cd ~/projects/tennis-lab
# 導入後の予定・前回結果を確認
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
# 本文を標準出力へ（issueは作らない）。agentは実行される。
.venv/bin/python .agents/ops/ops-report/ops_report.py report --dry-run --output /tmp/report.md
# 本番（ops-report ラベルのissueを作成）
.venv/bin/python .agents/ops/ops-report/ops_report.py report --agent codex
# チェック済み提案の子issue化（--dry-run で作成予定だけ表示）
.venv/bin/python .agents/ops/ops-report/ops_report.py triage --dry-run
```

実行ごとに `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report/<時刻>-<command>[-dry]-<pid>/` へ
`run.log`、`collected.json`、`prompt.md`、`agent_output.md`、agent CLIのログ、`report.md`、`result.json` を残す。
`--log-dir` / `OPS_REPORT_LOG_DIR` の相対パスは、起動時のカレントディレクトリを基準に解決する。
gh・git・agent・掃除モジュールの失敗、agent出力の形式違反、設定値の不正はすべて非ゼロ終了になる（黙ってskipしない）。
agentと掃除モジュールは専用のプロセスグループで起動し、タイムアウト・失敗・終了時にグループ全体を
SIGTERM → SIGKILL で止めて回収する（claude/codex が起動した子プロセスを残さない）。

集計期間は `[前回レポートの until, 今回の実行時刻)`（実行時刻は秒単位に切り捨て、メタデータと同じ精度にする）。境界は issue 作成時刻ではなく本文メタデータの `until`
（収集の基準時刻）で引き継ぐので、収集〜公開の間のイベントも欠落・重複しない。初回の開始境界はanchorの3日前で、初回が遅延してもその境界を維持する。merge済みPRは
`--search merged:>=<日付>` でmerge期間を検索条件に含めて取得し、件数上限に達したらエラーにする。

## 設定

CLI引数が優先。systemd実行時は `~/.config/tennis-lab-agents/ops-report.env`（任意）に書く。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `OPS_REPORT_AGENT` | `codex` | `claude` / `codex` / `alternate`（anchorを0番とする予定回が偶数→claude、奇数→codex）。既定が `codex` なのは Claude の使用枠を自動運用で消費しないため（#1178） |
| `OPS_REPORT_BASE_REF` | `origin/main` | 期間内commit・行数・TODO集計の対象ref。remote追跡refなら実行開始時に `git fetch <remote> <branch>` し（失敗はエラー）、そのcommit SHAを実行中固定して本文メタデータ `base=` に記録する。ローカルref/SHAはfetchせず、本文に「ローカルref、fetchなし」と明記する |
| `OPS_REPORT_STALE_PR_DAYS` | `7` | 何日更新がなければ放置PRとして載せるか |
| `OPS_REPORT_AGENT_TIMEOUT` | `5400` | agent実行のタイムアウト（秒） |
| `OPS_REPORT_CLEANUP_CMD` | `<python> .agents/ops/cleanup/cleanup.py scan` | 掃除モジュールのコマンド。`--report-json <一時ファイル>` を自動付与し、そのファイルのJSONを読む |
| `OPS_REPORT_LOG_DIR` | 上記state dir | ログの保存先 |
| `OPS_REPORT_AGENT_EXEC` | `../lib/agent_exec.sh` | agentラッパー（テストで差し替える） |
| `OPS_REPORT_LOCK_FILE` | `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report.lock` | issue書き込みの排他ロック（下記） |
| `OPS_REPORT_LOCK_TIMEOUT` | `600` | ロック待ちの上限（秒）。超えたら非ゼロ終了 |

- 放置PRへの推奨: agentは対象の放置PRすべてについて `{"recommendation": "継続|rebaseして仕上げ|close", "reason": …}` を返す必要があり、欠け・余分・形式違反は出力形式エラーになる。
- 掃除モジュール: `OPS_REPORT_CLEANUP_CMD` が未設定で `.agents/ops/cleanup/` も無ければ、本文に「掃除モジュール未導入」と書く。
  ディレクトリがあれば `cleanup.py scan --report-json <一時ファイル>` を実行してJSONをそのままagentへ渡す（`cleanup.py` が無ければエラー）。
  共有memory（`.agents/memory/`）も同様に、無ければ「未導入」と書く。あれば `SUMMARY.md` と、層の `index.md` を除く各エントリの先頭を渡す。
- agentは `--repo-root`（既定はこのcheckout）を読む。base refとcheckoutで内容が異なるファイル
  （無視対象を除く未追跡ファイルを含む）があれば、
  一覧をagentへ渡し、本文の冒頭に注意を出す。

## systemd --user への導入（merge後に人間が行う）

旧unitが動いたまま新unitを導入しない。以下は親が実機で行う。installerは旧unit本体またはtimerのwantsリンクが
残っていれば、リンク作成・daemon-reload・enableより前にエラーにする（壊れたsymlinkや他のwants内も検出）。
旧source unitも再導入しない。旧PR自動レビューunitが残る場合は [reviewの撤去手順](../review/README.md) も先に実施する。

```bash
# 旧reportと、切替中に古いCLIを呼び得るtriageを止める
systemctl --user disable --now tennis-lab-weekly-report.timer tennis-lab-ops-triage.timer
systemctl --user stop tennis-lab-weekly-report.service tennis-lab-ops-triage.service
rm -f ~/.config/systemd/user/tennis-lab-weekly-report.{service,timer}
rm -f ~/.config/systemd/user/timers.target.wants/tennis-lab-weekly-report.timer
# 任意の旧env fileがある場合だけ移す。旧変数はCLIがエラーにする。
if [ -f ~/.config/tennis-lab-agents/weekly-report.env ]; then
  mv ~/.config/tennis-lab-agents/weekly-report.env ~/.config/tennis-lab-agents/ops-report.env
  sed -i 's/^WEEKLY_REPORT_/OPS_REPORT_/' ~/.config/tennis-lab-agents/ops-report.env
fi
cd ~/projects/tennis-lab
.agents/ops/install_units.sh --dry-run
.agents/ops/install_units.sh                    # link + daemon-reload
systemd-analyze --user verify .agents/ops/ops-report/systemd/*.{service,timer}
# 導入完了後に基準を記録してからtimerを有効化（既存stateがあればinitは繰り返さない）
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule init
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
systemctl --user enable --now tennis-lab-ops-report.timer tennis-lab-ops-triage.timer
# unit実効値、日次の次回check、3日周期の次回slotを照合
systemctl --user cat tennis-lab-ops-report.service tennis-lab-ops-report.timer tennis-lab-ops-triage.service
systemctl --user show tennis-lab-ops-report.timer -p NextElapseUSecRealtime -p LastTriggerUSec
systemctl --user list-timers --all 'tennis-lab-*'
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
journalctl --user -u tennis-lab-ops-report.service
```

旧timer/serviceがinactive・disabledでunitリンクも無いこと、新timerだけがreportを起動することを確認する。
`list-timers` の次回は日次の予定判定時刻であり、実際にreportを生成する予定回は `schedule status` で確認する。
旧state/logは新stateに読み込まない。必要なログを保管後、旧 `weekly-report/` と `weekly-report.lock` を親が撤去する。
既存 `ops-report` ラベルの説明は `運用レポート（.agents/ops/ops-report）` に親が更新する（既存ラベルはCLIで上書きしない）。

unitは `%h/projects/tennis-lab`（メインcheckout）と `.venv/bin/python` を使う。`claude` / `codex` / `gh` は
`~/.local/bin` か `/usr/bin` にある前提で `PATH` を設定している。

## 改訂の仕方（固めすぎない）

- 分析観点・品質基準・出力形式は `prompt.md` にある。観点の追加や言い回しの変更はここだけ直せばよい。
  レポート自身が「制御ルールの改訂」カテゴリでこのプロンプトや本ツールの改善を提案してよい（プロンプトにそう書いてある）。
- カテゴリを増やす場合は `opsreport/proposals.py` の `CATEGORIES` を変える（プロンプトのカテゴリ一覧は自動で反映される）。
- 収集項目を増やす場合は `opsreport/collect.py` に決定的な関数を足す。観察データ欄に出したいものは `opsreport/report.py`。
- 挙動を変えたら `tests/e2e/agents_ops/test_ops_report*.py`（fake gh / fake agent_exec）を更新する。
