# weekly-report: 週次運用レポートと逆提案（#1057）

プロジェクト本体とAI制御（AGENTS.md・`.agents/`・skill・memory・GitHub運用・この自動化自体）を
共同で改善し続けるため、週1回「運用レポート」issueを作り、人間がチェックした提案を `ai-proposed` 子issueにする。
設計原則と合意事項は親issue #1053 を参照する。

## 構成

| パス | 役割 |
|---|---|
| `weekly_report.py` | CLI入口（`report` / `triage`） |
| `opsreport/collect.py` | 決定的な収集。gh（open PR/issue、merge済みPR、CI run、過去レポート）、git（期間内commit、行数の多いファイル、TODO系マーカー、worktree/branch数）、掃除モジュール出力、共有memory、`knowledge/`、AI制御ファイル |
| `opsreport/agent.py` | agent選択、プロンプト組み立て、`../lib/agent_exec.sh` による read-only 実行、出力JSONの厳格な検証 |
| `opsreport/proposals.py` | 提案のID付与・描画・本文からのチェック状態の解析・子issue番号の追記 |
| `opsreport/report.py` | issue本文（日本語）の組み立て |
| `opsreport/triage.py` | チェック済み提案 → 子issue化（重複防止） |
| `prompt.md` | agentへの依頼文（分析観点・品質基準・出力JSON形式） |
| `systemd/` | `report`（毎週月曜 08:00 JST）と `triage`（毎日 09:00 JST）の service / timer |

分担: 収集はスクリプト（同じrepo状態なら同じ事実）、コードベース読解と提案文の生成はagent（read-only）、
issue作成・編集はスクリプト。agentはJSONだけを返し、IDとMarkdownはスクリプトが付ける。

## 提案ID・チェック・triage

- ID は `R-<ISO年>W<ISO週>-<連番>`（例 `R-2026W41-03`）。放置PR → カテゴリ順に連番を振る。
  同じ週のレポートが既にあると `report` は失敗する（IDが衝突するため）。
- 通常の提案は行頭のチェックボックスで選ぶ。選択肢つきの提案（放置PRの「継続 / rebaseして仕上げ / close」など）は
  **選択肢を1つだけ**チェックする。2つ以上、または見出しだけのチェックはエラーとしてtriageが非ゼロ終了する（他の提案は処理する）。
- `triage` は open な `ops-report` issueをすべて走査し、選ばれた提案ごとに
  タイトル `[<ID>] <提案タイトル>`・ラベル `ai-proposed` の子issueを作り、レポートの該当行末に `→ #番号` を追記する。
- 重複防止: 作成前に全issue（open/closed）のタイトル先頭 `[<ID>]` を走査する。作成後の本文編集が失敗しても、
  再実行時は既存の子issueを再利用して追記だけ行う。同じIDの子issueが2件見つかった場合は停止する。
- レポートをcloseするとtriage対象外になる。不要な提案は放置でよい。

## 手動実行

```bash
cd ~/projects/tennis-lab
# 本文を標準出力へ（issueは作らない）。agentは実行される。
.venv/bin/python .agents/ops/weekly-report/weekly_report.py report --dry-run --output /tmp/report.md
# 本番（ops-report ラベルのissueを作成）
.venv/bin/python .agents/ops/weekly-report/weekly_report.py report --agent codex
# チェック済み提案の子issue化（--dry-run で作成予定だけ表示）
.venv/bin/python .agents/ops/weekly-report/weekly_report.py triage --dry-run
```

実行ごとに `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/weekly-report/<時刻>-<command>[-dry]-<pid>/` へ
`run.log`、`collected.json`、`prompt.md`、`agent_output.md`、agent CLIのログ、`report.md`、`result.json` を残す。
gh・git・agent・掃除モジュールの失敗、agent出力の形式違反、設定値の不正はすべて非ゼロ終了になる（黙ってskipしない）。

## 設定

CLI引数が優先。systemd実行時は `~/.config/tennis-lab-agents/weekly-report.env`（任意）に書く。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `WEEKLY_REPORT_AGENT` | `alternate` | `claude` / `codex` / `alternate`（ISO週が偶数→claude、奇数→codex） |
| `WEEKLY_REPORT_MODEL` | なし | agent CLIへ渡すモデル名 |
| `WEEKLY_REPORT_BASE_REF` | `origin/main` | 期間内commit・行数・TODO集計の対象ref |
| `WEEKLY_REPORT_STALE_PR_DAYS` | `7` | 何日更新がなければ放置PRとして載せるか |
| `WEEKLY_REPORT_AGENT_TIMEOUT` | `5400` | agent実行のタイムアウト（秒） |
| `WEEKLY_REPORT_CLEANUP_CMD` | なし | 掃除モジュールのコマンド（`--report-json` は自動付与） |
| `WEEKLY_REPORT_LOG_DIR` | 上記state dir | ログの保存先 |
| `WEEKLY_REPORT_AGENT_EXEC` | `../lib/agent_exec.sh` | agentラッパー（テストで差し替える） |

- 掃除モジュール: `WEEKLY_REPORT_CLEANUP_CMD` が未設定で `.agents/ops/cleanup/` も無ければ、本文に「掃除モジュール未導入」と書く。
  ディレクトリがあれば、その直下の唯一の実行可能ファイルを `--report-json` 付きで実行し、JSONをそのままagentへ渡す
  （実行可能ファイルが0個または複数ならエラー）。共有memory（`.agents/memory/`）も同様に、無ければ「未導入」と書く。
- agentは `--repo-root`（既定はこのcheckout）を読む。base refとcheckoutがずれていれば本文の冒頭に注意を出す。

## systemd --user への導入（merge後に人間が行う）

```bash
~/projects/tennis-lab/.agents/ops/install_units.sh --dry-run   # リンク予定の確認
~/projects/tennis-lab/.agents/ops/install_units.sh              # ~/.config/systemd/user へlink + daemon-reload
systemctl --user enable --now tennis-lab-weekly-report.timer tennis-lab-ops-triage.timer
systemctl --user start tennis-lab-weekly-report.service         # 1回試す場合
journalctl --user -u tennis-lab-weekly-report.service            # 実行ログ
```

unitは `%h/projects/tennis-lab`（メインcheckout）と `.venv/bin/python` を使う。`claude` / `codex` / `gh` は
`~/.local/bin` か `/usr/bin` にある前提で `PATH` を設定している。

## 改訂の仕方（固めすぎない）

- 分析観点・品質基準・出力形式は `prompt.md` にある。観点の追加や言い回しの変更はここだけ直せばよい。
  レポート自身が「制御ルールの改訂」カテゴリでこのプロンプトや本ツールの改善を提案してよい（プロンプトにそう書いてある）。
- カテゴリを増やす場合は `opsreport/proposals.py` の `CATEGORIES` を変える（プロンプトのカテゴリ一覧は自動で反映される）。
- 収集項目を増やす場合は `opsreport/collect.py` に決定的な関数を足す。観察データ欄に出したいものは `opsreport/report.py`。
- 挙動を変えたら `tests/e2e/agents_ops/test_weekly_report*.py`（fake gh / fake agent_exec）を更新する。
