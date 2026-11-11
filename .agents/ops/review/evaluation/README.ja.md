# 非公開のレビュー比較

[English](README.md) · [简体中文](README.zh-CN.md)

[親 #1797 の承認](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6786941605)
に基づく比較の準備と、明示実行用のharness。
この実装PRでは実LLMの起動も正式レビューの実機受入も行わない。
PRレビュー後、親から別の実行指示を受けて実施する。
通常のpytestはオフラインの契約テストだけを実行し、比較を起動しない。

## 固定した資料

[manifest.json](cases/manifest.json) をsnapshotの正本とする。
各ケースのhead、base、merge-base、diffの生バイトの照合値、cutoff、入力照合値を保持する。
必要なcommitがローカルに無ければ停止し、自動fetchしない。
[gold.json](cases/gold.json) はmajor 9件の正解表、当時の親の確認と回帰テストの根拠、
暫定負例2件の独立した静的確認を保持する。当時のテスト結果は元の確認者に帰属し、
今回の資料作成で再実行した結果として扱わない。

| PR | 比較する境界 | 妥当性の根拠 |
|---|---|---|
| [#1699](https://github.com/Motoki0705/tennis-lab/pull/1699) | URL資格情報とrawログの公開 | 親の修正確認、修正前後の再現、ignore規則の確認 |
| [#1766](https://github.com/Motoki0705/tennis-lab/pull/1766) | 取消と自動継続の競合 | 親が確認した取消の回帰テスト |
| [#1867](https://github.com/Motoki0705/tennis-lab/pull/1867) | 古い復旧で後続指摘を隠す経路、eventとfinalの不一致 | 親が確認した復旧とruntimeの回帰テスト |
| [#1872](https://github.com/Motoki0705/tennis-lab/pull/1872) | cleanup未確認、runtime初回保存前の中断 | 親が確認した排他と起動境界の回帰テスト |
| [#1888](https://github.com/Motoki0705/tennis-lab/pull/1888) | 部分reviewedの設定照合、handoff候補の欠落 | 親が確認した部分結果と障害時finalの回帰テスト |
| [#1686](https://github.com/Motoki0705/tennis-lab/pull/1686) | memory文書で人間の指示を帰属する規則 | 固定diff全体と参照先の指示を独立確認 |
| [#1744](https://github.com/Motoki0705/tennis-lab/pull/1744) | player-viewの保存scene bundle | bundle、assets、CLI、座標変換、テストのassertを独立確認 |

`inputs/` は、元のレビュー起動前に保存されたcontextから作った射影である。
cutoffは当時のrunner起動時刻。親の要件、対象子の要件、PR本文、変更パス、保存済みCI要約、
当時の指示を残す。兄弟の進捗と `parent-*` の運用コメントを除く。
cutoff後の作成と編集をともに除外し、古いコメントへの後日の追記も使わない。
provenanceには元contextの照合値、除外コメントID、無害化件数を残す。
保存されていなかったPRと子のdiscussion、CIのrawログは取得不能のままとする。

準備でコピーするのは固定入力とdiffだけ。正解表、修正commit、現在のcheckout、共有memory、
後日のGitHubコメントにはレビューtoolから到達できない。
git取得は固定revisionとその祖先だけを許す。GitHub GETは登録したendpointだけを返し、
未登録は `UNREGISTERED_ENDPOINT` で拒否する。実GitHubへの振替はない。
コメントの保存先はローカルだけで、レビューtoolには書込みを公開しない。
取得は既存の上限付きtoolとjournalを使い、正式CLIの認可、公開、排他、gateを緩めない。

reviewerのcwdには空の非公開git repoを作る。比較専用launcherでshell、画像、browser、appsとplugins、
hooks、memory、追加agent、code mode、web検索、既存MCP、project指示を無効化する。
共通runnerの引数と専用MCPの登録設定だけを受理し、必要な制約がCLIに無ければ起動前に拒否する。
最初のCLI版を固定する。設定は公式の
[Codex設定項目](https://learn.chatgpt.com/docs/config-file/config-reference) を利用する。
実際のtool一覧の確認は、別途承認されたpilotで残る。
CPUテスト実行は比較では提供しないため、正式レビューの全tool環境と同一ではない。

## 準備と明示実行

checkoutと正式parent stateの外に専用stateを置く。
harnessの出力、runtime、取得receipt、ローカル書込みをprivate権限で保存する。
Codex自身は通常のローカルCLI transcriptを別途保存する場合がある。
raw runtimeとstateはcommitしない。

```bash
export PYTHONPATH=.agents/ops
EVAL_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/review-evaluation/minimum"
.venv/bin/python -m review.evaluation verify --repo .
.venv/bin/python -m review.evaluation init --state-dir "$EVAL_STATE"
.venv/bin/python -m review.evaluation prepare --repo . --state-dir "$EVAL_STATE" --case 1699
```

最初の起動前に、親のケース別選択を
`record-selection --state-dir "$EVAL_STATE" --spec selection.json` で固定する。
specは `schema_version: 1`、親の具体指示を示す非空の `source`、
承認された各ケース1件の `{case_id, model, reason}` を持つ `choices`。
予約後は変更できない。未指定の起動はCLIの事前検査とモデル起動の前に拒否する。
条件③のケース別選択案を親が固定してから、pilotの起動枠を使う。

数値の実行契約は [protocol.json](cases/protocol.json) が正本。
pilot対象のAstraとSolを1回ずつ、逐次で起動する。最初の実測だけでも上振れと失敗で停止できる。
両方の実測後、`status` は [historical_usage.json](cases/historical_usage.json) を各モデルの
pilot比率で補正し、残りの組と追加枠を再見積もりする。
使用量の欠測、有効性を確認できないpilot、予算目安超過では次の起動を止め、親へ返す。
トークン予算は起動前の見積もりによる制限で、API上のhard capではない。
実行中の1回が目安を超える可能性は残る。

別ターンの親の実行指示後にだけ実行する:

```bash
.venv/bin/python -m review.evaluation run --repo . --state-dir "$EVAL_STATE" \
  --case 1699 --model gpt-6-astra --parent-instruction-file parent-execution.txt --execute
.venv/bin/python -m review.evaluation status --state-dir "$EVAL_STATE"
# 次にSolのpilotを実行し、再見積もりを確認してから残りを逐次実行する。
```

共通runnerが1回のtimeoutを守る。予約を永続化し、同時起動を拒否する。
集計済みの同じ組は再送しても再起動しない。予約後にファイルがあれば起動し直さず、
終了とcleanupを確認した上で `collect --run-id ID` を使う。
runtime欠落とcleanup未確認は予約を保持して次の起動を止める。枠を戻すために削除しない。
追加枠は `--additional` と `--reason-file` を必須とし、失敗した開始も予約数に含む。
正式レビュー、判断ログのcapture、GitHubコメントはこのharnessから作らない。
初期化時にコード、共通runner依存、入力、正解表、protocolの照合値を固定する。
比較を変更する場合は親へ返す。別stateの作成で、承認された起動数とトークン予算はリセットされない。

## 集計と親の指摘判定

`collect` は正式レビューと同じ保存final、event、runtime、実効設定、schemaの検証を使う。
runtime失敗、形式失敗、利用できない出力を別に保存する。
runtime証拠が検証できない出力は有効なレビューとして採点しない。
wall timeは予約からruntime終了までとし、手動集計待ちを除く。agent時間はrunnerの開始と終了から取る。

`adjudication-template --state-dir "$EVAL_STATE" --output adjudications.json`
で指摘とpath一致の候補を出す。親が発生条件、原因、影響を正解論点と比べ、
テスト、再現、独立したコード確認を根拠に判定する。path一致だけでは一致としない。
親は `adjudicator: parent` と正確な出力・正解表の照合値を保持し、
`matches` に `finding_id`、`gold_id`、`reason`、`evidence` を記す。
残りは理由と根拠つきで `false`、`unconfirmed`、`valid_new` に分類する。
未記入templateを判定済みとして受理しない。

`report --state-dir "$EVAL_STATE" --parent-adjudications adjudications.json` は次を分けて返す。

- 全起動の資格評価: AstraとSol、失敗、追加確認の実測を含め、元counterを1回だけ加算する。
- 条件① `astra_fixed`: 新規Astraレビューの実測。
- 条件② `reuse_fixed`: 同じAstra結果を品質の参照とし、過去の保存証拠による省略可否を別集計する。
  新しいモデル結果や実測していない復旧時間を作らない。
- 条件③ `reuse_select`: 事前の選択を各モデルの実測から射影し、実際の追加確認をすべて課金対象に含める。
  反実仮想の値を物理使用量へ再加算しない。追加確認との統合判定は親が行い、runごとの検出数を
  統合結果として足し合わせない。

検出と見逃しは正解ID単位とし、blockerとmajorを分ける。親の未判定は0件でなく保留とする。
全体が利用不能な出力は、そのケースの正解をすべて見逃したものとする。
誤指摘、指摘の確認不能、コードの `not_reviewed`、証拠待ち、形式失敗、runtime失敗、失敗の和集合を分ける。
追加起動率は、複数の異なるAI起動を持つepisode数を、観測episode数で割る。
資格評価のepisodeはケースとモデル、選択射影ではケース単位である。

`agent-runtime` のturn単位で入力、キャッシュ入力、出力を保持する。
入力はすでにキャッシュを含み、出力はreasoningを含むため、重ねて足さない。
累積counterと欠測は除外または不明のままとし、0で補完しない。
AI未起動の操作が元runの使用量を再請求する入力は拒否する。
元counterの重複は除き、同じcounterの矛盾は拒否する。料金とquotaの節約は推定しない。
改善率は全組の完了と使用量の確認後、入力+出力またはagent時間で算出する。

## 過去の再利用と残る受入

[historical_reuse.json](cases/historical_reuse.json) は
[#1790](https://github.com/Motoki0705/tennis-lab/issues/1790) の重複起動5件の読取り専用監査である。
元と再起動のsnapshotは一致し、現行validatorで保存証拠を検証できた。
ただし元結果はすべて `not_reviewed` なので、現在の復旧では5件ともコードレビューを省けない。
追加使用量と時間は実際に発生した量として残し、削減できた量としては扱わない。
coverageを昇格させず、証拠待ちも捏造しない。現在の担当権限、後続request、公開の照合は監査範囲外で、
歴史的な省略可否は正式な復旧の許可ではない。
元parent stateを明示して再生成する場合:

```bash
.venv/bin/python -m review.evaluation audit-reuse \
  --parent-state-dir "$PARENT_STATE" --output reuse-audit.json
```

最小案には正解blockerがなく、反復も1回だけである。暫定負例は無欠陥の証明ではない。
cacheと実行順の影響、正式レビューとのtool差も残る。既定はfixedを維持する。
限定selectは、Astraに比べmajor見逃しが増えず、誤指摘、確認不能、失敗が悪化せず、
実効設定を確認でき、追加確認込みで承認された改善幅を満たすことを親が確認してから行う。
公開、認可、gate、取消、競合、重大なデータ同定の変更はAstraを先に使う。

別途承認後、親が代表となる正式PRを選び、prepare/request、固定取得receipt、final/event、
runtime設定、結果公開、親の明示受領、gate dry-runまで確認する。
親未確認やCI欠落でgateが停止すること、可能な場合に新しいAIレビューなしで証拠を完了できることを確かめる。
適用できる承認済みrunで再利用、復旧、Astra handoffを確認し、元指摘と
[判断記録](../../decision_logs/README.ja.md) の因果リンクを保持する。
実PR、CI、gateの根拠と導入判断は親issueへ記録する。オフラインfixtureでは完了しない残作業である。

限定導入は対象を明示した低リスクrequestと専用configだけに使い、ほかはfixedを維持する。
重大見逃し、設定の説明不能、因果リンクの欠落、予算上振れで拡大を止める。
戻す場合は次のrequestを元のAstra/xhigh fixedへ戻す。固定済み・消費済みrequestを変更せず、
進行中runを照合し、指摘を残してから、明示指示の新requestまたはhandoffを作る。
自動retryと自動方針変更は追加しない。

## オフライン検証

変更したunitファイルと判断リンクのE2E 2 nodeを `pytest -n 0` で逐次実行する。
小さいgit履歴と人工runtimeを使い、CIで過去のcheckout、実LLM、GPU、実GitHubを要求しない。
将来資料の隔離、未登録endpointの拒否、runner引数、pilot停止、起動上限、事前選択の固定、
cacheと元runの集計、既存判断ログの公開射影とREADME生成を検証する。
