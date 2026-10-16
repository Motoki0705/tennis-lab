# 並列条件の観測と比較証跡

`ConditionSpec` と `ComparisonSpec` は10/25/50の並列条件と比較に使う入力を固定する。
共通予算scope、同じ順序のprediction receipt、task adapter、モデル、effort、コード版、
レビュー方針、学習条件、CPU/GPUの枠を照合する。変更できるのはcampaign ID、公開先の
dataset version、Codex枠の人数だけ。推論器や道具の変更は別実験とし、この並列比較へ混ぜない。
未公開の代表入力とレビュー方針の選定はpilotの責務であり、計測器はその品質を承認しない。

正式CLIから使う。全入力は固定annotation root内の不変JSONとhashを指定する。

~~~bash
.venv/bin/python -m src.tasks.annotation schema --type benchmark_condition
.venv/bin/python -m src.tasks.annotation schema --type benchmark_comparison
.venv/bin/python -m src.tasks.annotation benchmark-observe --operation /absolute/condition.json --sha256 HASH
.venv/bin/python -m src.tasks.annotation benchmark-report --operation /absolute/comparison.json --sha256 HASH
~~~

`benchmark-observe` は既存scopeの計測を使い、worker台帳と現在の子プロセスを読む。
workerを起動せず、素材取得・推論・公開・学習・予算作成/解除を行わない。
観測と比較レポート生成ではwork IDごとに固定したprediction receipt・manifestと
実WorkRequestのpayload・inputsを照合する。同じclip IDでも別の入力は拒否する。
観測ごとにprediction receiptとpayloadを検証し、比較レポート作成時にはモデル・設定・
manifest・mediaを含む固定入力の参照も検証する。重み・media全体のhash計算を
観測ごとに繰り返して負荷に混ぜない。

観測は `campaigns/<benchmark>/reports/benchmarks/<condition>/observations/` に連番で保存する。
仕様変更・連番欠落・上限到達時はエラーにし、過去の観測を置換しない。
比較レポートは同じrootの `comparisons/<content-hash>.json` に保存する。
未開始の条件も消さず、同じ証跡からは同じレポートを得る。

## 並列数の意味

supervisorは実際に起動したCLI子プロセスのPID/start ticks/group/session/boot IDを
`child.json` に不変保存する。supervisorだけ生きている状態や、資源を予約しただけの枠を
稼働workerに数えない。子プロセスのidentityを共通の時点の前後で読み、両側で同じ非zombie
プロセスが生存していた場合だけ、その時点の同時稼働として数える。
この値はAPI/tool待ちを含むCLIの生存数であり、同時API呼出し数ではない。
短時間で終了しidentityが取れなかった子、旧版にchild証跡がないattemptは数えない。
test rootやtest commandは必ずsyntheticとして分離する。

| 表示 | 意味 |
|---|---|
| `not_observed` | その条件の観測なし |
| `unreached` | 観測はあるが、実workerが設定人数に達していない |
| `synthetic_only` | 実人数の到達証拠がなく、合成workerを観測した |
| `observed` | 実workerの観測人数が設定人数以上だった |

`workers_complete` は対象worker全件の完了だけを示す。保留、失敗、未登録、資源停止は
別表示にする。品質policy未設定、評価証跡なし、taskによる評価未実施も区別する。
レポートは常に `acceptance: unverified` とし、ファイル添付だけで受入合格にしない。

## 資源とusageの集計

- PSS/RSS/保存量/ファイル数/VRAMはscope全体の観測最大値、空きは観測最小値。
  資源sampleの間隔と保存量の計測時刻の古さ、稼働中の他planのdigestも残す。
  真のピークや当該条件だけの消費量とは呼ばない。GPUの欠測はnull。
- GPU時間は同じscopeの累計課金値の差分と終了時の稼働分を別欄にする。
  元台帳に保守的な予約全額の課金があれば保持し、GPU使用率や工程別の実測時間へ読み替えない。
- I/Oは連続するsampleに共通のboot/PID/start ticksだけで差分を加算する。
  消えたprocessの終了直前のI/O、最初のsample以前のI/Oは推定で補わない。
- 同じcache済みResourceSampleを再取得しても計測回数を増やさない。
- usageはattempt/thread/response IDの一意な記録から集計する。通知件数をAPI回数にしない。
  reportやIDやmetricが欠ければ合計をnullにし、確認できた小計を保持する。
  reportはあるが応答別記録が0件のattemptは `unobserved_response_attempts` に数え、
  他のattemptで応答を観測していても合計と正確なAPI応答数をnullにする。
  画像は重複込みbytesとhashで重複を除いたbytesを分け、prompt/toolの文字数、操作分類、
  compaction、対応不能なtool出力も併記する。操作分類の意味はtask adapterが所有する。

## 同じ入力を条件別にレビューする境界

`phase = benchmark` のplanは、productionと同じ根拠付きreview policyを要求する。
`register_inputs(layout, condition_ref, adapters)` は共有台帳に登録された未採用の予測を
検証し、条件ごとの `campaigns/<campaign>/benchmark/registry.sqlite3` へ参照を登録する。
clip ID、frame/PTS、media、予測receiptを変更せず、素材取得や推論器の実行はしない。
元のtaskが採用済み、参照・モデル・設定が登録内容と違う、scopeが未登録なら拒否する。
同じ要求の再送では条件内のレビュー進捗を保持し、別の入力集合への差し替えは拒否する。

`AnnotationLayout.for_benchmark(campaign_id)` はreview台帳とannotation成果物だけを
条件別にし、ルート内の不変入力を共有する。通常planをこのlayoutで実行すること、
benchmark planを通常layoutで実行することを拒否する。条件内では画像確認・編集・
採用証跡まで記録できるが、データセット公開・学習・モデル採用の入口はproductionに限定する。
benchmarkの採用証跡を通常layoutへ渡しても、planのphaseで公開・学習を拒否する。

予算は常に共有registryの既存scopeを使う。条件を変えても使用済みGPU時間・保存量・
停止状態を引き継ぎ、新しいbaselineで予算を作り直さない。workerの資源要求IDには条件の
campaign IDを含める。clip単位の実行lockは共有のままなので、同じclipを含む条件は順番に実行する。

固定入力の保護は共有storage catalogへ先に記録する。その後の条件登録が失敗しても保護を
自動解除しない。条件内の採用成果物は条件内catalogが保護する。共有GCによる条件内catalogの
横断列挙と実削除は未実装であり、この登録を削除可の判断には使わない。

現在は計測・集計CLIと隔離・入力登録APIの実装である。通常controller/configureはbenchmarkを
拒否する。専用runnerによるworker起動と親サービスへの接続は後続で実装する。
公開・追跡/pose・学習の工程別時間とtask別の品質評価は、そのrunnerと各adapterから追加する。
CPU fixtureの結果は10/25/50の実Codex/GPU負荷試験の代用ではない。
