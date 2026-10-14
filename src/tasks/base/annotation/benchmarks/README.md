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
  画像は重複込みbytesとhashで重複を除いたbytesを分け、prompt/toolの文字数、操作分類、
  compaction、対応不能なtool出力も併記する。操作分類の意味はtask adapterが所有する。

現在は計測・集計APIとCLIの実装である。同じclipを3条件へ投入するrunnerには、
本番の再注釈防止を維持した条件別のreview台帳・成果物と、共有scopeの接続が必要。
公開・追跡/pose・学習の工程別時間とtask別の品質評価は、そのrunnerと各adapterから追加する。
CPU fixtureの結果は10/25/50の実Codex/GPU負荷試験の代用ではない。
