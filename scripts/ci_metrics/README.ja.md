# CPU CIの計測

[English](README.md) · [简体中文](README.zh-CN.md)

既存のCPU対象を観測する道具です。テストの削除、marker追加、fixture変更、変更パスでの
省略は行いません。CIは[全件の分割と完全性照合](sharding.ja.md)を使い、既定のローカル
コマンドは全件を1 jobで実行します。CPUのmarker式は `.spin/cmds.py` が正本で、court datasetの
既存の明示除外を維持します。CIとローカル計測は `spin test -q --no-cov --dist=worksteal`、
2 worker、native thread各1、CUDA不可視、`--durations=50 --durations-min=1`、JUnitを使います。

## 実行・中断後の回収

計測対象のcheckoutとlock済み環境で実行します。

```bash
.venv/bin/python -m scripts.ci_metrics run --output /absolute/new/output --workers 2
```

出力先は空である必要があります。repoのpytest並列実行方針に合わせ、既定では
ホスト共有lockを取得せず、`PYTEST_LOCK` も使いません。明示的に調整する比較計測だけ、
`--lock /absolute/path` で同じpathを指定した計測同士を直列化できます。通常のpytestは
排他対象にならないため、ホストの競合負荷を記録し、同条件で比較してください。

lock待ちを含め、起動後のsourceは固定します。待機中に変わった場合は、import済みの計測
コードが古いまま動かないよう新しいprocessで起動し直します。

部分実行は `-- tests/unit/utils/geometry/test_angles.py` のように指定します。追加引数があれば
`full_cpu_target` はfalseとなり、全件baselineにはできません。worker、marker、JUnit、
出力先、pluginを追加引数で上書きしないでください。

実agentのHerdr/session識別子と、外側pytestのworker識別子・オプションはテスト子process
からだけ除きます。呼出元の環境は変わりません。setup失敗やsetup skipでcall reportが
ないのは正常で、失敗・skipとphase欠測を区別します。signalはpytestのprocess group全体へ
転送し、cancelとして保存します。強制killやrunner消失では終了記録が残らないことが
あります。テストを再実行せず、残った証拠から集計できます。

```bash
.venv/bin/python -m scripts.ci_metrics report --output /absolute/partial/output
```

`run` はpytestの非0終了を保持します。pytestが成功しても計測に欠損があれば終了値2です。
`report` も成功した観測を確立できなければ終了値2です。欠損を別runで埋めません。
cancelや起動失敗でテストcommandが始まらなかった場合、`run.json` のテスト終了値は
nullです。CLI自体のsignal/error終了値とは分けます。

## 成果物と時間の意味

| 成果物 | 内容 |
|---|---|
| `environment.json` | revision・未commit差分、command、依存版、OS/image、CPU/affinity、RAM、worker/thread、run/attempt、cache、開始時負荷 |
| `run.json` | 要求・開始・終了、lock待機、終了値/signal、実行中のsource整合性 |
| `events-*.jsonl` | processごとにflushした収集・除外nodeid、収集失敗/skip、worker収集、fixture setup、各phase、終了 |
| `inventory.json` | 選択されたnodeid。すべてのworkerで順序も含め一致を確認 |
| `results.json` | nodeごとの成功・skip・失敗・xfail・xpass・未完了とphase時間 |
| `junit.xml`, `pytest.log` | 独立したJUnit件数・identity・時間照合、標準durations出力 |
| `summary.json`, `summary.md` | 完全性、process起動/収集、file/fixtureごとの寄与、結果件数 |

`full_cpu_baseline` は既定の全CPU対象と成功した観測の両方を必要とします。
runner/session/JUnit欠測、worker収集不一致、phase欠落・重複、JUnitとの件数・identity・
時間の不一致、実行中のsource変更があれば不成立です。失敗とcancelは成功にしません。
収集中のmodule skip/errorは選択nodeidに対応しない場合があるため別eventです。

UTC時刻はActionsとの対応に使います。command、lock待機、起動、収集、後処理の区間は
同じhost上の `time.perf_counter` で測り、host時計の補正に影響されない経過時間を残します。
後処理の起点は最後のworker phase reportです。実時刻の差と単調時計の経過時間との差も
別に保存します。単調時計の観測が欠けた区間は未確定とし、成功baselineにはしません。
実時刻の引き算で埋め合わせません。

phase時間は**worker間の経過時間の合計**で、CPU時間やjob実時間ではありません。
起動はPython/spin/pytestからplugin importまで、worker収集はsession開始から収集完了まで
です。各timestampとcontrollerが全worker収集を受信した時刻も保存します。後処理は最終
phase終了からcommand終了までです。fixtureは依存解決後の実setup回数・時間・失敗を
観測します。finalizerはtest teardownに含まれます。fixture setupはtest setupの内数であり、
加算しません。共有fixtureはscopeのnodeへ帰属させ、全consumerへ重複配賦しません。

RSSはpytest controller/workerの**単一processの最大high-water値**（Linux KiB）です。
テストの子processは含まず、process treeの同時ピークや必要メモリ合計ではありません。
eventの直列化/flush時間とreport生成時間を記録します。write時間は計測overheadの下限です。
全体overheadは同じlock・環境で小suiteの計測あり/なしを交互に実行し、command・全標本・
失敗も残して確認します。起動時間が支配する小標本から全件のoverheadを推定しません。

## Actionsのattempt

CIは依存準備より前にrunnerを記録します。summary・REST snapshot・artifactは `always()`
で準備/テスト失敗後にも実行します。強制cancel・runner消失でuploadができない場合も、
REST側ではcancel/不完全なattemptとして残ります。artifact名はrunとattemptを含みます。
cache hitは `setup-uv` の出力から取り、取得できないものはunknownです。

job内のREST snapshotは暫定値です。自身のjob、upload、post-cacheがまだ終わっていません。
**job終了後**に再取得して全runner時間と後処理を確定します。

```bash
.venv/bin/python -m scripts.ci_metrics fetch-actions --repo Motoki0705/tennis-lab \
  --run-id 39767861734 --output /absolute/new/actions
.venv/bin/python -m scripts.ci_metrics actions --input /absolute/new/actions/actions-raw.json \
  --output /absolute/new/actions
```

`fetch-actions` は読み取り専用 `gh api` で全attemptと各attemptの全job/pageを取得します。
`--attempt` は暫定snapshotを1件指定します。取得成功でも計測が完了したとは限らず、
`complete_measurement` と `successful_observed_checks` を確認します。`actions` は欠測・
不完全データで終了値2です。入力はattempt固有のREST fieldを持つ `{run, jobs}` の配列で、
任意の `requested_at`、`cache_condition`、出典を保持できます。重複attemptは拒否し、
以前のattempt欠落は一覧化して全attempt累計runner時間を未確定にします。

queueはattempt起点→最初のjob開始と、各job作成→開始を分けます。attempt 1の起点は
`created_at`、再実行は要求時刻を取得できないことが多いため `run_started_at` を**代理値と
明記**します。独立に観測できた場合だけ `requested_at` を渡します。check完了時間は起点から
指定checkすべての完了まで（既定 `--check Python`、複数指定可）です。失敗/cancel attemptや
並行した別jobも含む**全jobのrunner時間の和**とは分けます。`Python` は観測名であり、
廃止済みのCI merge条件を復活させません。

stepはcheckout、uv/cache/install、apt、lint、pytest、post、その他を分け、元の時刻と結論を
保存します。時刻欠測は理由付き未確定値で、0秒の成功にはしません。skipはskipのままです。

中央値とp95は成功し計測が完了したattemptだけを用い、nとnearest-rank式
`sorted[ceil(0.95*n)-1]` を記録します。n<20のp95は参考値です。結果件数とrunner累計には
失敗/cancelも含みます。revision・runner・cache・inventoryが異なる観測は改善前後として
比較しません。後の成功を前の失敗へ黙って置き換えません。

## 比較条件

job構成比較ではsource、inventory、runner image/CPU/RAM、依存、worker/thread、cache、
競合負荷を固定します。テスト統合ではinventory自体が変わるため、削除case→維持する
不具合/境界契約の対応、初期化・fixture/CLI回数、setup/call/teardown時間を記録します。
寄与時間は重なる場合があり、そのままjob実時間の削減幅ではありません。上記の全件実行
経路を維持します。候補とbaselineの根拠は [baseline.ja.md](baseline.ja.md) を参照。

APIの正本は [pytest hooks](https://docs.pytest.org/en/stable/reference/reference.html#hooks)、
[Actions attempt API](https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run-attempt)、
[`setup-uv`出力](https://github.com/astral-sh/setup-uv#outputs)です。
