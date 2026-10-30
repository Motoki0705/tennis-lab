# CPU CIの計測

[English](README.md) · [简体中文](README.zh-CN.md)

CPU対象を計画・観測する道具です。CIは[分割と完全性照合](sharding.ja.md)を使い、既定のローカル
コマンドは全件を1 jobで実行します。CPUのmarker式は `.spin/cmds.py` が正本で、court datasetの
登録表で選択し、runnerでの明示除外はありません。CIとローカル計測は `spin test -q --no-cov --dist=worksteal`、
worker数の明示（ローカル既定2、CIは[分割設定](sharding.ja.md)参照）、native thread各1、CUDA不可視、`--durations=50 --durations-min=1`、JUnitを使います。

pytestのstdout/stderrはjobログへ転送し、`pytest.log`にも保存します。既存の`ci_watch`が失敗nodeidとsetup例外を識別できます。出力pipeが閉じた場合はlauncherを失敗にし、テストのprocess groupを停止します。


## レーン登録表と計画

[`lanes.json`](lanes.json) には、CPUの常時実行から外す **file単位の例外だけ** を登録します。
各suiteは `id`（一意の識別子）、
`files`（実在する明示的なtest path）、`paths`（依存先のrepo相対shell pattern）、
`reason`・`owner`（空でない文字列）を持ちます。fileの重複・不存在、未知の項目、
担当者・理由の欠落、不正なpatternはplanを失敗にします。fileの探索は `pyproject.toml` の
pytest設定に従います。未登録fileは常に実行し、登録済みfileは新規caseを含む収集case全体が
例外の対象です。`paths` が空でないsuiteは、自身のtest file変更でも実行します。
空の `paths` 配列は、通常のpath選別PRでは自身のtest file変更も含めて繰延べる明示指定です。
ユーザーが承認した全repoのarchitecture巡回に使います。後述の全件要求には、すべてのsuiteを含めます。
登録時は局所helper・設定・依存先のpathも含めてください。suite一覧・理由・担当者の正本は登録表です。

PRではeventの `base.sha..head.sha` を比較し、前のPR runが取消されても全commitの変更を含めます。
path選別はPRだけです。mainへのpushと手動dispatchは常に全CPUを要求し、定期レーンは作りません。
共有基盤（`src/utils`・`src/tasks/base`・`tests/support`・すべての `conftest.py`・依存設定/lock・
`.spin`・workflow・本CIツール/登録表）の変更、rename、削除、差分取得不能では理由付きで全件を
要求します。不正な登録表・planを部分実行へ変換しません。

標準ライブラリだけを使う `plan` jobが、source/run/attemptに固定した `plan.json` を1つ生成し、
全shardと `Python` 集約へ渡します。

```bash
python3 -m scripts.ci_metrics.plan --output .ci-plan/plan.json
```

コマンドは `GITHUB_EVENT_PATH` とActions contextを読みます。登録表とsourceのtest file台帳を
各checkoutと照合します。通常のshardコマンドに `--lane-plan .ci-plan/plan.json` を指定し、
任意のpytest引数とは併用できません。plan未指定のローカルshard計測は明示的に全CPUを要求します。

planと集約のsummaryには、繰延べsuite・担当者・理由・mainで最後に全件成功したcommitを表示します。
read-only Actions APIで直近30件までの成功したmain pushを探索し、祖先関係・policy・run/attempt・
`cpu-test-coverage-*` の証拠を検証します。jobの緑・部分run・全skipのsuiteは成功証拠にしません。
証拠の欠落・期限切れ・非互換・API失敗は理由付きの **unknown** として残します。
coverage artifactは90日保存し、成功indexをrepoへcommitしません。

## 追加のテスト実行環境

`player-publication` レーンでは固定済み `third_party/DINO` を準備し、
`prepare_dino_cpu --output .ci-dino-cpu` で上流の `vision.cpp` をCUDAなしでビルドします。
出力先を `PYTHONPATH` に追加し、本物の matcher/loss を読み込みます。
attention の forward/backward は例外のままで、model forwardの代替はしません。
CIの準備不足は公開loss 2件のskipにせず失敗にします。

`javascript` レーンは既存の `*.test.mjs` を `node --test` で実行し、
外部資産不要の detection refresh race を実行します。このレーンと `ops-scenarios` は、
本ディレクトリの固定済みPlaywrightとChromium headless shellを準備します。
後者ではboardのブラウザ操作を実行し、CIの準備不足をskipにしません。
court datasetのfake renderとtemp rootによるtyped configは通常のCPU選択に含みます。

GPUと `local_data and not cuda` の手動受入は
[ローカルrunnerの使い方](../github_actions/README.md#使い方)を参照してください。

## テスト増減とコストの警告

`Python` 集約jobはcoverage監査後に1つの参考レポートをjob summaryへ追加します。
file別case追加・削除・純増、時間予算を超えるcase（新規・今回超過・既存・比較元なしで不明）、
展開数の大きいparametrize関数、重いcase/fixture setup、file別profileと実測phase秒のずれ
（差分と百分率）を示します。閾値と解釈の正本は
[テスト方針](../../tests/README.md#ケース数と実行時間)で、超過によってCIを失敗にしません。

比較元はPR baseの祖先（main/手動では現在commitの祖先）にある、直前の検証済み全件成功main push
runです。現在runは除外します。read-only APIで成功したmain runを直近30件まで調べます。
成果物の欠落・期限切れ、別schema、不正/欠損データ、祖先確認不能、API失敗は理由付きで表示します。
直前の全件成功を特定した後は、そのgrowthデータの欠損・非互換を古い比較元で黙って埋めません。
導入直後は通常growth比較元がない旨を表示しますが、現在のコストとprofile差は表示できます。

Dがある回では、比較元側も含めて比較をRのfileだけに限定すると明示します。
繰延べcaseは未計測で、削除・skip・0秒として扱いません。関数のfamilyはpytestの `originalname`・
parent identity・callspec情報で決め、parameter IDの括弧を解析しません。
collection詳細はworker間で照合し、既存の `inventory.json` とlane/coverage schemaは変更しません。
詳細が欠けた場合は理由付きでコストレポートを省き、coverageの判定には影響させません。

fixture行にはscope・実setup回数・setup合計・最長の単発setupを示します。
予算超過は単発の実行で判定し、軽い複数回の合計を重い1回と誤認しません。
fixture時間はcase setupの内数で、phase合計へ再加算しません。setup skipに伴うcall欠測は
未計測のままstatusを表示します。profileは実際の分割に使ったfingerprintと照合し、
欠落・不一致は理由を示し、新規fileはprofile未計測と表示します。

`cpu-test-coverage-<run>-<attempt>` artifactには、`growth.json`（schema 1、完全な要求対象
inventory/詳細/results/fixtureコストとrun identity、または不完全な理由）、`growth-summary.json`（全レポート行）、
`growth-summary.md` も90日保存します。summaryの各表は20行までで、全行はJSONに残します。

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
| `inventory-details.json` | schema 1のoriginalname/callspecを含む収集詳細、または欠測理由。lane実行ではUを保持 |
| `lane-plan.json`, `shard-*.json`, `cpu_universe` event | 固定要求、U/R/D、レーン選別前の独立収集証拠。[分割](sharding.ja.md#証拠と失敗)を参照 |
| `results.json` | nodeごとの成功・skip・失敗・xfail・xpass・未完了とphase時間 |
| `junit.xml`, `pytest.log` | 独立したJUnit件数・identity・時間照合、標準durations出力 |
| `summary.json`, `summary.md` | 完全性、process起動/収集、file/fixtureごとの寄与、結果件数 |
| 集約の `growth.json`, `growth-summary.json`, `growth-summary.md` | 監査済みの要求対象の観測と参考の増減/コスト比較。[警告](#テスト増減とコストの警告)を参照 |

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
並行した別jobも含む**全jobのrunner時間の和**とは分けます。`Python` は
[repoの運用規則](../../.agents/README.md#github運用ルール初版)で定めるmerge必須checkでもあります。

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

完全なActions成果物からの重み再生成は[`profile`コマンド](sharding.ja.md#実測profileの再生成)を使います。失敗caseを残し、不完全なattemptは拒否します。
