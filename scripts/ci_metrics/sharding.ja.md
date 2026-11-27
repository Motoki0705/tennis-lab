# 全CPUテストの分割

[English](sharding.md) · [简体中文](sharding.zh-CN.md)

CIはlintを独立実行し、要求されたCPU対象を通常の既定10 job（各2 worker）と必須のsystem runtime 1 jobへ分けます。手動dispatchでは `shards=1/2/4/6/8/10` と `workers=2/4` を選び、同じsourceで比較できます。ホストに依存する `auto` ではなくworker数を明示します。手動1 jobの基準測定は上限120分、通常shardは60分です。固定名 `Python` は
`always()` で動き、plan・lint・全test job・完全性照合の成功を要求し、merge必須checkを維持します。
PR選別とmain/手動の全件要求は[レーン登録表と計画](README.ja.md#レーン登録表と計画)が正本です。
先行runの取消はPRだけです。

## 分配とfixture

各workerは現行sourceを通常のCPU markerで収集し、レーン登録表で選ばれるcourt dataset統合も含めます。
CPU markerで選んだ後、レーン選別前の全件をeventへ記録し、marker対象の `system_deps_nodeids` もeventとmanifestへ記録します。marker対象は必ずRへ含め、通常群にはRからmarker対象を除いたcaseだけを分配し、専用群にはmarker対象全件だけを選びます。依存不要の隣接caseはfileの繰延べ方針を保ちます。未登録の新規テストも含めます。
`cpu-file-times.json` の実測phase時間を使い、重いgroupから負荷の軽いshardへ入れます。
同点は名前・index順で固定します。既定はfile単位で共有準備を保ち、独立したfunction
fixtureの重いfileだけ `split_files` に明示して、個別 `test_seconds` で分配できます。
追加前にfixtureの共有範囲を確認してください。後からmodule/class fixtureが追加された分割fileは収集時に拒否します。

未計測fileは計測済みfileの中央値、分割file内の新規caseはそのfileの平均を見積値にします。
推測値はmanifestに列挙し、対象の省略には使いません。不正profileは明示的に失敗します。
実行例は[英語版](sharding.md#assignment-and-fixture-boundaries)、分割しない全CPU実行は
[README](README.ja.md#実行中断後の回収)を参照してください。各shard単独は
`full_cpu_baseline=false` で、全体の検証は集約で行います。レーン選別は分割前に行い、時間やslow markerを選別規則にしません。

## 証拠と失敗

schema-3の各 `shard-gw*.json` に `system_deps_nodeids` と `universe_nodeids`（U）、`requested_nodeids`（R）、
`deferred`（D: nodeidとsuite id）、plan/policy/scope、group、選択nodeid（S_i）、profile、未計測値を保存します。
独立した `cpu_universe` eventにCPU marker後・レーン選別前のUを記録し、`lane-plan.json` に固定要求を残します。
通常のphase・JUnit・source・環境と合わせ、集約は次を要求します。

- 全shard indexと全worker manifestが過不足なく揃うこと。
- cleanなcommit、run、profile、固定lane plan（`plan_id`）、policy、Uが一致すること。
- attemptは正の整数の文字列で、lane planのattempt ≤ 各shardのattempt ≤ 集約jobのattemptを満たすこと。
- 登録表とsourceのfile台帳が固定planと一致し、独立eventのUとmanifestのUが一致すること。
- U = R ⊎ Dを満たし、各繰延べcaseが登録表とplanで説明できること。
- 要求scopeの観測が完全で成功し、実行一覧が選択一覧と一致すること。
- R = S_0 ⊎ ... ⊎ S_nを満たし、欠落・重複・空shardがないこと。
- marker一覧が独立eventと全shardで一致し、通常群に混入せず、専用群で全件がpassedであること（skip/xfail 0）。
- plan・lint・全matrix job・専用runtime jobが成功すること。skip/cancelは成功へ変換しないこと。

全manifestから同じcaseを消しても、独立した選別前eventとの照合で検出します。Dはskipにも成功にも数えません。
集約は `scoped_cpu_success` を表示し、`full_cpu_success` はさらにDが空であることを要求します。
Uは既存のmarker・明示除外後のCPU対象であり、全環境の全テストを表しません。
過去のschema-1/2観測は新runtime coverageを主張せずprofile用に読み取れます。schema-1 artifactは `full_nodeids` の意味を保ち、profile用に読み取れます。schema混在やlane planとの併用は拒否します。

準備失敗、分割失敗、成果物欠落、失敗・取消・別HEAD混入は `Python` を失敗にします。
意図したCPU除外と既存の正当なsetup/module skipは従来どおり可視化します。matrixの
fail-fastは無効です。[runtime準備の正本](README.ja.md#固定したシステム実行環境)に従って取得時間を制限・検証し、失敗を省略へ変えません。

部分再実行では、同じrun・同じHEAD（`source_sha`）・同じlane plan（`plan_id`）に限り、
固定planとshardごとの最新結果を使います。再実行しなかった成功shardは以前のattemptでも利用できます。
attemptは上記の範囲内で混在でき、別run・別HEAD・別planは拒否します。

planのartifact名は `cpu-lane-plan-RUN`、計測は `cpu-test-measurements-RUN-shard-N` です。通常indexは0からcount-1、専用runtimeのindexはcountです（既定10）。専用は通常CPU数を増やしません。
両方のuploadに `overwrite: true` を指定し、再実行したplan/shardだけを置き換えます。
失敗attemptの計測は上書きで消えることがありますが、job logは残ります。
全jobの再実行ではGitHubが以前のartifactを削除します。
downloadには `run-id` と `github-token` を明示し、run全体のREST APIを使います。
集約はdownload前にREST一覧を全ページ取得し、期限切れでないplanと各shardのartifactが
ちょうど1つずつあることを要求します。欠落・重複・余分なshardは理由付きで失敗させます。
検査済みのartifact IDでdownloadし、shard別ディレクトリを保ってファイルの静かな混在を防ぎます。

`cpu-test-coverage-RUN-ATTEMPT` の名前には集約jobのattemptを残します。
`coverage.json` に `attempt`（集約）、`plan_attempt`、`shard_attempts`（index → attempt）、
scope・suite結果・失敗理由を記録します。同じartifact内の `artifacts.json` に検査した取得集合の
ID・name・`created_at`・`size_in_bytes` と一覧の検査エラーを保存します。
Pythonだけの再実行でも、以前のattemptのshardをこの証拠とともに再検証できます。

## 計測上の制約

分配に使うphase合計はrunner実時間ではありません。jobを増やすと環境準備・収集も増えます。
負荷分散・メモリは実測で確認し、証拠と比較の制限は[baseline](baseline.ja.md)を参照してください。
fixtureやsuiteが大きく変わったら完全な計測からprofileを更新します。file件数だけで分割数を決めません。

準備とpin更新の正本は[runtime契約](README.ja.md#固定したシステム実行環境)です。

## 実測profileの再生成

Actionsの成果物を読み取れる認証済みの `gh` を用意し、checkoutから
`update-profile` を実行します。`--latest-main` は成功したmain pushを新しい順に
調べ、採用条件を満たさないrunは理由を表示します。`--run-id` と `--attempt` を指定
すると、成果物が有効な間は特定の更新を再現できます。

runとHEADを固定してlane plan、最新coverage、全shardをartifact IDで取得します。
集約の取得一覧を照合し、workerとJUnitの監査を再計算します。完全なmain push成功回の
うち、deferredが0件で専用runtime shardの監査を通ったものを採用します。同一planの
部分再実行では、上記のattempt規則で以前の成功shardを再利用できます。証拠の欠落や
不整合があれば出力profileを保全し、検証後の置換は原子的に行います。

計測元runが使ったprofileを比較元として、今回の更新を再現できます。

```bash
profile_work=$(mktemp -d)
git show 8d629a35289f475e514291167435c333876afdd2:scripts/ci_metrics/cpu-file-times.json > "$profile_work/baseline.json"
.venv/bin/python -m scripts.ci_metrics update-profile --run-id 45955929416 --attempt 1 \
  --baseline "$profile_work/baseline.json" --output "$profile_work/cpu-file-times.json" \
  --report "$profile_work/comparison.json" \
  --split-file tests/e2e/agents_ops/test_board_messages.py \
  --split-file tests/e2e/agents_ops/test_parent_session.py \
  --split-file tests/integration/agents_ops/test_storage_tier_crash_boundaries.py \
  --split-file tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py
```

最新runを選ぶときは `--run-id … --attempt …` を `--latest-main` に置き換えます。
既定の比較元はcheckoutの `cpu-file-times.json` です。分割指定を省くと計測元の分割宣言を
引き継ぎます。fixtureを確認して一覧を入れ替える場合は `--split-file` を繰り返します。
計測caseが0件の分割宣言は一覧の明示的な見直しが必要です。JSONレポートには出典・固定の再現
コマンド・fileの差分・全groupの割当と負荷を出力します。未計測fileやcaseが大きく増えた場合や、
実測の偏りが続く場合に候補を比較し、通常のレビュー付きPRで反映します。コマンドの書き込み先は
ローカルファイルです。

過去の計測や失敗回の診断には、取得済みshardディレクトリから再計算する
`profile --root /absolute/artifacts --output /absolute/diagnostic.json` を使えます。
完全に観測された失敗を `source.counts` と `unsuccessful_cases` に保持します。
attemptが混在する場合は最大のshard attemptを使い、deferredがある場合は拒否します。

### 現行profileのオフライン比較

旧profileは883 file・14,561件の[run 41240159671](https://github.com/Motoki0705/tennis-lab/actions/runs/41240159671)
に基づきます。更新元は[run 45955929416](https://github.com/Motoki0705/tennis-lab/actions/runs/45955929416)、
commit `8d629a35289f475e514291167435c333876afdd2` です。lane plan、計測11件、成功coverageの
runとHEADが一致し、attemptはすべて1です。1,160 file・26,189件のうちpassed 26,070件、正当なskip
119件、失敗・deferred 0件でした。専用jobは17件すべてpassedです。file重みの追加は282件、更新は
878件、現行の観測caseが0件の重みは5件です。artifact IDとinventoryのハッシュはprofileの
`source` に記録しています。

Aは旧重み、Bは観測inventoryに対して分割方針を維持した重み更新、Cは停止・復旧境界の2 fileを
追加でcase分割します。旧宣言の `test_merge_gate.py` は補助モジュールで収集caseが0件でした。
有効な分割はBが2 file、Cが4 fileです。Aで未計測値を使ったのは通常shardの281 fileと分割case
28件です。B/Cでは通常fileと分割caseに未計測がありません。

表はworkerのsetup・call・teardownの合計秒です。A実測は計測元runのA割当で観測した負荷です。
B/C推定には同じ観測を使います。専用jobの18.578 phase秒は通常shardの集計から外します。

| Shard | A推定 | A実測 | B推定 | C推定 |
|---|---:|---:|---:|---:|
| 0 | 610.579 | 725.360 | 1809.091 | 1360.607 |
| 1 | 610.580 | 1322.624 | 1310.776 | 1360.609 |
| 2 | 610.579 | 611.440 | 1310.776 | 1360.608 |
| 3 | 610.579 | 1304.961 | 1310.776 | 1360.609 |
| 4 | 610.579 | 1185.038 | 1310.777 | 1360.608 |
| 5 | 610.578 | 1397.866 | 1310.777 | 1360.607 |
| 6 | 610.579 | 1650.712 | 1310.776 | 1360.608 |
| 7 | 610.580 | 844.635 | 1310.778 | 1360.609 |
| 8 | 610.578 | 3311.545 | 1310.777 | 1360.608 |
| 9 | 610.578 | 1235.024 | 1310.778 | 1360.609 |
| 最小 | 610.578 | 611.440 | 1310.776 | 1360.607 |
| 最大 | 610.580 | 3311.545 | 1809.091 | 1360.609 |
| 平均 | 610.579 | 1358.921 | 1360.608 | 1360.608 |
| 最大 / 平均 | 1.000 | 2.437 | 1.330 | 1.000 |

復旧境界4,606件は1809.091秒で、Aのshard 8からBの0へ移ります。停止境界1,649件は
533.836秒で、Aの6からBの1へ移ります。Cは両fileを通常10 shardに分散します。次に重い
未分割fileは `test_parent_development.py` の423.166秒（A/B/Cのshardは1/2/0）、
`test_parent_completion_resume.py` の373.922秒（8/3/1）、`test_court_dataset.py` の
355.775秒（0/4/2）です。実測caseコストをCの割当に載せ替えた最大は1360.609秒で、Aの
実測最大より58.9%小さくなります。Testステップとrunner消費への効果は同一条件のActionsで比較します。

停止・復旧境界ではfunction scopeの `tmp_path`・`monkeypatch` と全体のcase別seed fixtureを
使います。pytestのsession scopeの一時ディレクトリ割当はcaseに個別のrootを渡します。
`make_lab` はrepo、mirror、模擬 `/proc`、journal、lockをその配下に置きます。trace hookは
forkした子だけで動き、子はjoinまたはkillで回収し、pytestがteardownでpath変更を復元します。
実際のshard pluginの収集guardを境界6,255件が通過し、全件が1回ずつ割り当てられ、未計測
分割caseは0件でした。

## 既定値の選定

既定は通常10 shard・各2 workerと必須の専用runtime jobです。既定値を選んだ計測、cache条件、
標本の制限は[baseline](baseline.ja.md#runner構成の比較)に記録しています。
