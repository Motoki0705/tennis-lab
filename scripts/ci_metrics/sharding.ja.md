# 全CPUテストの分割

[English](sharding.md) · [简体中文](sharding.zh-CN.md)

CIはlintを独立実行し、要求されたCPU対象を既定10 job（各2 worker）に分けます。手動dispatchでは `shards=1/2/4/6/8/10` と `workers=2/4` を選び、同じsourceで比較できます。ホストに依存する `auto` ではなくworker数を明示します。手動1 jobの基準測定は上限120分、通常shardは60分です。固定名 `Python` は
`always()` で動き、plan・lint・全test job・完全性照合の成功を要求し、merge必須checkを維持します。
PR選別とmain/手動の全件要求は[レーン登録表と計画](README.ja.md#レーン登録表と計画)が正本です。
先行runの取消はPRだけです。

## 分配とfixture

各workerは現行sourceを通常のCPU markerと既存のcourt dataset除外で収集します。
CPU markerで選んだ後、レーン選別前の全件をeventへ記録し、要求されたnodeidを分配します。未登録の新規テストも含めます。
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

schema-2の各 `shard-gw*.json` に `universe_nodeids`（U）、`requested_nodeids`（R）、
`deferred`（D: nodeidとsuite id）、plan/policy/scope、group、選択nodeid（S_i）、profile、未計測値を保存します。
独立した `cpu_universe` eventにCPU marker後・レーン選別前のUを記録し、`lane-plan.json` に固定要求を残します。
通常のphase・JUnit・source・環境と合わせ、集約は次を要求します。

- 全shard indexと全worker manifestが過不足なく揃うこと。
- cleanなcommit、run、attempt、profile、plan、policy、Uが一致すること。
- 登録表とsourceのfile台帳が固定planと一致し、独立eventのUとmanifestのUが一致すること。
- U = R ⊎ Dを満たし、各繰延べcaseが登録表とplanで説明できること。
- 要求scopeの観測が完全で成功し、実行一覧が選択一覧と一致すること。
- R = S_0 ⊎ ... ⊎ S_nを満たし、欠落・重複・空shardがないこと。
- plan・lint・全matrix jobが成功すること。skip/cancelは成功へ変換しないこと。

全manifestから同じcaseを消しても、独立した選別前eventとの照合で検出します。Dはskipにも成功にも数えません。
集約は `scoped_cpu_success` を表示し、`full_cpu_success` はさらにDが空であることを要求します。
Uは既存のmarker・明示除外後のCPU対象であり、全環境の全テストを表しません。
過去のschema-1 artifactは `full_nodeids` の意味を保ち、profile用に読み取れます。schema混在やlane planとの併用は拒否します。

準備失敗、分割失敗、成果物欠落、失敗・取消・別HEAD混入は `Python` を失敗にします。
意図したCPU除外と既存の正当なsetup/module skipは従来どおり可視化します。matrixの
fail-fastは無効です。aptのretryとstep時間には上限を設け、取得失敗を検証省略に変えません。

artifact名はrun/attempt/shardを含み、`cpu-test-coverage-*` の `coverage.json` には
scope・suite結果・失敗理由を保存します。利用側は同じ識別情報でJUnit/manifestと結合できます。
再試行は全jobを実行してください。以前のattemptの成功結果を部分再試行へ混ぜません。

## 計測上の制約

分配に使うphase合計はrunner実時間ではありません。jobを増やすと環境準備・収集も増えます。
負荷分散・メモリは実測で確認し、証拠と比較の制限は[baseline](baseline.ja.md)を参照してください。
fixtureやsuiteが大きく変わったら完全な計測からprofileを更新します。file件数だけで分割数を決めません。

## 実測profileの再生成

同じattemptの `cpu-test-measurements-RUN-ATTEMPT-shard-*` を全件ダウンロードし、shardごとに1ディレクトリ置きます。

```bash
.venv/bin/python -m scripts.ci_metrics profile --root /absolute/artifacts \
  --output scripts/ci_metrics/cpu-file-times.json
```

phase/JUnit reportを再構築し、source/run/attempt、全worker manifest、割当、全inventoryの多重集合を照合します。phase欠測、重複、dirty/mixed source、cancel、収集失敗は拒否します。完全に記録されたテスト失敗は重みと `source.counts` / `unsuccessful_cases` に残します。重み生成はCI成功を意味せず、`Python` は引き続きjobと観測の成功を要求します。監査の `observations_complete` は成功と別に完全性を示します。

繰延べcaseが1件でもある回からはprofileを生成できません。部分成功で全CPUの重みを上書きしたり、全件baselineと表示したりしません。

既定は観測した `split_files` を維持します。fixtureを確認して一覧を置き換える場合だけ `--split-file tests/path.py` を繰り返し指定します。収集時のguardが現行fixture graphに対して検証します。merge gate、board messages、parent sessionはfunction fixture単位の分割を維持します。座標学習はmodule scopeの `shared` を使うため、9件579.907秒を同じshardに保持します。テスト本体とfixture scopeは変更しません。

更新profileは[run 41043837101](https://github.com/Motoki0705/tennis-lab/actions/runs/41043837101)（#1527/#1529のfixture最適化後）の832 file、12,883件（12,787 passed / 96 skipped / 0 failed）、合計6,681.075 phase秒に基づきます。4 / 6 / 8 / 10 shardの推定最大負荷は1,670.269 / 1,113.513 / 835.135 / 668.108秒です。重複するworker phaseの和でありwall時間の予測ではありません。旧profileでの10 shardの実測負荷は156–916秒でした。テストコストを変える変更がmergeされたら再生成します。


依存準備は既存ffmpeg/rcloneのversionを確認し、不足binaryだけを導入して両コマンドを再確認します。aptは各180秒、network retry 2回・timeout 30秒、Actions stepは8分を上限とします。既存/不足・update/install失敗・timeout・導入後欠落・version失敗の7経路をsubprocessで検証します。

## 既定値の選定

既定は10 shard・2 workerです。実測した2 worker構成のうち、既知のREADME失敗以外の失敗なく15分以内に完了した最小構成を選びます。4 workerはprocess cleanup直後の追加assert失敗が再発したため、手動比較用として残します。時間・runner費用・cache・RSSの全比較表と標本の制限は[baseline](baseline.ja.md#runner構成の比較)を正本とします。親のfixture軽量化をmerge後、profileを再生成し、6/8 shardへ減らせるか再計測してください。

`test_parent_development.py`・`test_execution_basis.py`・`test_parent_completion_resume.py`もfunction fixtureであることを確認しました。file costは更新後の各shard予算内なので、実測した既定では既存3 fileの分割宣言を維持し、予定のfixture軽量化後に新しい計測で再判断します。
