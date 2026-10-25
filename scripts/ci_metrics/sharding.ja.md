# 全CPUテストの分割

[English](sharding.md) · [简体中文](sharding.zh-CN.md)

CIはlintを独立実行し、全CPU対象を既定4 job（各2 worker）に分けます。手動dispatchでは `shards=1/2/4` を選び、同じsourceで比較できます。手動1 jobの基準測定は上限120分、通常shardは60分です。固定名 `Python` は
`always()` で動き、lint・全test job・完全性照合の成功を要求します。CIはmerge前の条件へ
戻しません。今回は既存triggerを維持し、日次保守への移行は
[#1349](https://github.com/Motoki0705/tennis-lab/issues/1349) の範囲とします。

## 分配とfixture

各workerは現行sourceを通常のCPU markerと既存のcourt dataset除外で収集します。
通常の対象選択後に全nodeidを分配し、新しいテストも必ず含めます。
`cpu-file-times.json` の実測phase時間を使い、重いgroupから負荷の軽いshardへ入れます。
同点は名前・index順で固定します。既定はfile単位で共有準備を保ち、独立したfunction
fixtureの重いfileだけ `split_files` に明示して、個別 `test_seconds` で分配できます。
追加前にfixtureの共有範囲を確認してください。後からmodule/class fixtureが追加された分割fileは収集時に拒否します。

未計測fileは計測済みfileの中央値、分割file内の新規caseはそのfileの平均を見積値にします。
推測値はmanifestに列挙し、対象の省略には使いません。不正profileは明示的に失敗します。
実行例は[英語版](sharding.md#assignment-and-fixture-boundaries)、分割しない全CPU実行は
[README](README.ja.md#実行中断後の回収)を参照してください。各shard単独は
`full_cpu_baseline=false` で、全体の検証は集約で行います。変更pathやslow markerで省略しません。

## 証拠と失敗

各 `shard-gw*.json` に全CPU一覧、group、選択nodeid、profileの識別情報、未計測値を保存します。
通常のphase・JUnit・source・環境と合わせ、集約は次を要求します。

- 全shard indexと全worker manifestが過不足なく揃うこと。
- cleanなcommit、run、attempt、profile、全inventoryが一致すること。
- 分割前に全CPUを要求し、観測が完全で成功し、実行一覧が選択一覧と一致すること。
- 実行nodeidの和集合が全CPUと一致し、重複もないこと。
- lintと全matrix jobが成功すること。skip/cancelは成功へ変換しないこと。

準備失敗、分割失敗、成果物欠落、失敗・取消・別HEAD混入は `Python` を失敗にします。
意図したCPU除外と既存の正当なsetup/module skipは従来どおり可視化します。matrixの
fail-fastは無効です。aptのretryとstep時間には上限を設け、取得失敗を検証省略に変えません。

artifact名はrun/attempt/shardを含み、`cpu-test-coverage-*` の `coverage.json` には
集約の結果・失敗理由を保存します。日次CI側は同じ識別情報でJUnit/manifestと結合できます。
再試行は全jobを実行してください。以前のattemptの成功結果を部分再試行へ混ぜません。

## 計測上の制約

分配に使うphase合計はrunner実時間ではありません。jobを増やすと環境準備・収集も増えます。
負荷分散・メモリは実測で確認し、証拠と比較の制限は[baseline](baseline.ja.md)を参照してください。
fixtureやsuiteが大きく変わったら完全な計測からprofileを更新します。file件数だけで分割数を決めません。

初期profileは[baseline](baseline.ja.md#完了したローカル観測)の全件・失敗観測に基づき、
失敗も含めます。fixture/契約整理前で他作業の負荷もあり、配分の手掛かりであって成功や
高速化の証明ではありません。旧inventoryのphase合計見積りは1 jobで11,284.355秒、
2 jobは各約5,642.178秒、4 jobの最大は2,821.090秒です。worker間の重なりとjobごとの
準備・収集を含む実時間ではなく、Actionsの実時間・runner費用は別途測定します。
個別caseへ分けるのはmerge gate・parent session・board messagesの3 fileです。
function fixtureの独立性は収集時に検査し、その他のfileはまとめます。

現行の全収集は **12,775件の一意なCPU case** で、4 shardへ2,927 / 2,994 / 3,442 / 3,412件を割り当てます。fixture独立性も収集で確認しました。これは収集の証拠であり実行成功や性能の証拠ではありません。新規2 fileと変更・追加された分割case 23件は明示した見積りを使います。

依存準備は既存ffmpeg/rcloneのversionを確認し、不足binaryだけを導入して両コマンドを再確認します。aptは各180秒、network retry 2回・timeout 30秒、Actions stepは8分を上限とします。既存/不足・update/install失敗・timeout・導入後欠落・version失敗の7経路をsubprocessで検証します。
