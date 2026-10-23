---
id: run-court-scale-colab-resume-l4-20261023
type: run
task: synthetic_data_generation
sequence: 27
recorded_at: '2026-10-23'
title: Colab L4での4シーン分割生成・Drive永続化・再開検証
issue: 1355
provider: codex
date: '2026-10-23'
status: done
config:
  production_recipe: stratified_counter_production_v1
  method: stratified_counter
  generation_seed: 42
  trajectory_count_per_scene: 6
  density_level: 0
  scenes: [B00, B01, B02, B03]
  gpu: NVIDIA L4
  render_batch_size: 1
  without_source_images: true
  stop_after_caches: 1
metrics:
  candidate_unique: 303
  geometry_rejected: 17
  rendered_frames_total: 286
  render_cache_reused_frames: 10
  qa_accepted: 282
  qa_rejected: 4
  cache_count: 24
  generation_seconds: 137.06107280300012
  jpeg_shard_bytes: 14755301
  index_bytes: 589853
  gpu_sampled_peak_bytes: 778043392
  resumed_command_seconds: 8622.014
  fetch_command_seconds: 3663.088
  verified_image_count: 282
  verified_store_count: 24
  compared_file_count: 184
  drive_roundtrip_verified: true
  colab_stopped: true
repro:
  commit: 6ed3f29e848e71217d083f50ef0f050ecacb0981
  branch: codex/court-scale-colab-generation
artifacts:
  run_dir: knowledge/runs/run-court-scale-colab-resume-l4-20261023
  data: gdrive:tennis_lab/data/synthetic_data_generation/court-scale/smoke-20261023-6ed3f29e8
  operations: gdrive:tennis_lab/operations/colab/court-scale-generation-l4-20261023-6ed3f29e8
  dataset_version: knowledge/runs/run-court-scale-colab-resume-l4-20261023/50bb4f33f1a85e9a0afd35b6fd3ce8f3f0a1b0827b56af574e1fd6f59862537e.dataset.json
  generation_report: knowledge/runs/run-court-scale-colab-resume-l4-20261023/reports/a751e6ade3e30698ce4c5e234cf08f496d4bae755e88ccc490fa64de31675ce7.json
  verification: knowledge/runs/run-court-scale-colab-resume-l4-20261023/verification.json
  environment: knowledge/runs/run-court-scale-colab-resume-l4-20261023/environment.json
  colab_stop: knowledge/runs/run-court-scale-colab-resume-l4-20261023/stop.json
parents:
- run-court-assets-colab-l4-20261022
relations: []
papers: []
tags:
- court-detection
- colab
- rendering
- reproducibility
---

## 条件と検証対象

採用済みの画像なしB00〜B03 bundleと固定production recipeを使い、Colab L4上で
分割生成・cacheごとのDrive保存・意図的中断・空の出力rootへの復旧を接続する。
軌道groupとは別の処理単位を追加せず、既存の1軌道・1 density birth level・最大64候補のcacheを使う。
初回出力は保存したまま別名へ移し、再開とfetchはそれぞれ新しいrootを使う。

生成・選択アルゴリズム、D2/D5/D6、資産bundle、教師、QAの閾値を変えない。
seed 42・6軌道・密度0は復旧確認用の小規模条件で、本実験の規模や精度改善を検証する条件ではない。
学習・評価は実行しないためTensorBoard・収束曲線は対象外。

## 生成結果

全4シーンで生成を完了し、production v2のdataset versionとcompleted-request receiptをDriveへ保存した。
候補303件のうち幾何棄却17件、描画後QA棄却4件で、282枚をすべてtrainへ採択した。
pending・failed・空軌道は全シーン0。候補の補充や枚数を揃える選択は行っていない。
採択率は幾何棄却を含む全候補を分母とする。

| Scene | 候補 | 幾何棄却 | QA棄却 | 採択（率） | 生成秒 |
|---|---:|---:|---:|---:|---:|
| B00 | 71 | 4 | 2 | 65（91.55%） | 41.253 |
| B01 | 102 | 2 | 2 | 98（96.08%） | 36.871 |
| B02 | 63 | 4 | 0 | 59（93.65%） | 29.518 |
| B03 | 67 | 7 | 0 | 60（89.55%） | 29.419 |

幾何棄却の理由は`insufficient_in_frame_semantic_points`が16件、
`ambiguous_camera_relative_near_far:court-000`が1件。
QA棄却は`target_visible_kp_below_explicit_threshold`が4件だった。
採否の正本は[dataset version](../../runs/run-court-scale-colab-resume-l4-20261023/50bb4f33f1a85e9a0afd35b6fd3ce8f3f0a1b0827b56af574e1fd6f59862537e.dataset.json)のcounts/rejection_reasonsと、Drive上のQA ledger。
各cacheのgeneration time・GPU観測peak・JPEG/index bytes、group/scene集計は
[generation report](../../runs/run-court-scale-colab-resume-l4-20261023/reports/a751e6ade3e30698ce4c5e234cf08f496d4bae755e88ccc490fa64de31675ce7.json)から辿れる。

JPEG shardは計14,755,301 bytes、indexは589,853 bytes。
GPUの観測peakはB00が742MiB、B01〜B03が各722MiBだった。
生成後の`memory.used`は0 bytes、compute processなしを確認した。
この採択率は固定recipeの小規模条件に対する観測で、画質の目視評価・実画像への汎化・精度スケーリングの結果ではない。

## 中断と復元で確認したこと

初回は1 cacheのDrive保存確認後に`outcome=interrupted`となり、completed dataset receiptを作成しなかった。
初回rootは削除せず別名へ移し、空のrootにrequestのSHA-256だけを渡して再開した。
新しいrootに復元された最初のcacheのcompletionと計測JSONは初回とbyte単位で一致した。
根拠は[中断attempt](../../runs/run-court-scale-colab-resume-l4-20261023/interrupted-attempt.json)、
[出力rootの分離確認](../../runs/run-court-scale-colab-resume-l4-20261023/isolation.json)、
[復元cacheのhash照合](../../runs/run-court-scale-colab-resume-l4-20261023/restored-cache-check.json)。

初回cacheは11候補中1件が幾何棄却、10件が描画済みだった。
再開後のpublic NHT出力はB00が57、B01が100、B02が59、B03が60の計276撮影点で、
初回cacheのIDの再描画と新規IDの重複はともに0。
[renderer log照合](../../runs/run-court-scale-colab-resume-l4-20261023/renderer-log-verification.json)で
dataset versionの`rendered_unique`と一致した。候補数と描画数は幾何棄却の分だけ異なる。

初回cacheの生成は12.252秒、0.5秒間隔のdevice memory観測peakは742MiBだった。
中断後はGPU memory.used=0 bytes、compute processなしを確認した。
初回コマンド全体は733.712秒で、凍結入力・cache・attemptのDrive保存と確認を含む。
生成時間にはDrive転送を含めない。GPU観測peakは瞬間peakの下限で、Torch allocator peakは取得していない。
原値は[cache metrics](../../runs/run-court-scale-colab-resume-l4-20261023/first-cache-metrics.json)と
[run metadata](../../runs/run-court-scale-colab-resume-l4-20261023/run.json)に残した。

最初の計画jobはNotebook接続を失い、実コマンド開始前のpendingで停止した。
同じassignmentへ再接続し、元jobのcancelとDrive保存を確認してから`plan-retry`を提出した。
再提出で計画が完了し、コード・GPU・資産・生成条件は同じだった。
接続エラーは[提出ログ](../../runs/run-court-scale-colab-resume-l4-20261023/plan-transport.log)に保持した。

## Drive版の再取得と終了確認

公開済みDrive版を別の空rootへ`fetch`した。
`load_dataset_version(..., verify_images=True)`と、全24 storeの`validate(decode=True)`が通り、282枚のJPEGを再読込した。
さらに生成rootと取得rootの184ファイルの内容hashが一致した。
完了済み1 cacheの`reused/restored_from_drive/drive_acknowledged`はすべてtrue、
残る23 cacheは新規生成で、全cacheがDrive保存確認済みだった。
根拠は[roundtrip verification](../../runs/run-court-scale-colab-resume-l4-20261023/verification.json)、
[再開attempt](../../runs/run-court-scale-colab-resume-l4-20261023/resumed-attempt.json)、
[Drive publication receipt](../../runs/run-court-scale-colab-resume-l4-20261023/drive-receipts/completed.json)。
実際のVM破棄ではなく、初回出力rootを別名で保持し、元のpathを使えなくした同一VM内の復旧試験である。

計画・読込側はPython 3.11.16／NumPy 2.3.5／SciPy 1.17.1、
分離したNHT runtimeはTorch 2.9.1+cu130／gsplat 1.5.3／tinycudann 2.0／NumPy 2.4.6。
コードはpush済みのproducer commitから変更されていない。
検証後も同じL4で`memory.used=0`、compute processなしを確認した。
全8 jobのterminal状態と`drive_saved_at`を確認して`stop`に成功し、
終了後のdoctorはActive assignments=0／Usage rate=0.00/hrだった。
[環境](../../runs/run-court-scale-colab-resume-l4-20261023/environment.json)、
[job一覧](../../runs/run-court-scale-colab-resume-l4-20261023/all-jobs-terminal.json)、
[stop receipt](../../runs/run-court-scale-colab-resume-l4-20261023/stop.json)に保存した。

## 費用の解釈と次の試走

生成本体は計137.061秒で、render・QA・JPEG/index公開を含み、Drive転送・復元を含まない。
durable commandは初回733.712秒、再開8,622.014秒だった。
後者は凍結入力とcacheの復元・照合、未完了cacheの生成・保存、version/receipt/attemptの保存を含む。
Drive版の`fetch`は3,663.088秒だった。
セットアップは新しいVMで`uv sync`が24.798秒、NHT render-onlyが899.545秒、
資産取得・照合が57.061秒、manifest/幾何検証が21.370秒、計画が29.474秒。
VMの確保時間、durable commandのwall time、生成本体の時間を同じGPU計算時間と扱わない。

durable commandのwall timeが生成本体より大きく、本番規模の見積りには転送・確認の時間が必要である。
Drive adapterは各操作でpath成分の同名重複を確認するため、繰り返すpath解決が遅延要因の候補だが、
API単位のprofileは行っていない。cache/QA/hashの意味と保存完了の境界を維持し、
Drive CLI/library側の操作をまとめる改善を優先して調べる余地がある。

ローカルの再開テスト18件を再実行して合格した。前回のproducer commitで行ったunit等370件・integration15件の合格とは
[検証記録](../../runs/run-court-scale-colab-resume-l4-20261023/local-validation.json)で区別した。
追加の検証用Python 3本はruff・mypy、実行payloadは`bash -n`を通した。
実行したコマンド・source snapshot・returncode・Drive保存時刻は`colab-operations/`に保持している。
Gitには小さなJSON・コード・ログだけを残し、画像・index・archive本体はDriveから取得する。
学習へは`fetch`後の完全なrootにある明示的な`*.dataset.json`を渡す。

次は#1356で実画像のコート全体／部分の写り方別の評価、生成・学習の費用と規模上限を確認する。
本番規模の生成・学習、生成方式やD2/D5/D6の再選択、画質の目視による採用判断は実施していない。
