---
id: run-ball-sequential-length-20261025
type: run
task: ball_detection
sequence: 68
recorded_at: '2026-10-25'
title: 共通DPTからの逐次座標回帰・T64学習と事前学習時間の検証
issue: 1514
provider: codex
date: '2026-10-25'
status: done
config:
  input_channels: 5
  resolution:
  - 720
  - 1280
  seed: 42
  input_frames_per_update: 128
  common_pretraining_updates: 10000
  dpt_posttraining:
    T64: 6000
  query_posttraining:
    T32: 6000
    T64: 6000
  skipped_T:
  - 128
  compile_mode: default
  activation_checkpointing: false
  frame_chunk_size: null
  warmup_updates: 500
  posttraining_cnn_freeze_updates: 600
  encoder_lr_ratio: 0.1
  optimizer_state: inherit matching parameters
metrics:
  download_bytes: 157028755547
  download_files: 2925
  download_seconds: 2068.5761816501617
  accepted_training_packs: 30478
  excluded_position_free_packs: 2031
  unique_observed_training_frames: 491380
  unique_negative_training_frames: 95674
  edge_only_training_packs: 32
  benchmark_updates_each_batch: 30
  benchmark_bs1_median_update_seconds: 2.038441351000074
  benchmark_bs2_median_update_seconds: 2.047568384999977
  benchmark_bs1_pretraining_compute_hours: 5.662337086111317
  optimizer_updates: 28000
  terminal_validation:
    common-dpt-T32:
      stage_updates: 10000
      cumulative_updates: 10000
      distance_threshold_source_px: 20.0
      false_negative: 9372
      false_positive: 7715
      mean_error_px: 42.58463082369366
      median_error_px: 12.256753921508789
      negative_fpr: 0.24196330562960638
      observed_fnr: 0.10964866098066056
      p95_error_px: 248.14239807128848
      support:
        frames: 128668
        masked: 11310
        observed: 85473
        out_of_frame: 2712
        reviewed_empty: 29173
        scored_frames: 117358
      within_distance_rate: 0.8403004457547998
      presence_threshold: 0.5046385526657104
    dpt-T64:
      stage_updates: 6000
      cumulative_updates: 16000
      distance_threshold_source_px: 20.0
      false_negative: 15021
      false_positive: 4918
      mean_error_px: 36.559837116156196
      median_error_px: 12.235901832580566
      negative_fpr: 0.1542418064920809
      observed_fnr: 0.1757397072759819
      p95_error_px: 183.1951019287102
      support:
        frames: 128668
        masked: 11310
        observed: 85473
        out_of_frame: 2712
        reviewed_empty: 29173
        scored_frames: 117358
      within_distance_rate: 0.8657821768277701
      presence_threshold: 0.5712883472442627
    query-T32:
      stage_updates: 6000
      cumulative_updates: 16000
      distance_threshold_source_px: 20.0
      mean_error_px: 42.9942052864783
      median_error_px: 24.203832626342773
      p95_error_px: 136.21399841308568
      support:
        frames: 128668
        masked: 11310
        observed: 85473
        out_of_frame: 2712
        reviewed_empty: 29173
        scored_frames: 117358
      within_distance_rate: 0.30641255133199957
    query-T64:
      stage_updates: 6000
      cumulative_updates: 22000
      distance_threshold_source_px: 20.0
      mean_error_px: 28.200801773291
      median_error_px: 7.661407947540283
      p95_error_px: 116.888377380371
      support:
        frames: 128668
        masked: 11310
        observed: 85473
        out_of_frame: 2712
        reviewed_empty: 29173
        scored_frames: 117358
      within_distance_rate: 0.7628607864471821
artifacts:
  run_dir: knowledge/runs/run-ball-sequential-length-20261025
  output_dir: outputs/ball_detection/train/sequential_length
  terminal_validation: knowledge/runs/run-ball-sequential-length-20261025/terminal-validation.json
  drive_campaign: gdrive:tennis_lab/outputs/ball_detection/train/sequential_length/s42-20261025-main/
  data_preparation: outputs/ball_detection/precompute/sequential_length/observed-packs-20261025
parents:
- run-ball-colab-checkpoint-resume-20261024
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
repro:
  commit: cda340b2b210
  command: PATH=/tmp/ball-tools:$PATH .venv/bin/python knowledge/runs/run-ball-sequential-length-20261025/run.py
    benchmark --workspace-root /content/tennis-lab --run-id benchmark-20261025-a --rclone-config
    /tmp/ball-drive-mount.conf
---

## 完了結果（2026-10-31）

campaign `s42-20261025-main` は2026-10-31 03:06 JSTに完了した。
campaignと4段階の `COMPLETED.json` はすべて `complete: true` で、物理実行は計28,000 optimizer更新。
共通DPTの10kからDPT T64とquery T32へ分岐し、query T64はquery T32から継承した。
したがって来歴はDPT T64が16k、query T64が22kである。共通10kを物理費用として二重計上しない。
終端checkpointのvalidation結果を[指標要約](../../runs/run-ball-sequential-length-20261025/terminal-validation.json)へ保存した。
元の終端validation JSONは親がVMから回収したものを読み、採点・教師・全owner identityの一致、
全subgroupの支持数、observed加重平均と距離内率、FP/FNからの率を再計算して照合した。
中央値とP95は終端JSONの値であり、subgroupの分位点から再計算した値ではない。
予測配列 `*.frames.jsonl`、重み、更新ログはGitへコピーしていない。

| 終端段階 | 段階の更新数 | 来歴の累積更新数 | mean px | median px | P95 px | ≤20px率 | FPR | FNR | val調整閾値 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 共通CNN+DPT T32 | 10,000 | 10,000 | 42.58 | 12.26 | 248.14 | 84.03% | 24.20% | 10.96% | 0.5046 |
| DPT T64 | 6,000 | 16,000 | 36.56 | 12.24 | 183.20 | 86.58% | 15.42% | 17.57% | 0.5713 |
| query T32 | 6,000 | 16,000 | 42.99 | 24.20 | 136.21 | 30.64% | — | — | — |
| query T64 | 6,000 | 22,000 | 28.20 | 7.66 | 116.89 | 76.29% | — | — | — |

### 採点集合と指標の意味

全128,668 owner frameのうち、observed 85,473、reviewed_empty 29,173、out_of_frame 2,712、masked 11,310。
scored 117,358はobservedと確認済み不在31,885の合計で、位置誤差の分母85,473とは異なる。
4段階とも3種のidentityと各支持数が一致する。queryとDPTの位置指標は同じobserved集合を使う。
`evaluation/temporal_heatmap.py` と `evaluation/temporal_coordinates.py` は、
正規化予測とGTの差に元画像の `(W−1, H−1)` を掛けたユークリッド距離を記録する。
pxは入力のfeature格子でなく元画像のpixel単位。frame重みで平均・中央値・P95（線形補間）と≤20px率を計算する。

`evaluation/temporal_heatmap_metrics.py` の `location_metrics` は **全observed** を選ぶ。
DPTの存在閾値で不在になったframeも座標誤差へ含め、除外も別の罰則への置換もしない。
DPT座標はnative heatmapのglobal argmax＋log-parabolic subpixel補正、scoreは最大sigmoid値。
FPRは確認済み不在に対する存在判定の割合、FNRはobservedに対する不在判定の割合で、20px条件を含まない。
共通T32のFP/FNは7,715/9,372、T64は4,918/15,021である。
閾値は各終端モデルの **同じvalidation** でbalanced FPR/FNRを最小化して選んだため、独立評価の率ではない。
queryは直接座標のみを出し、presence head・存在閾値・FPR/FNRを持たない。
不在frameはcoverageへ残すが座標指標へは混ぜず、DPTとの検出率の比較は行っていない。

### 観察

DPT T32→T64ではmeanが6.02px、P95が64.95px低下し、≤20px率が2.55ポイント上がった。
中央値はほぼ同じで、誤差の裾の変化が大きい。存在判定はFPRが下がる一方でFNRが上がっており、全面的改善とはしない。
query T32→T64ではmeanが14.79px、medianが16.54px低下し、≤20px率が45.64ポイント上がった。
終端query T64はこの4条件でmean・median・P95が最小だが、≤20px率はDPT T64の86.58%より低い76.29%。
単一指標でモデル採用を決めず、既存モデル・checkpointの採用判断は変更しない。

query T32のFPS別medianは25fpsで24.22、約59.94fpsの `2997003/50000` で25.89、
30fpsで20.27、`30000/1001` で23.33、50fpsで23.83、59fpsで26.46、60fpsで25.93、
`60000/1001` で23.18pxだった。複数FPS区分で約24px前後に分布することは観察だが、一定方向のずれを示す証拠ではない。

### 仮説・比較の限界と次の実験

- query T32には座標の系統ずれ、座標変換・時刻対応の問題、または学習途上の誤差がある可能性がある。
  距離の中央値だけではこれらを区別できず、原因は未確定。保存予測から符号付きdx/dyと位置依存性、
  owner offset・MDD先頭・移動速度別の残差を調べ、同じframeのGT overlayと座標の往復変換を確認する。
  時間を前後にずらした対応との比較も行い、空間offsetと時間ずれを区別する。
- 系列長と累積更新数が交絡している。T32→T64は両系統とも追加6k更新、最終DPT16kとquery22kでは来歴も異なる。
  共通のCNNを使ってもquery headは新規で、目的関数・復号方式も異なる。T64の効果や方式単独の優劣は識別できない。
  次は同じ親から同じ追加更新数・入力frame数でT32/T64を分岐し、複数seedで比較する。
- seed42の1回だけで分散・信頼区間は未評価。これは終端validationの比較で、別版testは未評価。
  #1476 / PR #1489のtest配布・評価契約を先に整え、checkpointとDPTのval閾値を固定して独立testを評価する。
- queryにはpresence出力がないため、FPR/FNRをDPTと比較していない。位置指標だけから実用上の不在検出能力を推定しない。

## VM復旧と完了までの実行記録

以下の復旧性能・費用・時刻は親と前ターンの子の実行観測であり、本記録時に更新ログから再測定した値ではない。
2026-10-30の復旧は同じ配布入力2,925ファイル・157,028,755,547 bytesを読み取り専用rclone mountから取得し、
全件のサイズと独立2系統のSHA-256を照合した（約3.8時間）。共有OAuthクライアントのDrive API quotaによる
403 `rateLimitExceeded` は再試行で回収した。下の初回取得時間2,068.58秒とは別attemptの測定である。
`mounted_drive` aliasを同じDrive rootへ向け、元receipt名を維持した。
復旧controllerのcommitは `terminal-validation.json` の `recovery_commit` に記録した。
`COMPLETED.json` 内のplanのcommitは初回計画の版を保持しており、復旧コードの版と区別する。

DPT T32/T64のcompleted世代は検証して再学習をスキップした。
query T32は `checkpoint-00005000-a3833983` の完全checkpointからAdamW状態201件、
RNG、schedule、抽出位置15,000を復元し、5,001から再開した。
初回compile込み120.50秒、定常1.038秒/update、入力待ち平均0.52ms、
`nvidia-smi` は10,114/23,034 MiB、peak allocatedは6.39 GiB。
5,500 snapshotの保存・検証は442.72秒で、更新時間と別に数える。
lossは5,000で0.006965、5,001で0.006707、中断前500更新平均0.010883、再開後500更新平均0.009597。
これは局所的な連続性の観測であり、異なるsample列の平均から再開の数値的同一性を証明するものではない。

復旧train jobは2026-10-30 18:02〜2026-10-31 03:06 JST、returncode 0で終了し、
成果物はDriveのcampaign相対pathの各世代と `persistence/<stage>/*.json` に保存済み。
この時間は初回計算・入力再取得を含むcampaign総所要時間ではない。Colab消費は約18 CU、sessionは停止済み。
TensorBoardは生成せず、曲線の根拠はDrive最終completed世代の `train.jsonl` とする。
本記録では全学習曲線を再解析せず、収束を主張しない。入力・過去の成果物・保存世代は変更・削除していない。

## 初回準備時点の記録（2026-10-25）

以下は当時の計画・測定であり、未開始・計測中の記述も準備時点を指す。

ユーザーが#1514の実装と本学習を依頼し、最初にDriveから学習データを取得した。
新しい共通事前学習はCNN＋DPTのT32で10,000更新とし、過去の2ch/5ch checkpointを流用しない。
LRとAdamW状態の逐次継承は会話で確認済み。T128はL4の分割なし試走でOOMを観測した後、
ユーザーが今回の実行から明示的に除外した。残る物理実行は共通10kと事後3段×6kの計28k更新。
DPTの来歴は16k、queryの来歴は22kで、共通事前学習を実行費用として二重計上しない。

Driveのtrain/validation配布版を全件取得し、全ファイルのサイズと独立した2系統のSHA-256を照合した。
入力は1,537 clipの既存の不変配布版で、元データ・注釈・splitを変更していない。
同じ128枚組をT32/T64に分割するmanifestを再生成し、32,509→30,478組を確認した。
位置教師のない子窓はT32で10,383出現、T64で2,780出現あり、座標lossのゼロ分子と更新全体の分母で扱う。
全組の先頭/末尾だけに位置教師がある32組では、遮蔽のrequested/eligible/appliedを記録して教師を保持する。
validationの共通集合は128,668 frame、observedは85,473 frameを保持する。
別版test配布の完成・評価は今回の学習完了を根拠に推定しない。

可変Tの直接座標の保存形式、DPTからCNNのみの転送、同構成の全重み転送、
名前に基づくAdamW moment/step継承と段階ごとのCNN凍結/解除を実装した。
CPUの実モデル試験ではDPT T32→T64→T128とquery T32→T64→T128、queryの中断再開を検証した。
T128のCPU対応は実装の検証であり、今回のGPU本学習へT128を追加したものではない。
直接座標評価はheatmapと同じowner/observed集合を採点し、存在head・FPR/FNRを新設しない。

生の合成RGBによる720p・BF16・compile有効・checkpointing無効の予備診断では、
DPT T32/BS1は初回130.61秒、定常約0.48秒/32枚、peak allocated約7.77 GB、
T64/BS1は初回112.56秒、定常約0.98秒/64枚、peak allocated約15.43 GBだった。
T128/BS1とT32/BS4はOOM。T128/chunk4の追加診断はユーザーのT128除外指示で中止し、
成立・不成立や速度を推定していない。これらのraw計算値を、拡張・復号・保存を含む事前学習時間と混同しない。

実データ30更新のBS1/BS2比較が完了し、更新中央値はそれぞれ2.0384/2.0476秒だった。
速度差は小さく、メモリ使用量の少ないBS1を選ぶ。10k更新の計算部分は約5.66時間と見積もる。
現在は全validationの時間計測を実行中で、保存・入力検証・validation込みの時間は未確定。
本学習はまだ開始しておらず、所要時間・収束・精度の改善を確定していない。
終端checkpointで比較し、testでcheckpointを選ばない。
TensorBoardはこの更新数ベースのrunnerでは出しておらず、監査は更新ごとのtrain.jsonlとcheckpointを使う。
Driveの既存mountをrclone aliasで明示接続し、既存RcloneSnapshotStoreで世代別保存と復元を確認した。
小さな保存世代はGoogle Drive connectorでも実体を確認した。Driveの入力や過去世代は削除していない。
