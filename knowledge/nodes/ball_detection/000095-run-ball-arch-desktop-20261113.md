---
id: run-ball-arch-desktop-20261113
type: run
task: ball_detection
sequence: 95
recorded_at: '2026-11-13'
title: RTX 5060 Tiへの本学習移管とactivation checkpointing検証
issue: 1859
provider: codex
date: '2026-11-13'
status: running
summary_impact: none
config:
  seed: 42
  gpu: NVIDIA GeForce RTX 5060 Ti
  torch: 2.14.1+cu130
  python: 3.13.16
  allocator_fraction: 0.8
  msda_levels: [16, 32, 64]
  msda_chunk_size: 4
  frames: 64
  compile_mode: default
metrics:
  official_msda_gradcheck_widths_passed: 7
  full_shape_query_parameters_compared: 242
  full_shape_cnn_parameters_compared: 133
  maximum_query_absolute_gradient_error: 1.1920928955078125e-07
  maximum_cnn_absolute_gradient_error: 0.0
  probe_t1_peak_allocated_bytes: 10803609600
  probe_t1_peak_reserved_bytes: 12595494912
  probe_t1_warm_median_update_seconds: 3.7699728485004016
  probe_t2_peak_allocated_bytes: 10803609600
  probe_t2_peak_reserved_bytes: 12532580352
  probe_t2_warm_median_update_seconds: 3.16164077600024
  probe_c1_peak_allocated_bytes: 9396956160
  probe_c1_peak_reserved_bytes: 10475274240
  probe_c1_warm_median_update_seconds: 1.805800879999879
  t1_query12000_val_observed: 85473
  t1_query12000_val_mean_error_source_px: 35.33672768363863
  t1_query12000_val_median_error_source_px: 8.93928050994873
  t1_query12000_val_p95_error_source_px: 176.73656311035154
  t1_query12000_val_within20px_rate: 0.7419068009780867
artifacts:
  run_dir: knowledge/runs/run-ball-arch-desktop-20261113
  output_dir: outputs/ball_detection/train/arch_efficiency/desktop-recovery-20261113-a4
  drive_outputs: outputs/ball_detection/train/arch_efficiency/s42-20261111/desktop-attempts/20261113-a1
repro:
  commit: 15cdc684ff60a210cb1d8209f393e60490ca8f44
  branch: codex/ball-arch-desktop-20261113
parents: []
relations: []
papers: []
tags: [checkpointing, desktop-resume, msda]
---

# 移管の範囲と観測

#1859のT1後半5,500更新（query累積11,500）から、T1残り→T2→C1→C2→C3を再開する。元のL4実験・Drive世代は読み取り専用とし、デスクトップの保存先を分けた。T1/T2だけでactivation checkpointingを使い、同じ範囲を本学習前に固定する。T・解像度・batch・microbatch・dtype・MSDA level/chunk・compile方針とC1〜C3のrecipeは保持する。別GPUなのでbitwise継続を保証せず、このGPUの時間・VRAMを旧L4未学習ベンチマークと比較しない。

元の固定snapshot2,925 payload・157,028,755,547 bytesを同じstoreからhardlinkし、distributionと全件SHAを照合した。元のDrive実験1,917ファイル・19,243,387,824 bytesはE保管領域へ期待manifest付きで取得し、独立readbackと更新停止を確認した。Eは学習データ・tensorの入出力には使わない。

専用Python3.13.16 / Torch2.14.1+cu130と元の167 runtime package版を一致させた。元環境自身の依存metadata conflictは記録して明示的に同じ版を復元し、共有.venvを変更していない。sm_120向け公式MSDAの7 CUDA/C++ソースは既存の公式照合済みmanifestと同一。無制限register版はdouble backwardの1,024-thread blockでregister上限を超えたが、ソースを変えずcompilerのmaxrregcount=64で公式全7幅gradcheckが通った。

BF16実形状T64/720pのquery・CNNをchain-rule VJPで比較し、全375パラメータの勾配を照合した。query最大絶対差1.192e-7、CNN差0。全体非checkpoint参照が16GBに収まらないため、queryのfeature VJPを各非checkpoint CNN stageへ適用した診断であり、本番にはdetachを入れていない。native CUDA5テストも成功した。FP32のcompiled replayは出力・全勾配一致を確認したが、BF16 compiled graph同士はfusionが異なり要素ごとの勾配一致を主張しない。出力の近似一致・全勾配有限と、別のeager BF16等価性を記録している。

# 実データprobeと容量

MSDAのみ再計算する候補はT1/T2とも80% allocator上限でOOM。CNN各stageとMSDA chunkを再計算する`encoder_msda`では、T1の5,500 full stateから3更新、CNN非凍結のT2で3更新が有限に進んだ。両者のallocated最大10.80GB、reserved最大はT1 12.60GB / T2 12.53GB。C1はcheckpointingなしで3更新、allocated9.40GB / reserved10.48GBだった。T1/T2の範囲を同一に固定し、C2/C3はrecipeを変えない。probe更新は科学上の予算に数えない。各初回はcompileを含み、後続2更新だけの中央値はT1 3.77秒、T2 3.16秒、C1 1.81秒である。短い資格確認であり、一般的な速度推定やL4との比には使わない。

最初の本番attemptは初期5,500 checkpointのreadback中、filesystem増分19.20GBで19GBガードに止まり、新規optimizer更新はなかった。24GB host RAMのavailableは約19.8GBで、RAM不足ではない。中断した環境downloadの一時3 directoryを、使用中descriptorなし・共有hardlinkなしと確認して削除し、1.54GBを回収した。失敗と終了eventを残し、5,500 full stateを再保存・独立照合してから新しいattemptへ移す。compiler cacheも再利用して重複を避ける。詳細はbundleの`attempt-1-storage-recovery.json`。この時点では20GBの容量制約を維持した。後述の親の復旧判断でD:実空きへ置き換えた。

# 起動の受入と引継ぎ

attempt 2のT1は5,500から再開し、5,501〜5,511の11更新（compile初回の後に10更新）でloss・全勾配norm・LRが有限、sample位置21,501〜21,511が連続していた。全model/AdamW/RNG/samplerの復元が実runnerで一致し、新規state keyはない。最初の3更新のdata drawとaugmentationは隔離probeと一致した。allocated最大10,803,609,600 / reserved最大12,595,494,912 bytesはallocator上限13,682,291,507以内。host guardも下限内で発火していない。詳細な数値と実ログの受入prefixは`startup-acceptance.json`。

attempt 2はpush済み`cb3243ed4`、systemd user serviceと共有queue resource=allで継続したが、6,000保存時に停止した。親の監視契約・実PID/PGID・terminal event・各段のstate/log/receipt・正確な復旧コマンドを`monitoring-handoff.json`へ残した。workerはT1の12k評価やT2の節目を待たず、定期follow-upを登録しない。C2/C3の実GPU peakとT1以外の条件の精度はまだ未確認。

# 判断と残る評価

このノードは移管・数値・メモリ資格確認の記録で、精度改善の根拠にはしない。モデル採用、独立test未評価、#1859の5条件完走後に精度を比較するというsummaryの判断は変わらない（summary_impact: none）。実データprobe・起動確認の結果は同run bundleへ保存する。学習曲線とper-frame予測はDriveに保持し、Gitには小さな受入要約・再現手順を残す。T1の12k評価、T2初回500保存と600解除は検証済みとなり、残条件の完了を親が監視する。workerの終了を本学習完了とは扱わない。

# 6,000保存停止からの復旧

attempt 2はT1後半6,000更新と固定validationを終えたが、checkpoint独立readbackの一時展開で旧19GB filesystem増分ガードに止まった。RAMは下限内だった。親の2026-11-13の継続指示で、容量停止を`/mnt/d`のstatvfs availableが10GiB未満へ変更し、RAM4GiBとallocator80%は保持した。WSL増分はtelemetryだけにする。guardとadapter計23件、修正後のCI計測plugin付きguard10件、実行commit `2cb49a907` のCI Pythonが成功した。

Driveの`checkpoint-00006000-8586c177`をmarker SHA固定で独立復元し、全11ファイル・550,906,644 bytesとfull model/AdamW/RNG/LR/sampler/log・weights artifactを停止時の状態と完全照合した。厳格resumeもupdate6,000・sample22,000を受け入れ、pointerをpublish/readbackした。新rootでも固定2,925 payloadを全件SHA照合し、src差分なし・新規state keyなしで復旧した。T1更新や評価は繰り返さない。source画像座標の12k固定validationはobserved85,473、mean35.3367px、median8.9393px、P95 176.7366px、≤20px率74.1907%。reportとper-frame予測は同じ検証済み世代にある。精度改善やscene採用の結論にはしない。

旧attemptの重複tensorとreadback cacheだけをexactなDrive世代・SHA・削除前HEAD・公開ledger付きで整理し、WSL allocated blocksを1,476,173,824 bytes解放した。専用env/native build/compile cacheを保持し、他issueには触れていない。詳しい再開点の根拠は`attempt-2-storage-recovery.json`、整理記録は`attempt-3-local-cleanup.json`。実行は新しいsystemd unit/queue attempt 3とし、監視の正本は更新した`monitoring-handoff.json`とする。

attempt 3ではT1をupdate6,000・sample22,000で実runnerへ復元し、追加更新なしで完了snapshotとvalidation/per-frame予測を保存・独立照合した。続くT2の1〜11更新・sample10,001〜10,011が連続し、loss/gradient norm/LRは有限だった。compile初回後10更新の中央値1.8348秒、allocated最大7,147,210,752 / reserved最大9,737,076,736 bytes。CNNはこの歴史的prefixでは凍結中。host available最小17,030,041,600 / D: available最小33,458,446,336 bytesは当時の下限内だった。根拠は`attempt-3-startup-acceptance.json`に保持する。

# RAM待機への変更とattempt 4

attempt 3はT2 backwardの1,283更新付近でhost availableが旧4GiB下限を下回り停止した。数値は有限・連続で、Drive上の検証済み最新世代は1,000更新だった。並行pytestやduは親が観測したが、当時は学習プロセスのRSS履歴がなく、原因を外部要因だけに帰属できない。

親の復旧指示で、次の更新前にavailableが6GiB未満なら無期限に待ち、6GiB以上で再開する。30分超の通知は1回だけ残し、更新途中を含む2GiB未満は緊急停止する。D:10GiBとallocator80%、checkpointing範囲・recipeは変えない。待機時間を更新秒から除外し、sampler/RNGを進めない。各更新に学習親と再帰的な子のRSS合計を記録し、待機開始・再開・30分超・中断を`ram-waits.jsonl`へ残す。RSSには共有pageの重複を含む。guard/adapter/実model比較36件が20.66秒で成功し、6GiBを4GiBへ下げる誤りと待機を計測内へ移す誤りをmutationで検出した。実行commitのCI Pythonも成功した。

attempt 4は検証済み`checkpoint-00001000-e2f35d3f`を独立readbackし、model381 key・AdamW375 state・RNG・LR位置・sample11,000・1,000行logを完全復元した。既存state keyの追加はない。明示許可したrunner observerのsource差分だけをbyte固定で移管し、未commitの1,001〜1,283更新は採用していない。systemd user serviceと共有queue resource=allで、push済みsourceから1,001〜1,011更新・sample11,001〜11,011が有限・連続に進んだ。data draw/augmentation/LRは同じattempt-3 prefixと一致した。CNN解除済みでallocated最大10,803,609,600 / reserved最大12,763,267,072 bytes、compile初回37.7527秒・後続10更新中央値3.4532秒だった。host available最小8,412,168,192、admission最小8,616,722,432、D:最小16,144,359,424 bytesは各下限内。親子RSS合計の最大は13,527,920,640 bytes。受入prefixではRAM待機は発生していない。

現在の実行・保存先・unit/queue/PID/PGID・終了event・待機logのキーと新attempt復旧コマンドは`monitoring-handoff.json`、受入根拠は`attempt-4-startup-acceptance.json`。実行checkoutは固定し、報告commitを別に残す。T2完了や以降の条件は待たず、監視を親へ返す。定期follow-upは登録しない。
