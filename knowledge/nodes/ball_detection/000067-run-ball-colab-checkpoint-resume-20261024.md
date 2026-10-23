---
id: run-ball-colab-checkpoint-resume-20261024
type: run
task: ball_detection
sequence: 67
recorded_at: '2026-10-24'
title: 720p Colab学習のcheckpoint逐次保存と新VM再開の検証
issue: 1375
provider: codex
date: '2026-10-24'
status: running
config:
  architecture: hybrid
  frames: 32
  native_image_size: [1280, 720]
  frame_chunk_size: 4
  activation_checkpointing: true
  updates: 12
  interrupted_after_update: 5
  input_frames_per_update: 128
  seed: 42
metrics:
  first_environment_setup_seconds: 41.91178623799999
  pinned_second_environment_setup_seconds: 28.940033822000032
  first_data_download_and_cli_verification_seconds: 3289.7598790270004
  first_full_manifest_preparation_seconds: 1350.1026073379999
  control_updates: 12
  interrupted_updates: 5
artifacts:
  run_dir: knowledge/runs/run-ball-colab-checkpoint-resume-20261024
parents:
- run-ball-colab-persistence-preflight-20261023
- run-ball-portable-distribution-20261023
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
---

720p・本番モデル幅の短い学習で、checkpoint完成直後のDrive保存と、新VMへの同run復元を検証する。
先行の注釈path問題は解消し、移動済み配布版と破損注釈を含む6テストをこのcheckoutでも確認した。
最初のL4 VMではPython 3.11・`uv sync --locked --no-cache`の環境構築が41.91秒で成功した。
全train/validation配布storeの新規取得・照合後に対照runと中断runを実行し、このVMを停止した。
既存の他者所有の可能性がある割当は操作していない。

実験controllerはこのrunの`resume_probe.py`に置き、常設runnerには同期保存callbackだけを接続した。
汎用の`RcloneSnapshotStore`はrun treeを世代ごとに保存し、全fileのサイズとMD5、markerの内容を照合して
receiptを返す。復元はreceiptに固定したmarkerのSHA-256、各fileのSHA-256、相対pathと集合を照合する。
保存失敗を成功へ読み替えず、同名世代や部分世代を上書きしない。
CPUでは実モデルの中断・snapshot復元・resumeと、通信断後に更新が進まないことを確認した。

主比較や本学習の予算決定は行わない。全量storeの取得・検証とT=32 manifestの再生成を行い、
保存復旧だけの短い更新・評価にはそこから作る独立した4 clipの診断storeを使う。
trainはTrackNet・Meiji・Chat、validationは位置正例と確認済み不在を持つChatを選ぶ。
すべて1280×720のnative画像とし、幅・深さ・5ch・全frame教師・128frame/更新を維持する。
12更新・5更新目で中断、損失許容差は絶対1e-6・相対1e-4を実行前に固定した。
この診断の精度や時間を本学習の収束・予算・長系列上限の根拠には使わない。

最初のVMでは全1,537 clip・2,926 file・157.03 GBを新規取得し、CLI照合込み54.83分、
独立providerの全件SHA-256照合7.37分、18窓の意味照合18.14秒で成功した。
配布storeに対するT=32 train/evaluation manifestの全件検証・生成は22.50分かかった。
先行runの共通区間793,017 frame・39,185 pack・5種のidentity hashとの一致も確認した。
入力一式の17 file・122.17 MBは48.74秒でDrive保存とサイズ/hash照合を完了した。
正確な値とmarkerのDrive IDは[receipt群](../../runs/run-ball-colab-checkpoint-resume-20261024/README.md)に残す。

対照runは12更新、中断runは元の12更新予算を保持して5更新で停止した。
前半5更新はpack・教師数・LRが一致し、損失の最大絶対差は8.98e-7で事前許容差内だった。
同一seedでもGPU損失のbit一致は観測しておらず、最終重みのbit一致を受入条件へ追加しない。
checkpoint世代は約174.5 MBで、対照runのcheckpoint直後の保存・照合は40.50〜44.89秒だった。
通信断注入runは実際の1更新後の保存で失敗し、jobのreturncodeは1、完了markerは無く、
最後に確認できるDrive世代は初期状態だけだった。失敗job自身のログ・receiptはworkerが別途保存した。

最初のVMの停止後に新しいL4 VMを作成し、boot IDが異なることを確認した。
新VMの既定setupはlockfile内のpackageが一致してもPython 3.11.17を選び、対照の3.11.16と異なった。
データ取得・再開前に止め、uv 0.12.21・Python 3.11.16のビルド文字列と主要依存の一致を確認してから進めた。
版を固定した環境再構築は28.94秒だった。installerのHTTP 403とColabの配置変数優先順位による
2件の失敗も保存しており、成功した試行の時間へ含めていない。
両学習armは同じcommitを使い、環境helper追加後もその学習commitへ明示的にcheckoutし直す。
新VMでの全量取得・復元・残り7更新・全12更新の照合は実行中である。
