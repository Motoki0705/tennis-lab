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
status: done
config:
  architecture: hybrid
  frames: 32
  native_image_size:
  - 1280
  - 720
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
  second_data_download_and_cli_verification_seconds: 2890.1638309560003
  resumed_updates: 12
  max_absolute_loss_difference: 8.972599468332933e-07
  cpu_cuda_rng_equal: true
  final_model_state_bit_equal: false
  estimated_compute_units: 5.286776771344444
artifacts:
  run_dir: knowledge/runs/run-ball-colab-checkpoint-resume-20261024
parents:
- run-ball-colab-persistence-preflight-20261023
- run-ball-portable-distribution-20261023
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
repro:
  commit: 768322617a9e3eb6c354bca834e90820ed0b6760
---

720p・本番モデル幅のhybrid/T=32を12更新する診断で、checkpoint直後のDrive保存、5更新での中断、
VM停止、新VMへの同run復元、残り7更新の完了を確認した。固定配置で既存のpathを含むidentity照合が通り、
内容hashを理由にpath差を無視する入口は追加していない。

[全12更新の照合](../../runs/run-ball-colab-checkpoint-resume-20261024/receipts/comparison-b.json)は成功した。
pack ID/index・教師数・累積入力・LR列が一致し、更新数とサンプル位置は12、全optimizer stepも12で、
元の12更新予算とwarmup/cosine scheduleを保持した。CPU/CUDA RNGも一致した。
損失の最大絶対差は8.98e-7で、実行前に決めた絶対1e-6・相対1e-4の許容差内だった。
[中断前5行のbyte一致](../../runs/run-ball-colab-checkpoint-resume-20261024/receipts/prefix-comparison-b.json)も確認した。
最終重みのhashは対照と異なる。同じseed・条件でも中断前5更新から微差が観測されており、原因の切り分けはしていない。
この診断は損失の許容差を使う連続性の証拠であり、GPU演算のbit一致を証明しない。
CPUでは実モデルの重み・optimizer・schedule・RNG・サンプル位置の一致もintegrationで確認した。

汎用の`RcloneSnapshotStore`は世代を上書きせず、全fileの集合・サイズ・MD5を照合してからmarkerを公開し、
その再照合後にだけ保存receiptを返す。復元はreceiptに固定したmarker SHA-256と各fileのSHA-256を照合し、
検証後にrun treeを公開する。常設runnerには同期の`persist(output, stage)` callbackを接続した。
実験controllerと環境再構築・順序制御のsourceは[run bundle](../../runs/run-ball-colab-checkpoint-resume-20261024/README.md)に置いた。
完成checkpoint・weights・latest/run情報・両manifest・学習ログを世代ごとに保存し、過去世代を削除しない。
各世代は現在のcheckpointを明示選択し、古いcheckpointを毎回再転送する増大を避ける。

全train/validation配布storeを両VMで新規取得し、1,537 clip・2,926 file・157.03 GBの全件SHA-256と18窓の意味一致を確認した。
最初のVMで配布store用の全フレームmanifestを再生成し、先行の共通793,017 frame・39,185 packと5種のidentityが一致した。
元snapshot用manifestの流用や既存資産の書換え・hardlinkはしていない。
短い更新・評価には、そこから独立に作る4 clipの診断storeを用いた。trainはTrackNet・Meiji・Chat、
validationは位置正例と確認済み不在を持つChatで、実際のtrain列にも3供給源が現れた。
すべてnative 1280×720で、幅・深さ・5ch・全frame教師・128入力frame/更新を維持し、chunk=4とactivation checkpointingを明示した。
この精度・時間から本学習の収束、長系列上限、速度や予算は決めない。

学習sourceは両VMで同じcommitに固定し、Git bundleの内容hash・prerequisite・Drive IDを保存した。
新VMの既定setupはlockfile内の主要packageが一致してもPython 3.11.17を選んだため、データ取得前に拒否した。
uv 0.12.21・Python 3.11.16のビルド文字列とlockfileを固定して再構築し、主要依存とTorch/CUDA/cuDNNを記録した。
最初の環境構築commit・学習commit・環境helper追加commitのlockfileが同一であることも確認した。
固定配置の具体的path、正確な引数、復元markerと各fileのDrive IDはrun bundleのreceiptが正本である。

| 実測区間 | VM A | VM B |
| --- | ---: | ---: |
| 成功した環境構築 | 41.91秒 | 28.94秒 |
| 全量取得＋CLI照合 | 54.83分 | 48.17分 |
| 独立providerでの全件SHA-256 | 7.37分 | 7.73分 |
| 18窓の意味照合 | 18.14秒 | 18.14秒 |
| 全配布版のmanifest準備 | 22.50分 | 保存済みmanifestを復元 |
| 診断store/manifest準備 | 4.05秒 | 入力一式を復元 |
| 入力一式122.17 MBの保存/復元 | 保存48.74秒 | 復元30.91秒 |
| 中断run一式174.51 MBの保存/復元 | 保存45.02秒 | 復元33.37秒 |
| 学習・評価・同期保存を含むrun時間 | 対照302.99秒／中断159.21秒 | 再開266.20秒 |

checkpoint直後の保存・照合はAで40.50〜44.89秒、Bで49.88〜51.57秒だった。
これらは短い保存復旧診断の値で、長系列の速度・メモリ計測ではない。
成功した環境構築時間には、Python版の不一致、installerのHTTP 403、配置変数の優先順位による失敗、調査の待ち時間を含めない。
各失敗jobと時間はsession receiptに残し、session全体の時間・消費推定には含めた。

通信断注入runは実際の1更新後の保存で失敗した。jobはreturncode=1で終了し、完了markerは無く、
確認済みDrive世代は初期状態だけだった。失敗jobのログ・receiptはworkerが別途保存した。
新VMのデータ検証後にはNotebook接続断で最初の復元要求が未実行のpendingに残った。
元要求をcancelしてDrive保存を確認し、同じboot IDのVMへ再接続して別job IDで復元した。データは再取得していない。
これらの失敗を学習成功や復元成功へ読み替えていない。

[VM A](../../runs/run-ball-colab-checkpoint-resume-20261024/receipts/session-a.json)を停止してから
[VM B](../../runs/run-ball-colab-checkpoint-resume-20261024/receipts/session-b.json)を作成し、最後にBも停止した。
両sessionの全jobの終了・Drive保存と、各sessionの停止を確認した。所有者不明の既存割当は操作していない。
全体の割当0件は完了条件にしていない。
[compute unit記録](../../runs/run-ball-colab-checkpoint-resume-20261024/receipts/compute-units.json)の約5.29 CUは、
所有sessionの合計wall timeとB単独時の観測rate 1.54 CU/時からの推定で、providerのsession別請求値ではない。
アカウント残高差には他の割当の活動が混ざるため、本実験へ全額帰属させない。

最終sourceの保存・学習integrationは共有ロック・`-n 4`・CPU限定で41件成功した。変更Pythonのruff/mypy、
knowledgeの構造と再現用pathも検査した。先行の注釈path阻害は解消済みで、このcheckoutの移動/破損6テストも成功した。
次は本番データで長系列の速度・VRAM・保存間隔と総時間を測り、合意した予算を固定する。
採用checkpointや精度の判断は今回の保存復旧診断から変更しない。
