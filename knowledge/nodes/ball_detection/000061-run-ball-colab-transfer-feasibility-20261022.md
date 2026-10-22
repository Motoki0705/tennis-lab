---
id: run-ball-colab-transfer-feasibility-20261022
type: run
task: ball_detection
issue: 1367
sequence: 61
recorded_at: '2026-10-22'
title: ボール検出データのDrive・Colab転送実測とアブレーション実行先の検討
provider: codex
date: '2026-10-22'
status: done
config:
  snapshot: mdd_hybrid_1231/20261018-mix1800-720p-v1
  repeats: 3
  files_per_repeat: 6
  colab_gpu: NVIDIA L4
metrics:
  snapshot_bytes: 185522383458
  transferred_sample_bytes: 1781103723
  verified_sample_files: 18
  upload_end_to_end_MB_per_second_min: 6.032148995096295
  upload_end_to_end_MB_per_second_max: 6.466280575837703
  download_end_to_end_MB_per_second_min: 16.195597028321565
  download_end_to_end_MB_per_second_max: 18.425611667518496
  colab_initial_free_bytes: 206567641088
repro:
  commit: a6812b3a88333c578d7fc0b03017d9b14caf406d
  command: '実行argvは同run bundleのupload-*.timing.jsonとdownload-*.timing.json。遠隔実行本文はcolab_download_code.py。'
artifacts:
  run_dir: knowledge/runs/run-ball-colab-transfer-feasibility-20261022
  output_dir: outputs/ball_detection/transfer_probe/20261022
parents: []
relations: []
papers: []
tags: []
---

## 結論

ローカル→Drive→Colabの転送は、直近の720pボール検出snapshotから選んだ18 shard・計1.781 GBで成立した。
3回ともDrive CLIの内容照合を通過し、Colabで全18ファイルのSHA-256・サイズを固定manifestと照合した。
全185.522 GBは送っていない。初回アップロードは時間単位の負担であり、条件間で同じ入力版を再利用する価値がある。
その後の同日のユーザー指定で、後続実験はColabでの本学習完了を目指す方針になった。
実行計画・合意・完了条件の正本は[親issue #1367](https://github.com/Motoki0705/tennis-lab/issues/1367)。
学習速度・精度・長時間再開を今回検証したとは扱わない。

ユーザーは[Courtのデータ規模計画 #1343](https://github.com/Motoki0705/tennis-lab/issues/1343)を例に、
ボール検出のアブレーション軸選定と同時に、巨大データの転送時間を実測するよう依頼した。
このrunは転送可否の実測を記録する。Court計画の実行先・モデル・データ条件をボール検出へ自動適用しない。

## 対象と測り方

対象は `outputs/ball_detection/precompute/mdd_hybrid_1231/20261018-mix1800-720p-v1/ball_snapshot`。
全1,800 clip、1,065,949 frame、JPEG連結shard 1,800個とindex/metadataで185,522,383,458 bytes。
原動画、他のデータ版、pose artifact、学習出力の重複を含めていない。
学習1,370 clipのshardは132.816 GB、validation 200 clipは24.268 GB、test 230 clipは28.373 GB。
index/metadataを加えたtrain・validationだけの物理量は157.150 GBだが、後述の現行読込契約を満たすには別途準備が要る。

TrackNet、Meiji、Chatの異なるサイズ帯から、互いに重ならない6 shardずつを3集合選んだ。
Chatは各集合で4サイズ帯を含めた。サンプルの選択条件、正確なpath・サイズ・hashは
[sample-inventory.json](../../runs/run-ball-colab-transfer-feasibility-20261022/sample-inventory.json)にある。
元のhardlinkのctimeを変えないよう通常コピーし、転送前に固定manifestの`rgb_shard_sha256`と照合した。
`media_sha256`はMeiji等では元動画のhashを表すため、JPEG shard照合には使用しない。
サンプル作成・事前hash照合は転送時間から除外した。

Drive skillの`upload/download --verify`を既定並列数4で実行し、CLIの前後をmonotonic時計で計測した。
この時間にはpath一意性検査、コピー、転送後の読戻し、共通内容hashの照合を含む。
別にrcloneの最終`Elapsed time`をコピー工程の時間として取得した。これは通信線だけの速度ではなく、
コピー内のファイル単位の処理も含む。Colabでの固定SHA-256照合はCLI完了後に別計時した。
ローカルはWSL2、Python 3.11.15、ColabはPython 3.13.15、双方rclone v1.60.1-DEV。
依存ライブラリの導入やGPU学習は実行していない。

## 実測

GB/MBは十進数。総時間はCLI全体、コピーは上記rclone工程。

| 回 | データ量 | upload総時間 / コピー | upload総時間の速度 | Colab取得総時間 / コピー | 取得総時間の速度 |
|---|---:|---:|---:|---:|---:|
| 1 | 544.69 MB | 90.30 / 65.9 秒 | 6.03 MB/s | 33.63 / 7.1 秒 | 16.20 MB/s |
| 2 | 586.33 MB | 94.34 / 68.4 秒 | 6.21 MB/s | 34.39 / 8.2 秒 | 17.05 MB/s |
| 3 | 650.09 MB | 100.54 / 73.1 秒 | 6.47 MB/s | 35.28 / 8.2 秒 | 18.43 MB/s |

Colabでの追加SHA-256照合は各1.60 / 1.72 / 1.91秒。
詳細値、全18照合結果、推定式の入力は[summary.json](../../runs/run-ball-colab-transfer-feasibility-20261022/summary.json)。
個々のCLI receiptとrcloneログも同じbundleに保存した。
公開用のパス・識別子処理は同bundleの[README](../../runs/run-ball-colab-transfer-feasibility-20261022/README.md)を参照。

## 全量への外挿と限界

| 工程 | コピー工程からの外挿 | 小分けCLI総時間を容量に比例させた外挿 |
|---|---:|---:|
| ローカル→Drive 185.522 GB | 5.79〜6.23 時間 | 7.97〜8.54 時間 |
| Drive→Colab 185.522 GB | 0.65〜0.72 時間 | 2.80〜3.18 時間 |

**両列とも全量実測ではなく、同日の3サンプルからの見積もり。**
CLI総時間には1コマンドあたり約24〜27秒のコピー外処理がある。小分け総時間の比例はこの固定的な処理まで
全量に繰り返すため、特にColab取得を過大に見積もり得る。右列を大型一括転送の実測値や信頼区間とは呼ばない。
左列には全量の追加hash検証・移植・ライブラリ導入を含まない。今回のSHA-256走査速度だけなら全量で約9.1分だが、
ページキャッシュ、ディスク、ファイル数、API制限、長時間の回線変動は未検証。
初回準備の計画枠は時間単位で取り、固定費の償却を含めて比較する。小さな実験を1回行うだけならローカルの利点が大きい。

同じ入力版をDriveに保持すればローカルからの全量uploadは毎条件には不要。ただし新しいVMには取得と検証が必要。
Colabの同時実行数や実学習の高速化は今回未測定であり、L4という名前だけから速度向上を仮定しない。

## 容量と可搬性

- Driveアカウントの空きは確認時4.755 TBで、容量自体は今回の入力を妨げなかった。
- 今回のL4 VMは総253.055 GB、初期空き206.568 GB。全snapshot配置後は計算上約21.045 GB残る。
  task環境・compile cache・全epoch checkpoint・出力を含めた本学習時の余裕は未確認。
  参考として現在のローカル`.venv`は6.139 GBだが、Colabの導入量と同じとは限らない。
- `BallFrameStore._validate`は宣言した全clipのshard存在・長さを検査する
  （[store.py](../../../src/tasks/ball_detection/data/store.py)）。testファイルだけ省いた転送では現行storeを開けない。
  157.150 GB案は、split・教師・RGBを保持した独立train/validation storeを明示生成し、manifestを対応付ける工程が前提。
- 固定manifestのball-store参照はローカル絶対path。`reviewed_empty_evidence`はChatの原注釈JSONを絶対pathから開き、
  hashとnotesを検査する（[hybrid.py](../../../src/tasks/ball_detection/data/windows/hybrid.py)）。
  Chat注釈全1,534ファイルは230,502,990 bytes。原動画を運ぶ必要はないが、必要な注釈とpath対応は準備する。
  pathやmetadata hashを黙って読み替えず、移植版と元版の意味的同一性を照合する。
- 1集合544.687 MBのzlib level 1追加圧縮は542.315 MB、削減0.4355%、処理16.63秒だった
  （[compression.json](../../runs/run-ball-colab-transfer-feasibility-20261022/compression.json)）。
  この方法でJPEG payloadの転送負担を大幅に減らす根拠はない。他codec一般の限界や、再圧縮後の精度は検証していない。

## 保存と終了

計測用18 shardはDriveの`staging/ball-detection-transfer-probe-20261022/sample-{1,2,3}`に保持した。
内容hashと相対pathは[drive-inventory.json](../../runs/run-ball-colab-transfer-feasibility-20261022/drive-inventory.json)。
曖昧な同名pathは0件。ローカル側の全文inventoryはoutput_dirに残し、そのhashをsample-inventoryに記録した。

計測用Colab sessionのcapacityとdownload-benchmarkは終了値0、Drive保存完了を確認。
遠隔の小さな計測結果は`outputs/ball_detection/transfer_probe/20261022-colab`から内容照合付きで回収した。
最後にsessionを停止し、[colab-stop.json](../../runs/run-ball-colab-transfer-feasibility-20261022/colab-stop.json)の
`status: stopped`と未回収sourceなしを確認した。学習・既定checkpointの変更・全量転送はこのrunの成果に含めない。

## 後続実験への引継ぎ

2026-10-22に[親issue #1367](https://github.com/Motoki0705/tennis-lab/issues/1367)を作成した。
ユーザー指定の主軸は時系列長 × DPT／Transformer＋DPT（hybrid）。
RGB＋MDDの5ch、共通出力・損失、両モデル32フレームで同量の事前学習を固定する。
最新の指定は、転送・学習・保存再開を検証した上で、最終的にColabで本学習を行うもの。

実験行列・予算・教師領域・評価・子工程・未決事項は親issueで管理し、本ノードへ二重管理しない。
以下には今回確認したコード上の根拠とデータ計数を残す。転送runの完了は学習実験の完了を意味しない。
親issue起票時点で本学習は未開始、子issueは未分割。

## アーキテクチャと可変長について確認したこと

- 現行DPTは共通CNNの4解像度特徴をフレーム単位のDPTに渡す。hybridは同じCNNから各時刻のqueryを読み、
  時間Self-Attention後のqueryを最下層へconcatしてDPTへ渡す。
- 共通CNNの時間融合は幅3の畳み込み2層で、特徴段階の最大受容範囲は5時刻。
  MDDの隣接差分を含む内部フレームの元RGB依存範囲は最大6時刻。正規化とDPTはフレーム単位であり、
  Tを伸ばしても局所範囲自体は伸びない。hybridの時間Attentionは全時刻のqueryを参照できる。
  これはコードの構造からの推論であり、精度改善の実測ではない。
- 全時刻Self-AttentionにはTの二乗に比例する演算項がある
  （[Attention Is All You Need, Table 1](https://arxiv.org/html/1706.03762v7#S4)）。
  この実装では各時刻1 queryを時間方向で処理する。共通CNN・DPT・SDPAを含む実速度やVRAMは未測定。
- `MDDPretrainConfig.frames`、query設定、5ch入力契約、window manifest/dataset、
  annotation window policy、model I/Oと評価には32固定がある。query/RoPEのparameter形状に
  T固定の学習tableがなくても、既存CLIが任意Tに対応するわけではない。
- 現行hybridの初期化はquery/continual、損失はheatmap＋presence。DPTとの等予算の直接初期学習、
  共通heatmap objective、中央教師mask、32→Tの全重み転送は対応が必要。
  既存のDPT→query→hybridの累積予算を、DPT単独と同じ事前学習量とみなせない。

根拠は `models/mdd/{cnn_encoder,blocks,detectors,attention,hybrid,config}.py`、
`preprocessing/{mdd,rgb_mdd}.py`、`training/dpt_pretrain/objective.py`、
`training/hybrid/{configuration,objective}.py`、`data/windows/`。
調査対象の版は下記CPU計数artifactの `inspected_code_commit` に記録した。

## 長い窓を取れるデータのCPU調査

現行stage1の固定manifestにある32フレーム窓を中心に、同一play区間内へ対称に文脈を広げられるかを数えた。
画像の復号・GPU計算・新しい長窓manifestの生成は行っていない。
[temporal-support-probe.json](../../runs/run-ball-colab-transfer-feasibility-20261022/temporal-support-probe.json)
に入力manifest、判定式、split・間引き別の計数を保存した。

| 入力T | 間引きなしで対応できるtrain clip | 元動画group | 現行32窓のうち文脈を確保できる数 |
|---|---:|---:|---:|
| 32 | 1,324 | 254 | 31,868 |
| 64 | 1,306 | 254 | 26,775 |
| 128 | 1,213 | 253 | 20,768 |
| 256 | 855 | 227 | 11,678 |

これは長い窓全体の教師適格性や中央教師領域を検査した最終採択数ではなく、不在だけの窓も含めない。
長さごとにこの別々の集合を使うと、文脈だけでなくデータ構成も変わる。
共通の対象フレーム集合を固定し、source/group・不在・遮蔽のcoverageを調べる必要がある。
長い区間に限定した評価を、元の全データ上の精度と混同しない。

現行frame_step=1/2/4のうち、T=256・step=4のtrain候補は0。
間引きを維持できる範囲、実PTSによる実時間幅、中央教師領域の適格性は本学習前に確定する。
この調査はColabの学習速度・長系列のGPU実行可否・精度・deploy判断を検証したものではない。
