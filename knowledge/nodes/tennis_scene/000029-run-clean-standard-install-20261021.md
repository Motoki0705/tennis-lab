---
id: run-clean-standard-install-20261021
type: run
task: tennis_scene
sequence: 29
recorded_at: '2026-10-21'
date: '2026-10-21'
title: 公開標準pipelineのクリーン導入・GPU完走と公式CLIP-ReID比較
issue: 1274
provider: codex
status: done
config:
  device_used: cuda
  gpu_executed: true
  gpu: NVIDIA GeForce RTX 5060 Ti
  gpu_vram_mib: 16311
  cuda_toolkit: 13.0.88
  cuda_arch: '12.0'
metrics:
  clip_reid_embedding_max_abs_diff: 0.0
  clip_reid_visual_keys: 152
  missing_preflight_failures: 28
  cpu_tests_passed: 56
  cpu_preparation_seconds: 761.4679381140013
  cuda_build_seconds: 154.31974403099593
  preflight_checks_passed: 28
  disk_apparent_bytes: 19779048171
  disk_allocated_bytes: 19940069376
  pipeline_to_save_seconds: 1827.129
  visualization_process_seconds: 363
  visualization_logged_load_to_save_seconds: 355.09
  vram_sampled_device_max_mib: 4928
  vram_baseline_mib: 1215
  vram_sample_count: 368
  vram_sample_span_seconds: 1806
  scene_schema_version: 2
  frames: 1010
  width: 1920
  height: 1080
  fps: 59.94006
  ball_3d_frames: 822
  player_joint_frames: [1008, 993]
  player_root_frames: [984, 993]
  player_smpl_frames: [984, 993]
  completed_disk_apparent_bytes: 20703572320
  completed_disk_allocated_bytes: 20866301952
repro:
  commit: 261c2186134bbe09e11b757e55aa1cb70bd02335
  branch: codex/install-guide
artifacts:
  run_dir: knowledge/runs/run-clean-standard-install-20261021
  clip_reid_comparison: knowledge/runs/run-clean-standard-install-20261021/clip_reid_comparison.json
  missing_preflight: knowledge/runs/run-clean-standard-install-20261021/missing_preflight.json
  local_asset_hashes: knowledge/runs/run-clean-standard-install-20261021/local_asset_hashes.json
  preparation: knowledge/runs/run-clean-standard-install-20261021/preparation.json
  build: knowledge/runs/run-clean-standard-install-20261021/build.json
  log: knowledge/runs/run-clean-standard-install-20261021/build.log
  preflight: knowledge/runs/run-clean-standard-install-20261021/preflight.json
  validation: knowledge/runs/run-clean-standard-install-20261021/validation.json
  execution: knowledge/runs/run-clean-standard-install-20261021/run.json
  gpu_validation: knowledge/runs/run-clean-standard-install-20261021/gpu_validation.json
  pipeline_log: knowledge/runs/run-clean-standard-install-20261021/pipeline.log
  visualization_log: knowledge/runs/run-clean-standard-install-20261021/visualization.log
  metadata: knowledge/runs/run-clean-standard-install-20261021/scene.metadata.json
  vram_samples: knowledge/runs/run-clean-standard-install-20261021/vram-samples.csv
parents:
- run-multiview-sample-standard-pipeline-20261021
relations: []
papers: []
tags:
- installation
- public_sample
- cpu_validation
- gpu
- visualization
---

## 範囲と受入の限界

#1274の導入ガイド・資産preflight・独立cloneによるCPU準備の後、
親セッション（Claude Code）がtraining queueでGPU推論を完走し、新sceneをCPUで可視化した。
この追記では既存のログ・scene・動画・VRAMサンプルをCPUで検査し、GPU処理は再実行していない。
同じホストの独立clone・新venvによる導入検証であり、新規OS・ドライバの構築や
独立した3D精度の評価ではない。[先行実行](000028-run-multiview-sample-standard-pipeline-20261021.md)との
有効フレーム数の差と、clone・資産取得・VRAM測定の限界を下記に分けて記録する。
導入の正本は[docs/installation](../../../docs/installation/README.md)。トップREADMEは今回変更していない。

## 公式CLIP-ReIDと従来使用版の比較

[公式READMEのViT-CLIP-ReID / Market / model](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/README.md#trained-models-and-test-logs)
が指す[Google Driveファイル](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view)を
`gdown`で取得した。ファイル名は`Market1501_clipreid_ViT-B-16_60.pth`。
比較相手はローカルで従来使用している`occurra/person_vit_clip_reid`由来の
`ckpt/player_association/person_vit_clip_reid.pth`で、今回HFから再取得したものではない。

両者は **507,391,033 bytes**、SHA-256
`bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc`で一致した。
両方を`ClipReIDEncoder`でCPUロードし、`image_encoder.`以下の全152キーの不足・過剰がないこと、
位置埋め込みが`(129,768)`（256×128入力、16px patch）であることを確認した。
seed 42の同一RGB乱数crop 4枚を入力し、`(4,1280)` embeddingが完全一致、最大絶対差0だった。
詳細は[comparison JSON](../../runs/run-clean-standard-install-20261021/clip_reid_comparison.json)、
再現スクリプトは[compare_clip_reid.py](../../runs/run-clean-standard-install-20261021/compare_clip_reid.py)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  knowledge/runs/run-clean-standard-install-20261021/compare_clip_reid.py \
  --official "$DOWNLOADED_OFFICIAL_MARKET" \
  --current "$LOCAL_CHECKPOINT_ROOT/player_association/person_vit_clip_reid.pth" \
  --output outputs/clip_reid_comparison.json
```

同一byteのファイルなので、取得先の変更による追跡・人物対応への入力値の差は生じない。
`person_tracking`の採用済みprofileと`player_association`の`appearance.slope=62.7`・`center=0.847`は
変更しない。公式Driveだけを導入先として案内する。これは品質改善の実験や新しいデータでの
追跡精度の測定ではなく、配布物のidentityと同一入力出力の比較である。人間の追加判断は不要だった。

## DINOv3単体ファイルの要否（調査のみ）

現在の推論は、`load_court_checkpoint()`が`build_court_inference_pair()`でモデルを構築した後、
全`model.` stateを`strict=True`でロードする
（[checkpoint.py](../../../src/tasks/court_detection/inference/checkpoint.py)の構築・ロード箇所）。
構築は[dinov3_dpt.py](../../../src/tasks/court_detection/models/dinov3_dpt.py)の
`CourtDINOv3Encoder` → `load_dinov3_backbone()`へ進み、
[共通loader](../../../src/utils/models/loading/dinov3.py)が単体ファイルの存在とstrict loadを要求する。
`torch.hub.load(..., source='local', pretrained=False)`自体は、事前学習重みを読まずに構築できる。

main treeの従来コートcheckpointについて、`model.encoder.backbone.module.`配下の全188 tensor・
85,697,296要素が別取得のViT-B/16単体ファイルとキー・値とも一致した。
Hubから再取得した**公開版でも全188 tensor・85,697,296要素の一致**を確認した。
公開版のSHA-256と比較結果は[dinov3_published_probe.json](../../runs/run-clean-standard-install-20261021/dinov3_published_probe.json)、
再現手順は[inspect_court_backbone.py](../../runs/run-clean-standard-install-20261021/inspect_court_backbone.py)にある。

したがって、推論専用の明示的な未初期化構築 → 全checkpoint strict loadへ分離すれば、
単体ファイルは不要にできる見込みである。ただし、現行コードはこれを実装しておらず、
今回のガイドとpreflightでは単体ファイルを必須とする。次の実装では、構築・推論の完全一致、
checkpointがbackboneを欠く場合の拒否、`backbone_sha256`を含むidentityの扱い、学習時の初期化経路を
検査する必要がある。DINOv3のsourceとLicenseへの準拠は引き続き必要であり、
単体ダウンロードを省くことが利用条件を消すとは扱わない。

## preflightと通常検証

資産を置いていないworktreeで、重み・補助資産・外部source・サンプル・DINO拡張の28項目を
一括して失敗として出力し、終了値1を確認した
（[missing_preflight.json](../../runs/run-clean-standard-install-20261021/missing_preflight.json)）。
Hydraはtask関数の戻り値をCLI終了値に伝えないため、明示的な`SystemExit(1)`を使う。
破損・読取中のidentity不整合・複数欠損・不適合拡張の集約、同一CLIPの二重hash抑制、
コピー先symlink・既存ファイルの拒否をCPUで検査した。新しい設定境界はaudit inventoryへ登録した。
関連pytest 56件、ruff、mypy、shell構文検査、設定/path audit 113境界、実装commitのpre-commitは合格。
初回CLIテストで終了値0を検出し、その修正後に全56件が成功した。
正確なコマンドと結果は[test_results.txt](../../runs/run-clean-standard-install-20261021/test_results.txt)と
[validation.json](../../runs/run-clean-standard-install-20261021/validation.json)にまとめた。

## 独立cloneでのCPU準備

未公開の実装commitを含むこのworktreeから`git clone --no-local`し、git object・venv・重みを
元treeと共有せず検証した。clone元がローカルの全branchを含むため、git容量は公開mainの通常cloneと
同じとは扱わない。チェックアウトするコードは上記の新履歴commitに固定する。
登録・申請・Google Driveが必要な資産は、この検証に限り、main treeの利用者取得済み`ckpt/`を
読み取り、表のSHA-256を検査して独立した実ファイルへコピーする。sourceの重み・data・venvは変更しない。
自作重みとサンプルは公開Hubから再取得する。取得やbuild中にGPU処理は起動しない。


CPU準備は`ready_for_gpu`で完了し、全28 checkが`ok`、Git statusは空だった。
新venvはPython 3.11.15 / torch 2.14.1+cu130で、main treeのvenvへのsymlinkではない。
第三者資産14点はコピー前後のSHA一致と別inodeを確認した。公式の取得先は導入ガイドを正とし、
このコピーを第三者向けの配布方法とは扱わない。
HFから再取得したモデルrevisionは`2186bf799bd6066a7b1b6db773fbb4ec48f5c0a5`、
sample revisionは`ebe2969bac795b5e8837dba463b6d687b3c99141`で、いずれもbyte pinに一致した。

| 工程 | 実測秒（queue待ち・GPU推論なし） |
| --- | ---: |
| clone | 53.73 |
| uv sync --locked | 310.92 |
| submodule | 4.15 |
| CUDA拡張ビルド＋CPU dispatch検査 | 154.32 |
| 利用者取得済み第三者資産14点の検査・コピー・再検査 | 49.47 |
| 公開モデルの取得 | 85.75 |
| 公開サンプルの取得 | 10.37 |
| 一括preflight（hashを含む） | 84.82 |
| 全CPU準備（checkout・付随検査・duを含む） | 761.47 |

準備完了時のworkspace全体は、apparent **19,779,048,171 bytes**、allocated
**19,940,069,376 bytes**（約19.94 GB、18.57 GiB）だった。
独立clone、新venv、isolated uv/HF cache、重み、サンプル、ビルド出力、検証ログと下記2frame smokeを含む。
この準備時点の容量にはGPUのcomponent cache・scene出力を含まない。完走後の容量は後述する。
uv cacheとvenv内のhardlinkは`du`で重複計上しない。
CUDA 13のbuild logは[build.log](../../runs/run-clean-standard-install-20261021/build.log)、
toolchainとpath契約は[build.json](../../runs/run-clean-standard-install-20261021/build.json)、
全工程の正確なコマンド・秒数・コピーreceiptは[preparation.json](../../runs/run-clean-standard-install-20261021/preparation.json)にある。
生ログの一部はローカルに残し、path・size・SHAをvalidation JSONに記録した。

追加の環境smokeとして、先行runの保存済みsceneを**読み取りだけ**で新venvに渡し、
CUDAを隠してSMPL meshの2frameを720×480 H.264へ出力した。ffprobeで2frameを確認した。
これは新しいcloneでのMatplotlib・SMPL資産・ffmpegの動作確認であり、
新pipelineが生成したsceneの可視化や見た目の受入ではない。
動画はoutputsにだけ置き、knowledgeには含めない。学習は実行しておらず、学習曲線はない。

## クリーンcloneでのGPU完走とscene検証

親はclean cloneをCWDとし、元repoの共有queue・resource `all`で
`bash <workspace>/run_pipeline.sh`を登録した。jobは
`0-8207442519961621865_2026-10-21T04-38-00-038378134Z_1544414_clean_install_standard`。
queueは`done`で、終了値0は親の実行報告による。実行時のCWD・commit・コマンドは
[run.json](../../runs/run-clean-standard-install-20261021/run.json)を原文のまま保存した。
実行前のgit statusとuncommitted patchは空だった。
実際に呼び出した[run_pipeline.sh](../../runs/run-clean-standard-install-20261021/run_pipeline.sh)も
保存し、準備時に発行したコマンドと一致することを確認した。GPU処理の再実行はしていない。

検証commitは`261c2186134bbe09e11b757e55aa1cb70bd02335`。
追記時のworktreeとclean cloneの`src/` treeはともに
`2f4801f8c883540a0adeb57281d6291d098132f7`で一致する。
Ninjaはlockに含まれず、OSの`/usr/bin/ninja` 1.11.1を使用した。

[pipeline.log](../../runs/run-clean-standard-install-20261021/pipeline.log)の
`Running pipeline...`（2026-10-21 13:38:08.538 JST）から`Saved:`（14:08:35.667 JST）まで
**1,827.129秒（30分27.129秒）**だった。queue待ち、Python/Hydraの起動、CPU準備、可視化は含まない。
CPU準備の761.47秒も、新規のCUDA toolkit・OS導入、第三者サービスへの登録・申請・手動取得は含まない。

`scene.npz`と必須sidecarの保存を確認し、`CUDA_VISIBLE_DEVICES=''`で正規reader
`load_scene_result()`を実行した。v2のmask・理由コード・コート参照座標系の整合性検査を通過し、
maskから独立に再集計した有効フレーム数も[metadata](../../runs/run-clean-standard-install-20261021/scene.metadata.json)と一致した。

| 項目 | クリーン導入 | 先行run 000028 |
| --- | --- | --- |
| schema / status | 2 / ok | 2 / ok |
| フレーム / FPS / 解像度 | 1,010 / 59.94006 / 1920×1080 | 同じ |
| camera / track ID | cam0, cam1, cam2 / 0, 1 | 同じ |
| ball_3d_frames | 822 | 821 |
| player_joint_frames（1関節以上が有効） | [1008, 993] | [1008, 993] |
| player_root_frames | [984, 993] | [1000, 993] |
| player_smpl_frames | [984, 993] | [1000, 993] |

同じ入力・推論処理でも、ballは1frame増、選手0のroot・SMPLは16frame減となった。
GPU推論の非決定性は候補だが、原因を切り分けていないため断定しない。
完走と保存契約の成立を確認した結果であり、bitwise再現性や精度改善・劣化の判定ではない。
ログの`Ball visible frames: 1010/1010`は2D観測のあるframeで、3D有効数822とは区別する。

## VRAMの測定範囲

親が`nvidia-smi`のGPU全体の`memory.used`を約5秒ごとに測り、
[CSV](../../runs/run-clean-standard-install-20261021/vram-samples.csv)を残した。
headerなしの2列はUnix timestamp（秒）と使用量（MiB）。368サンプルの最大は**4,928 MiB**だった。
親が報告したジョブ開始前のbaselineは**1,215 MiB**（他プロセス）で、CSVの最初の値も同じだった。
この最大値はGPU全体の使用量であり、baselineとの差を当該プロセスのピークとは扱わない。

CSVの先頭は13:38:31 JST、末尾は14:08:37 JSTで、timestamp上の範囲は**1,806秒（30分6秒）**。
件数×公称5秒による約30分40秒ではなく、実timestampの差を採る。実間隔は4〜6秒、中央値5秒だった。
pipelineのログ開始後22.462秒を含まず、測定間隔より短いピークも捉えないため、
瞬間的なピークや必要VRAMの下限は不明である。実機はRTX 5060 Ti / 16 GBで、
より小さいVRAMのGPUへ動作保証を広げない。学習や学習曲線はない。

## 新sceneのCPU可視化と完走後の容量

親は生成直後のsceneに[visualize.sh](../../runs/run-clean-standard-install-20261021/visualize.sh)を実行し、
終了値0、プロセス起動込み**363秒（6分3秒）**と報告した。
[visualization.log](../../runs/run-clean-standard-install-20261021/visualization.log)の
scene読込開始14:09:04.373 JSTから保存成功14:14:59.463 JSTまでは**355.090秒**で、
親の計時とは開始地点が異なる。`CUDA_VISIBLE_DEVICES=''`のCPU描画であり、GPU処理は起動していない。

新MP4を`ffprobe -count_frames`で検査し、H.264・1200×800・1,010フレーム・
16.850167秒・567,965 bytesを確認した。親が9秒時点のframeでコート・両選手のSMPL・
ボール・ミニマップの描画を確認した。この目視は親の報告による単一frameの確認で、
全frameの3D位置・姿勢や真値との精度を承認したものではない。

完走・可視化後のworkspaceを`du -sB1`で再測定し、apparent **20,703,572,320 bytes**、
allocated **20,866,301,952 bytes**（約20.87 GB、19.43 GiB）だった。
独立clone・新venv・分離cache・重み・ビルドに加え、component cache・scene・全編MP4・
先行2frame smokeとログを含む。システムのCUDA toolkit・OSパッケージと、第三者資産を
別途ダウンロードする作業領域は含まない。最低ディスク容量の保証にはしない。

## 保存した証拠と残る限界

時間の算出根拠、VRAMの測定範囲、mask再集計、`ffprobe`、完走後のディスク、
元ファイルのpath・size・SHA-256は[gpu_validation.json](../../runs/run-clean-standard-install-20261021/gpu_validation.json)にある。
通常上限256 KiBを超える生queueログ（291,485 bytes）は元のqueueに残し、同JSONにreceiptだけを保存した。
knowledgeに加えたログ・CSV・metadata・実行scriptはいずれも通常上限以下である。
scene予測配列（153,173,597 bytes）、動画、重み、component cache、ビルド済みバイナリは
サイズにかかわらず保存対象外とし、元のoutputsに残している。

導入のclone元は手順書のGitHubではなく、このworktreeを使う`git clone --no-local`だった。
新履歴・固定commitの独立checkoutと新venv・公開モデル/サンプルの再取得は検証したが、
GitHubのネットワーク転送、新規の第三者登録・手動取得、新規OS/driver導入は検証していない。
GPU環境の最小容量、上記frame数差の原因、独立した3D精度・全frameの目視確認も今回の受入外である。
この1clipで確認した導入手順と動作環境を #1276 のREADMEへ渡し、#1275 のGIF作成へ利用できる。

追記後の通常検証では、GPUを隠した`pytest -n 2`でpreflight・導入準備・DINO・設定とknowledge保存契約の
計91件が成功した。設定/path auditは113境界を通過し、branchの実装差分に対するpre-commitの
ruff・mypyも成功した。knowledgeの再現script検査はmissing 0、旧履歴commitが手元にない参照8件は
未検証のwarningとして残した。新しく保存した7ファイルは全て元のSHA-256と一致し、
run全体も保存基準内だった。コマンドと結果は`validation.json`の`completion_checks`に追記した。
