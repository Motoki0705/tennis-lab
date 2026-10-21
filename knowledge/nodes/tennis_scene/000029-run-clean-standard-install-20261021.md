---
id: run-clean-standard-install-20261021
type: run
task: tennis_scene
sequence: 29
recorded_at: '2026-10-21'
date: '2026-10-21'
title: 公開標準pipelineのクリーン導入準備と公式CLIP-ReID比較
issue: 1274
provider: codex
status: done
config:
  device_used: cpu
  gpu_executed: false
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
parents:
- run-multiview-sample-standard-pipeline-20261021
relations: []
papers: []
tags:
- installation
- public_sample
- cpu_validation
---

## 範囲と受入の限界

#1274の導入ガイド・資産preflight・独立cloneによるCPU準備を検証した。
GPU推論は親セッションがtraining queueで実行する担当であり、このrunでは実行しない。
[既存環境での標準サンプル完走](000028-run-multiview-sample-standard-pipeline-20261021.md)と、
クリーン環境のGPU完走・3D精度・可視化の受入を混同しない。
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
元treeと共有せず検証する。clone元がローカルの全branchを含むため、git容量は公開mainの通常cloneと
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
GPUのcomponent cache・scene出力はまだ存在しない。uv cacheとvenv内のhardlinkは`du`で重複計上しない。
CUDA 13のbuild logは[build.log](../../runs/run-clean-standard-install-20261021/build.log)、
toolchainとpath契約は[build.json](../../runs/run-clean-standard-install-20261021/build.json)、
全工程の正確なコマンド・秒数・コピーreceiptは[preparation.json](../../runs/run-clean-standard-install-20261021/preparation.json)にある。
生ログの一部はローカルに残し、path・size・SHAをvalidation JSONに記録した。

追加の環境smokeとして、先行runの保存済みsceneを**読み取りだけ**で新venvに渡し、
CUDAを隠してSMPL meshの2frameを720×480 H.264へ出力した。ffprobeで2frameを確認した。
これは新しいcloneでのMatplotlib・SMPL資産・ffmpegの動作確認であり、
新pipelineが生成したsceneの可視化や見た目の受入ではない。
動画はoutputsにだけ置き、knowledgeには含めない。学習は実行しておらず、学習曲線はない。

## 親へ引き継ぐGPU検証

実行環境はローカルstateの`campaign-1267/clean-env/standard-install/repo`、
`run_pipeline.sh`と`visualize.sh`はその1つ上にある。
親は**clean cloneをCWD**にし、元repoの共有`TRAINING_QUEUE_DIR`を明示し、resource `all`で
`bash <workspace>/run_pipeline.sh`を登録する。こうすることでqueueのreproが実際の実行commitを記録する。
サブエージェントはqueueへ登録していない。

完走の判定は、queue成功に加え、`scene.npz`と`scene.metadata.json`の保存、
`load_scene_result()`でのv2 mask・理由コード・座標系の検証、schema 2 / status ok / 1,010 frames / 3 camerasを
確認すること。3Dの有効数を再集計し、既存runの821 ball frames等と差があれば内容を記録する。
GPU所要時間とピークVRAMも採録する。その後に`visualize.sh`で新sceneを描画し、動画で確認する。
これらはこのrunの**未実施事項**であり、CPU準備の成功だけを根拠に完走とは記録しない。

実測後の変更は導入ガイドのNinja要件・このknowledge記録だけで、推論に使う`src/`は
検証commit `261c2186134bbe09e11b757e55aa1cb70bd02335`と同じである。
Ninjaはlockに含まれず、OSの`/usr/bin/ninja` 1.11.1を使用した。
