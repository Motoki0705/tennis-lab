---
id: run-multiview-sample-standard-pipeline-20261021
type: run
task: tennis_scene
sequence: 28
recorded_at: '2026-10-21'
title: 公開3カメラサンプルの標準pipeline完走確認
issue: 1273
provider: codex
date: '2026-10-21'
status: done
config:
  clip: video_000/clip_000
  pipeline: src/tennis_scene/configs/pipeline.yaml
  device: cuda
  camera_ids: [cam0, cam1, cam2]
  sampling_max_frames: 1024
  sampled_frames: 506
metrics:
  scene_schema_version: 2
  frames: 1010
  width: 1920
  height: 1080
  fps: 59.94006
  pipeline_to_save_seconds: 1835.979
  ball_3d_frames: 821
  player_joint_frames: [1008, 993]
  player_root_frames: [1000, 993]
  player_smpl_frames: [1000, 993]
repro:
  commit: cb8f1be66c34256046d1388962ee0f1dde3a9811
  branch: codex/multiview-sample
artifacts:
  run_dir: knowledge/runs/run-multiview-sample-standard-pipeline-20261021
  execution: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/run.json
  build: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/build.json
  log: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/pipeline.log
  failed_attempt_log: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/attempt-1.log
  input_manifest: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/clip.json
  metadata: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/scene.metadata.json
  validation: knowledge/runs/run-multiview-sample-standard-pipeline-20261021/validation.json
  output_dir: outputs/tennis_scene/generate/multiview_sample_release_20261021_a2
parents: []
relations: []
papers: []
tags:
- integration_validation
- public_sample
- gpu
---

## 目的と観測

#1273の公開サンプルとして準備したMeiji `video_000/clip_000` の動画3本から、
標準の `src.tennis_scene.scripts.run_pipeline` がSceneResult v2の保存まで完走するかを確認した。
親セッション（Claude Code）が共有training queueでGPU実行し、このノードはそのログと出力を
CodexがCPUで検証して記録したもの。記録時にGPU処理は再実行していない。

attempt-1はDINO CUDA拡張の欠損で停止し、明示的な拡張ビルド後のattempt-2は `status=ok` で完走した。
既存のローカルcheckpoint・外部sourceと公開予定bundleを使った動作確認であり、
Hubからの再取得やクリーンcloneからの導入、独立した3D精度の受入は #1274 以降で確認する。

## 入力とコードの固定

公開先は `Motoki098/tennis-lab-multiview-sample`。公開予定bundleと、実際にpipelineへ渡した
`outputs/multiview_sample_preflight/input_data/samples/` の各ファイルのSHA-256が一致した。
HTTP経由で取得した入力ではない。公開manifestには元manifestの同期値・時間軸・letterboxを残し、
ローカル絶対パスをraw動画の相対識別子へ置き換えている。

| ファイル | SHA-256 |
| --- | --- |
| cam0.mp4 | `f99f0024067da49805654d65eae848407c47ee710411d91892f0921fe8331f39` |
| cam1.mp4 | `ac333c334b81141ee72098be6c95f5a6f6e1ccc5faf6e058f6fd3491cc1ac96e` |
| cam2.mp4 | `fd82f0b573e9f8788cc56567dabdaf0c0c3775fd5dbede4835707277ef70c461` |
| clip.json（公開manifest） | `410d77571f553123ffd57ebb457f45f4798f57d7ab6a0828e70b0a47ec42eaf2` |

動画3本は1920×1080・59.94006fps・1,010フレーム。
事前のCPUデコードは全3,030フレームで成功し、実際のframe samplingは先頭0から末尾1009まで506フレームで、
`frame_sampling.max_frames=1024` の上限内だった。sceneは元動画の1,010フレームへ戻して保存される。
cam0は1280×960から1440×1080へ拡大し、左右各240pxの余白を付けた入力である。
被撮影者の同意・公開条件は [LICENSES.md](../../../LICENSES.md#サンプルdatasetの利用条件) を正とする。

実行時commitは旧履歴の `925631791ceb271ef88cf919061317891097dc0e`。
#1310の履歴書き換え後に、新しい履歴へ載せ替えた同じ変更は `001a0afafb804b032f4dff9bf099d10f426ad305`（未公開）で、treeは実行時と同じ `9006f4f657eaf6e2fec11fc92eb33f15858125f5` だった。
その後、参照の付け替え（#1312）を含むmainへ載せ替えた公開commitは `cb8f1be66c34256046d1388962ee0f1dde3a9811` である。実行時のtreeとの差は、#1312 による参照の付け替え（`src/` 配下のREADME 4件とknowledge配下）だけで、実行されるコード・設定は同一である。
queueが実行前に採録したgit statusとuncommitted patchはともに空だった。
[run.json](../../runs/run-multiview-sample-standard-pipeline-20261021/run.json) は実行時の記録をそのまま保存しているため、
旧commitと当時の絶対パスを保持する。再現時のコード参照には上記の新commitを使う。
既存のknowledge内のcommit参照は #1312 の担当範囲なので変更していない。

## 実行手順と失敗の原因

1. 親が [SMPL補助資産の取得・配置手順](../../../src/submodules/README.md#smpl補助資産の取得) に従い、
   `install_smpl_assets` で補助資産5点を `ckpt/body_models/gvhmr/` へ配置した。
   記録時にも5点すべてを公式固定版の期待SHA-256と照合した（[validation.json](../../runs/run-multiview-sample-standard-pipeline-20261021/validation.json)）。
   配置時の取得元ディレクトリと所要時間は採録されていない。
2. 最初のqueue job `0-8207457415463193113_2026-10-21T00-29-44-536806886Z_172080_multiview_sample_release` は
   `DINO CUDA extension is not installed` で終了した（exit code 1）。
   [失敗ログ](../../runs/run-multiview-sample-standard-pipeline-20261021/attempt-1.log) の案内は
   `TENNIS_LAB_BUILD_CUDA_OPS=1 .venv/bin/python setup.py build_ext --inplace` だが、
   現在のビルド契約には `TENNIS_LAB_DINO_OPS_BUILD_CONFIG` も必要である。
   案内どおりのビルドが `TENNIS_LAB_DINO_OPS_BUILD_CONFIG is required` で失敗したことは親の実行報告による。
   このビルド失敗の生ログは未採録であり、pipeline失敗ログに含まれるとは扱わない。
3. 親は既存の [build_dino_extension.sh](../../runs/run-scene-component-meiji-fullclip-b863-20260925/build_dino_extension.sh) を用い、
   worktreeの `outputs/tennis_scene/generate/multiview_sample_release_20261021_dino/` に拡張をビルドした。
   `CUDA_HOME=/usr/local/cuda`、`TORCH_CUDA_ARCH_LIST=12.0`、`MAX_JOBS=2` と実際のpath契約は
   [build.json](../../runs/run-multiview-sample-standard-pipeline-20261021/build.json) に残る。
   このscriptは `TENNIS_LAB_CUDA_OPS_BUILD_TARGET=all` でビルドする。build中はGPUを使用しない。
4. その `lib/` を `PYTHONPATH` に明示し、出力先とcacheを新しいattempt用に分けて共有queueへ再登録した。
   完走jobは `0-8207457047915595751_2026-10-21T00-35-52-084404248Z_247100_multiview_sample_release_a2`、
   GPU予約は `all`、queue状態は `done`。標準設定へのoverrideはpath・出力名・cache保存先だけで、
   手動ball・人物対応・コート注釈のimportは追加していない。正確なコマンドは `run.json.command` に保存した。

再ビルド部分の再現例（新commitをcheckoutした専用worktreeのrootで実行し、
`ASSET_ROOT` は `third_party/DINO` を持つ取得済み資産のrootへ明示する）:

```bash
git checkout --detach cb8f1be66c34256046d1388962ee0f1dde3a9811
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  bash knowledge/runs/run-scene-component-meiji-fullclip-b863-20260925/build_dino_extension.sh \
  "$ASSET_ROOT" "$PWD/outputs/tennis_scene/generate/multiview_sample_release_20261021_dino"
```

これは実行時のGPUアーキテクチャを固定した再現用の例であり、他GPUの導入手順ではない。
続くGPU推論は `run.json.command` の入力・資産・出力rootをそのcheckoutに合わせ、共有training queue経由で実行する。
汎用のビルド手順の正本整備と、古いREADME・エラーメッセージの修正は #1274 に引き継ぐ。

## 結果と所要時間

[pipeline.log](../../runs/run-multiview-sample-standard-pipeline-20261021/pipeline.log) の
`Running pipeline...`（2026-10-21 09:36:05.790 JST）から `Saved:`（10:06:41.769 JST）まで
**1,835.979秒（30分35.979秒）**。queue待ち、SMPL配置、DINOビルドは含まない。
それらの所要時間、ピークVRAM、クリーン導入時のディスク使用量は未測定。

`load_scene_result` により保存済みarchiveと必須sidecarをCPUで読み込み、v2のmask・理由コード・
参照座標系の整合性検査を通過した。metadataの値をmaskから独立に再集計して一致を確認した。

| 項目 | 結果 |
| --- | --- |
| schema / status | 2 / ok |
| フレーム / FPS / 解像度 | 1,010 / 59.94006 / 1920×1080 |
| camera / track ID | cam0, cam1, cam2 / 0, 1 |
| ball_3d_frames | 821 |
| player_joint_frames（少なくとも1関節が有効なフレーム） | [1008, 993] |
| player_root_frames | [1000, 993] |
| player_smpl_frames | [1000, 993] |

ログの `Ball visible frames: 1010/1010` は、いずれかのcameraで2D観測があるフレーム数である。
3Dの有効数は `ball_3d_valid` の821で、両者を混同しない。
これは1clipでの処理成立と保存契約の検証であり、真の3D位置との誤差・見た目・他clipでの完走率は未評価。
過去の注釈ballを使った同clipのrunとは入力・コードが異なるため、精度改善の因果比較は行わない。
モデル採用や品質に関する従来の判断は変更しない。

## 成果物の保存と次の確認

sceneは `outputs/tennis_scene/generate/multiview_sample_release_20261021_a2/scene.npz`（154,410,684 bytes）と
`scene.metadata.json` に残した。`scene.npz` は予測配列なので、
[保存基準](../../README.md#run成果物の保存基準) に従いknowledgeへコピーしない。
動画・重み・component cache・ビルド済みバイナリも追加しない。

knowledgeには、元の `run.json`・`build.json`・小さなHydraの `pipeline.log`・`attempt-1.log`・
公開 `clip.json`・`scene.metadata.json` と、入出力のサイズ/SHA-256・queue完了状態・CPU検証結果を含む
`validation.json` だけを保存した。生queueログ（290,925 bytes）は256 KiBを超えるため元のqueueに残し、
そのpath・サイズ・SHA-256を `validation.json` に記録した。
これは学習ではなく、TensorBoard曲線は存在しない。

HFへのuploadは親が行う予定。#1274では公開物を再取得するクリーン環境でDINOビルド案内を検証し、
同じ入力hashとv2検査で再実行する。#1275の全体像GIFは、このsceneと元動画を使って別途作成・確認できる。
