---
id: run-slcs-meiji-v8-observe-v1
type: run
title: Meiji v8の56clip観測生成後に3cameraのproducer SHA不一致を検出
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: failed
config:
  stage: observe
  dataset: meiji_3cam
  observation_directory: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  people_policy: temporal_continuity
  checkpoint_pins: build_slcs_dataset.yaml
metrics:
  source_clips: 57
  excluded_clips: 1
  generated_clips: 56
  generated_clip_frames: 29148
  generated_camera_frames: 87444
  camera_observations: 168
  people_producer_pin_mismatches: 3
  queue_recorded_clip_failures: 0
repro:
  commit: fdb6aa99580a0b5e92c68bd4254a91457ce3f950
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=observe paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v8-observe-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v8/s42-001
  inventory: knowledge/runs/run-slcs-meiji-v8-observe-v1/inventory.json
  preservation: knowledge/runs/run-slcs-meiji-v8-observe-v1/preservation.json
parents:
- run-slcs-meiji-v8-root-replay-v2
- run-slcs-meiji-inference-repeatability-v1
relations: []
tags:
- slcs
- meiji
- observation
- provenance-failure
---

## 考察 / Findings

### 要約
対象56clip・29148clip-frameの観測ファイルが揃いqueueはdoneだったが、事後の固定pin照合で3cameraのpeople receipt SHA不一致を検出した。観測の採用監査をfailedとし、後続buildをcancelした。正常終了をデータセット完成とは扱わない。

### アーキテクチャ詳細
実行commit fdb6aa99580a0b5e92c68bd4254a91457ce3f950、Meiji v8のtemporal_continuityで3cameraを処理した。開始時checkpoint_verificationはCourt/DINO/ViTPoseの固定SHAを通過。people/raw detectionsのhashは当時からdual_sha256であり、現HEADまで関連4sourceに変更はない。固定pinとの照合が開始時のみで、後の各receipt計算値を固定pinへ再照合する境界が無かった。

### メトリクスの解釈
57clipのうち002/001を既存理由で除外。全168people NPZのshapeとmask dtype、metadata schema/policyを照合し、NPZ+metadata計336fileの前後dual SHAは一致した。005cam2/006cam2のdetector SHAと012cam2のpose SHAの計3件が固定値と異なる。他165cameraの両SHAは一致した。元jobのfailuresは空で、queue終了状態と監査結果を分離する。最終3D教師は未生成である。

### アーキテクチャ⇄メトリクスの因果考察
3cameraのraw detection receiptは期待DINO SHA、people receiptだけ上記値が異なった。二実装一致は一度の読取内の整合性を確かめるが、異なる時点の読取が同じ期待内容であることは別途照合が必要だった。重みの実差、読取異常、hash計算等の根本原因は未確定。receipt保存時刻からhash計算の順序を推定しない。metadataを正しい文字列へ書き換えても配列の由来は証明できない。

### 既存実験との比較
過去の間欠的hash不一致では二実装や別process間の差を確認した。今回は二実装経路で作られたproducer receiptが開始時pinと食い違う別の検出境界である。過去の652frame反復一致も、この全体runの正常性の証明にはならなかった。

### 次に有効な実験
カメラごとの生成/再利用時にpinと兄弟receiptを照合し、推論前後の不一致を公開前に拒否する。該当3cameraのraw/people計12fileは独立出力へbyte一致で保存済み。元receiptを改変せず隔離・再生成し、旧新配列を比較する。後続buildはcancel済み、Meiji v8のscene/qualityは0件。1clip再現性は校正採用後に再投入する。
