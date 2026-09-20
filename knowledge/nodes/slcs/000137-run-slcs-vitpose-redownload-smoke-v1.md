---
task: slcs
sequence: 137
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-vitpose-redownload-smoke-v1
type: run
title: '再取得ViTPose-H: strictロードと実RGB 2人GPU smoke'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  model: ViTPose-H COCO17, head1280/17/2deconv, bfloat16, flip_test=true
  loss: inference smoke only
  data: Meiji video_000/clip_000 cam0 frame0, frozen s42-004 two-person boxes
metrics:
  all_finite: true
  confidence_min: 0.34765625
  confidence_mean: 0.7111672759056091
  confidence_max: 0.89453125
  confidence_ge_03_fraction: 1.0
  keypoints_inside_image_fraction: 1.0
  load_seconds: 8.51287542499631
  forward_seconds: 0.9770847969994065
  gpu_peak_allocated_bytes: 3940516864
repro:
  commit: 6ca6507564f091ac924040e78156c063e1948bc1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash knowledge/runs/run-slcs-vitpose-redownload-smoke-v1/repro.sh NEW_ABSOLUTE_OUTPUT_DIRECTORY SESSION_ID
artifacts:
  run_dir: knowledge/runs/run-slcs-vitpose-redownload-smoke-v1
  output_dir: outputs/tennis_scene/analyze/vitpose_redownload_smoke/s42-001
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789775189411271550_3174748_slcs-vitpose-redownload-smoke-v1.log
  overlay: knowledge/runs/run-slcs-vitpose-redownload-smoke-v1/overlay_zoom.png
parents:
- run-slcs-vitpose-redownload-v1
relations: []
tags:
- slcs
- vitpose
- gpu-smoke
- real-rgb
---

## 考察 / Findings

### 要約
再取得したViTPose-Hは既存モデルにstrict=Trueでロードでき、不足キー・余剰キーはともに0だった。実動画の2人に対するGPU推論で(2, 1, 17, 3)の有限出力を得た。共有training queueのresource=allで1ジョブのみ実行し、成功した。

### アーキテクチャ詳細
元動画cam0のframe0（1920×1080）を既存ViTPosePose2D経路で直接読む。旧観測s42-004の同フレームにある2人分のbboxは、両方ともobserved/support maskがtrueだった。TrackResultでbase_enlarge=1.2のsquare boxへ変換し、COCO17・head1280/17・2deconv・bfloat16・flip testを使用した。DINOはロードも推論もしていない。入力の識別・box・source hash・依存バージョン・実行driver hashはprovenance.json、入力配列と予測はinput.npz/predictions.npzを正とする。

### メトリクスの解釈
34点すべて有限かつ画像内で、confidenceは最小0.34765625、平均0.7111673、最大0.89453125。ロード約8.51秒、2人の推論合計約0.98秒。これらは機能smokeの観測値で、速度ベンチマークやGTに対する精度評価ではない。GPUはRTX 5060 Ti、ピークallocated memory約3.94GB。metrics.jsonにstrictロードの実行時結果を保存した。

### アーキテクチャ⇄メトリクスの因果考察
公開pin一致だけでなく実際のモデル構築・strictロード・CUDA forward・UDP復号の一連の経路が動くことを確認した。元画像をcrop/拡大してから同じ保存済み予測を座標変換して細線で重ねたoverlay_zoom.pngでは2人の骨格を視認できる。画像は予測であり正解ラベルではない。遠方人物は元解像度が小さく、位置精度を断定できない。

### 既存実験との比較
親の再取得runが確認した公開SHA256とinstallation.jsonを参照した。本runではcheckpointのhashを再読せず、inode/size/mtimeが記録と一致し、実行前後でも不変であることを確認した。DINO検証で止まった全パイプライン再利用とは切り離した、ViTPoseのみの成功である。

### 次に有効な実験
DINO側の明示的な方針が確定後、親タスクで全パイプライン再利用を確認する。本runを理由にDINOや既存cache全体の検証成功を主張しない。

実行時baseはqueue captureのrun.jsonに残した。driverは実行時未追跡であり、そのhashをprovenance.jsonへ記録している。queue_repro.shは当時の自動captureの原本で、未追跡driverを含まないため単独の再現手順とは扱わない。repro.shはdriverを含むcheckoutから新規保存先・共有queueへ再投入する。nodeのrepro.commitはdriverを含む版に固定した。共有queue workerの起動方法はtraining-queue skillを参照する。CPU描画は推論を再実行せず、元frame SHAの一致と保存prediction SHAをvisualization.jsonへ記録した。
