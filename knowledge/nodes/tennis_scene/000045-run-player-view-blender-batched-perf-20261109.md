---
id: run-player-view-blender-batched-perf-20261109
type: run
task: tennis_scene
sequence: 45
recorded_at: '2026-11-09'
title: 連続animation dispatchは選手視点描画の処理時間を短縮せず
issue: 1727
provider: codex
date: '2026-11-09'
status: done
config:
  clip: clip000_20261103
  player_ids:
  - 0
  - 1
  source_frames:
  - 16
  - 17
  - 18
  - 19
  - 730
  - 731
  - 732
  - 733
  - 734
  - 735
  - 736
  - 737
  width: 1280
  height: 720
  fps: 59.94006
  samples: 16
  persistent_data: true
  render_dispatch: contiguous_animation_batches
  human_acceptance: pending
  throughput_gate: failed
metrics:
  rendered_frames: 24.0
  mean_interval_id0_s: 4.207164
  mean_interval_id1_s: 4.202864
  max_valid_interval_id0_s: 4.248469
  max_valid_interval_id1_s: 4.282473
  max_evaluated_pc2_error_m: 0.0
  metadata_crc_passed: 1.0
  throughput_gate_passed: 0.0
repro:
  commit: caca7e69b71da55b8a4430b52d587e4c81992380
  command: bash outputs/tennis_scene/visualize/player_view/blender-acceptance/run_perf_batched_queue.sh
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-view-blender-batched-perf-20261109
  drive_record: operations/colab/player-view-c3-20261109/c3-perf-batched
  drive_outputs:
  - outputs/tennis_scene/visualize/player_view/blender-acceptance/perf-batched/artifacts
parents: []
relations:
- to: run-player-view-blender-persistent-perf-20261109
  rel: related
  note: 同じ描画設定で連続区間のdispatchをまとめる性能仮説を検証した。
papers: []
tags: []
summary_impact: none
---

# 結果

`caca7e69b` のBlender rendererで、明示的なframeごとのcamera/PC2/球/mask設定を保ったまま、
連続元frame区間を一度の`render(animation=True)`へまとめた。
ID0/ID1のそれぞれ16〜19と730〜737の12frameを、Blender5.2.1 LTS・NVIDIA L4/OPTIXで実描画した。
1280×720、16 samples、元FPS59.94006、persistent data有効。
水平画角90度、球半径0.05m、本人mesh全体非表示は仮設定で、材質alpha1/emission0、motion blurなし。
frame_change_preで元frameの値を直接設定し、render_writeで完成PNGだけを公開した。
keyframe曲線・Bezier・時間再サンプリングは使っていない。

全24frameでcamera/ID/source/cache index/全mask・PC2全頂点・PNG CRCを照合し、
PC2最大誤差は0m、書出し順は選択元frame列と一致した。
既存previewにあるsampleの画像差は平均約0.27〜0.46LSBで、元previewのditherとの差を含む。
camera730→731→732と734→735→736、球16→17→19、相手meshだけの欠測を確認した。
比較元にない18・733・737は参照画像の数値比較を欠測として記録し、metadata/CRCの確認は行った。
この検証はrendererが保存入力を忠実に読むことを確かめるもので、3D精度や人間の受入の評価ではない。

source差1の区間だけで、初期化を除くPNG mtime間隔を測った。
ID0の平均4.20716秒・有効cameraでの最大4.24847秒、ID1は平均4.20286秒・最大4.28247秒。
先行するstill呼出し方式の4.20046/4.21246秒とほぼ同じで、処理時間の改善は観測されなかった。
**連続dispatchで固定費を減らせるという仮説はこの実測では支持されず、全区間再開のgateは失敗。**
full-batched本番は起動しなかった。effectのなかったhandler分岐は次のsource変更で撤去した。

GPUレイトレーシング以外の処理を調べ、次は同じOpenImageDenoiseのGPU使用を明示して検証する。
[Blender5.2公式addon source](https://github.com/blender/blender/blob/blender-v5.2-release/intern/cycles/blender/addon/properties.py)
ではdenoising_use_gpuのdefaultはFalseで、render deviceとは別項目である。
このrunはdenoising deviceそのものを記録していないため、CPU denoisingが律速だったという説明は
ここでは仮説に留まる。次のrunでは設定とrender秒数を描画記録へ保存する。

# 保存と再現

Colab job `c3-perf-batched` はcompleted・returncode0、Drive保存は2026-11-08T18:41:57Z。
Drive recordをdownload --verifyして登録した。親が次の性能修正へ継続指示を出すため
運用Codex turnを明示停止したが、remote GPU jobの失敗へ読み替えない。
このrunでのdoneは少数frame検証の終了だけを指し、全区間動画は未完成、人間の定性受入はpending。

Drive root相対入力は
`outputs/tennis_scene/visualize/player_view/blender-acceptance-20261109/package`。
出力tarと記録はfrontmatterのDrive参照。実行したqueue/wrapperは同梱し、
`colab-original-repro.sh`は記録由来の原文を保持する。外側の`repro.sh`が同梱scriptを元pathへ配置する。
run bundleを新規checkoutの外へ置き、`TENNIS_REPO`で指定する。
先行runの監査済みprefix388frameが必要なので、同runのcheckpoint tarからPNGとrowを復元する。
既存の異なる出力を上書きしない。歴史的なGPU期限は更新せず、新実行は別の予算・記録で行う。

学習・採用checkpoint・標準pipeline・3D精度の判断は変わらず、summary_impactはnone。
TensorBoardは該当しない。C3の人間受入やC5開始の承認を意味しない。
