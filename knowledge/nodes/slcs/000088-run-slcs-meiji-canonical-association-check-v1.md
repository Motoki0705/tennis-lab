---
task: slcs
sequence: 88
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-canonical-association-check-v1
type: run
title: Meijiの視点別人物番号をproduction経路で整列するCPU照合
provider: codex
date: '2026-09-19'
status: done
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
config:
  observations: tennis_scene/precompute/meiji_dino_vitpose/s42-004
  generation_config: tennis_scene/generate/meiji_rgb_v8/s42-001/config.yaml
  reference_camera: cam0
  view_half_turns: [false, false, true]
  device: cpu
  diagnostic_only: true
  threshold: 0.3
  joint_indices: [5, 6, 11, 12]
metrics:
  clips: 10
  camera_streams: 30
  clip_frames: 6601
  array_equality_passed_clips: 10
  unchanged_input_source_files: 60
  changed_clip_player_counts: 1
  video000_clip008_canonical_p1_support_frames: 447
  video000_clip008_canonical_p1_support_fraction: 0.9331941544885177
repro:
  commit: 740e788877a00971a7ab3d913458c8328379815e
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-meiji-canonical-association-check-v1/repro.sh /absolute/path/to/NEW_OUTPUT_DIRECTORY
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-canonical-association-check-v1
  output_dir: outputs/tennis_scene/analyze/meiji_canonical_association/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-canonical-association-check-v1/results.json
  provenance: knowledge/runs/run-slcs-meiji-canonical-association-check-v1/run.json
parents:
- run-slcs-meiji-observation-review-v4
- run-slcs-meiji-observation-review-v3
relations: []
tags: [slcs, meiji, cross-view, canonical, cpu-diagnosis]
---

## 考察 / Findings

### 要約
review-v1〜v4の重複なし10clipについて、実productionの`associate_people`をCPUで呼び、raw人物配列に支持maskを適用して固定順に並べた配列と全要素一致した。全30streamでcam0/cam1はraw `[0,1]`、cam2は`[1,0]`となる。逆端のcam2でraw P0が異なる人物なのは視点内near/far順と整合し、教師入力には既存の整列経路がある。

### アーキテクチャ詳細
読み込んだproductionの`__file__`、cwd、git root、HEADを記録し、reconstruction sourceが実行commitと一致することを確認した。モデルload/forward、追加推論、生成stageは実行していない。設定はreference_camera=cam0、view_half_turns=[false,false,true]、全manifestのcamera順はcam0,cam1,cam2。court.npzの実キーはkeypointsとhomographiesで、後者の3×3×3配列をそのままproductionへ渡した。productionが保存する`track_ids_near_far_cam0`をraw track_idsへ逆引きして順序を得た。confidenceへpose_supported_maskを適用した独立配列の期待順との完全一致、shape/finite、想定順一致を例外付きで検証した。入力NPZ・court・config・manifest・過去結果・読んだsourceの60fileは実行前後のdual_sha256が一致。配列を保存先へ書き戻していない。再実行scriptと、実行commitに未収録だった診断script・review-v4入力JSONだけのpatchを保存した。別の一時indexへ実行commitを読み込み、patch適用と両ファイルのbyte一致を確認した。

### メトリクスの解釈
整列後の同一playerについて両肩5,6・両腰11,12がすべて0.3以上のカメラを数え、2視点以上のframe数と率を全6601frameで算出した。video_000/clip_008のP1だけがraw-index集計476/479=99.37%からcanonical集計447/479=93.32%へ29frame減少。他19組のclip/playerは数値不変で、詳細はresults.jsonに保存した。これは再投影・速度・幾何・品質weightを含まないconfidence支持診断で、最終教師coverage、scene生成成功、教師精度を意味しない。学習runではなく収束曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察
productionは支持された足首の平均をhomography逆変換し、cam2のコート座標を半回転して、全期間の中央値yで固定人物順を決める。raw列番号のまま跨カメラ集計するとcam2だけ異なる人物のconfidenceを混ぜる。clip_008の差は、同一の配列・閾値に人物整列だけを適用した結果である。他clipで支持数が同じでも、raw-indexが同一人物という根拠にはならない。この照合は固定court-end政策の実装動作を検証したもので、全frameの追跡ID正解やコート校正の3D精度を独立に証明しない。

### 既存実験との比較
v3 support_checks.jsonの3clipは当時のNPZ SHAとraw countを再現した。従って旧集計の再現性はあるが、未整列raw P0/P1を跨カメラで数えた診断であり、canonical人物ごとの支持率として解釈できない。v4の視覚的な逆転疑義に対して、本runは教師入力の既存補正経路が10clipで想定どおり動く根拠を追加した。過去の視点内画像所見やraw値自体を改変していない。

### 次に有効な実験
観測生成の完了後、既存の整列経路を使ったscene生成結果について、再投影・速度・幾何・品質weightを含む実際の教師採用率を別runで検証する。今回のconfidence支持率を最終採用率へ置き換えて報告しない。
