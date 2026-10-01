---
id: run-i964-detectors-val-meiji-r1-20260929
type: run
task: player_detection
sequence: 4
recorded_at: '2026-09-29'
title: 選手DINOのvalidation比較とMeiji旧box不一致の診断
issue: 964
provider: codex
session: 01a0eb46-5947-7a73-a409-92ae9b952dfd
date: '2026-09-29'
status: done
config:
  chat_split: val (checkpoint selection split)
  meiji_scope: pipeline_court_roi
  score_threshold: 0.3
  min_iou: 0.5
  short_side: 800
  max_long_side: 1333
metrics:
  chat_val:
    dataset: /home/kamimura/projects/tennis-lab/data/player_detection/chat-player-v1
    split: val
    frame_stride: 1
    selection_stats:
      split_frames: 8175
      dropped_unreviewed: 0
      dropped_unresolved_player: 804
      dropped_no_visible_box: 0
      dropped_by_stride: 0
      dropped_small_instances: 0
      instances: 14742
      frames: 7371
    metrics:
      coco_person:
        map: 0.4261105954647064
        map_50: 0.7801627516746521
        map_75: 0.4052041172981262
        precision: 0.24001468300442239
        recall: 0.9314204314204314
        f1: 0.3816764186738197
      player_ft:
        map: 0.5178632736206055
        map_50: 0.9452608227729797
        map_75: 0.4981183409690857
        precision: 0.9575064862211626
        recall: 0.9262650929317596
        f1: 0.9416267282694893
  meiji_old_box_agreement:
    coco_person: 0.9999513571359082
    player_ft: 0.8289230469890068
  player_ft_unmatched: 3517
  player_ft_unmatched_far_proxy: 3489
  player_ft_unmatched_zero_iou: 3354
repro:
  commit: 312f11ec8c7242bef3f3fbeae14ccc0f69d70129
  branch: campaign930/i964-1-player-detection
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i964-1-player-detection/tests/benchmarks/player_detection_comparison.sh
    /home/kamimura/projects/tennis-lab /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/pipeline_switch/i964-detectors-r1-20260929
artifacts:
  run_dir: knowledge/runs/run-i964-detectors-val-meiji-r1-20260929
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790654976098662591_2067170_i964-detectors-val-meiji-r1-20260929.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/pipeline_switch/i964-detectors-r1-20260929
  diagnostic_dir: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/pipeline_switch/i964-detectors-r2-20260929
  video: /home/kamimura/projects/tennis-lab/outputs/player_detection/evaluate/pipeline_switch/i964-detectors-r2-20260929/unmatched/unmatched_old_vs_new.mp4
parents: []
relations: []
papers: []
tags:
- validation
- old-box-bias
- court-roi
- small-far-players
---

## 結果と評価範囲

chat-player-v1 validation 7,371 frame / 14,742 instance（score 0.3、IoU 0.5）で、
FTはmAP50 **0.945261** / precision **0.957506** / recall **0.926265**、COCOは
0.780163 / 0.240015 / 0.931420。主コート選手へのprecisionは大きく上がり、recallは少し下がる。
このvalは#937のepoch選択に使ったsplitなので、未見testの改善とは言えない。
元FTはaugmentation RNG修正前の重みである。checkpoint・JPEG shardの全SHA-256は
[bundleのprovenance](../../runs/run-i964-detectors-val-meiji-r1-20260929/chat_val_provenance.json)に保存した。
COCO `e61688afe3af91b25955e9f9601d04b42068327a2e076ee871828089aa4ffed5`、
FT export `eb8db0ac1b87a0c0a4730534e5dc8107487a53273221ef90231b9351f1d979d4`。

Meijiは**court ROI内**の比較。#933の評価boxは旧COCO trackerが出したboxをレビューしたものなので、
旧検出器と比べると循環しておりCOCO側に有利。保存済みrun-1 JSONの`known_player_recall`は
**旧boxとの一致率**と読む。run-2以降のコードでは`known_player_box_agreement`へ改名した。
全20,558人物・camera・frame単位でCOCO=0.999951、FT=0.828923、対応した選手boxの平均IoUは
0.965048 / 0.820793。非選手boxの一致率は1.0 / 0.147283で、減少は選手検出器の目的に沿う。
これは検出recall、全画面評価、未見testではない。

run-1のlogにclipごとに表示されるmetricsは**累積値**。clip単独のFT旧box一致率は、
clip_000 4,958/5,966、clip_007 3,090/3,775、video_001/clip_001 6,254/7,624、
video_002/clip_013 2,739/3,193。最終値を4 clipそれぞれの指標と扱わない。

## 未一致の分解

run-1のimmutable storeをCPUで読み、元の1対1対応と件数の一致を確認。
[bundleの診断JSON](../../runs/run-i964-detectors-val-meiji-r1-20260929/disagreements.json)に全集計と動画出自を保存した。

| camera | 旧選手box単位 | FT未一致 |
|---|---:|---:|
| cam0 | 6,767 | 2,659 |
| cam1 | 6,927 | 789 |
| cam2 | 6,864 | 69 |

未一致3,517のbest-overlap IoU histogramは `[0,.1):3359, [.1,.2):13, [.2,.3):0,
[.3,.4):34, [.4,.5):109, [.5,1]:2`。IoU=0が3,354。最後の2件は1対1対応の競合。
box高は `<32px:448, 32–64px:2989, 64–128px:62, 128–256px:18`、p10/中央値/p90=31.5/34.6/57.6px。
近側28、遠側3,489、近遠不明0（全母数の不明は424）。近遠は各frameに選手2人がいるときの
box下端y順位による**画像内proxy**で、3D距離の正解ではない。高さは同一人物の重複旧boxのうち最大面積のもの。

12 camera-clipの中央値の未一致frameを中心に2秒ずつ、計24秒の
`outputs/player_detection/evaluate/pipeline_switch/i964-detectors-r2-20260929/unmatched/unmatched_old_vs_new.mp4`
を保存。橙=レビュー済み旧box、緑=FT検出、右に対象領域の拡大。
cam0 clip_000の抽出画像では小さい遠側選手に旧boxだけがある。自動集計だけで全3,517件を
真の見逃しとは断定しないが、全てをboxサイズ差で説明することもできない。

## 判断と次の実験

ユーザー指定どおり#937を既定にする。追跡の選択は未実施で、検出coverageの低下を
良いIDF1/pair F1で隠さない。共通検出に対する複数trackerを開発clipで比較し、
小さい遠側人物の欠落、ID switch、断片化、対象外への反応を併記する。
新しいclipの一回限りの未見評価は調整後まで実行しない。TensorBoard曲線は推論比較のため対象外。

## run-2のデータ管理

評価ラベルを各Meiji clipの`annotations/player_association/{labels.json,review.yaml}`へ移動し、
boxと人物IDの不変を検査。旧新hashと保存先は[bundleの移動記録](../../runs/run-i964-detectors-val-meiji-r1-20260929/label_migration.json)。
旧DINO cache4 rootは全227 clipでv2 scene完成markerが欠落しSLCS readerが拒否したため、
ユーザー承認に従って削除。[全件の検査記録](../../runs/run-i964-detectors-val-meiji-r1-20260929/dino_cache_audit.json)と
[#931への報告](https://github.com/Motoki0705/tennis-lab/issues/931#issuecomment-5886136703)を参照。
