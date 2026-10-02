---
id: run-i964-fullframe-selection-r6-20260930
type: run
task: person_tracking
sequence: 5
recorded_at: '2026-09-30'
title: ROI前7条件：COCO .30を暫定推薦、unionはwideとcam0遠側に利点
issue: 964
provider: codex
status: done
config:
  device: cpu
  core_half_width_m: 4.115
  baseline_limit_m: 16.885
  min_dwell_fraction: 0.25
  max_candidates: 6
  observation_spatial_gate: false
metrics:
  source_variants: 7
  camera_clips_per_source: 12
  reference_player_units: 20558
  wide_reference_units: 143
  sources:
    ft_base_0.01:
      player_kept_03: 18323
      player_kept_05: 17967
      player_wide_kept_03: 141
      non_player_excluded_03: 3818
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 1844
      cam0_far_covered_03: 2538
      cam1_far_covered_03: 2699
      cam2_far_covered_03: 3285
    ft_base_0.02:
      player_kept_03: 18021
      player_kept_05: 17767
      player_wide_kept_03: 135
      non_player_excluded_03: 4125
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2151
      cam0_far_covered_03: 1598
      cam1_far_covered_03: 2882
      cam2_far_covered_03: 3364
    ft_base_0.05:
      player_kept_03: 17499
      player_kept_05: 17296
      player_wide_kept_03: 121
      non_player_excluded_03: 4432
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2458
      cam0_far_covered_03: 1310
      cam1_far_covered_03: 3033
      cam2_far_covered_03: 3358
    coco_fullframe_0.05:
      player_kept_03: 14280
      player_kept_05: 14143
      player_wide_kept_03: 49
      non_player_excluded_03: 4452
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2478
      cam0_far_covered_03: 1091
      cam1_far_covered_03: 2000
      cam2_far_covered_03: 1771
    coco_fullframe_0.10:
      player_kept_03: 18045
      player_kept_05: 18019
      player_wide_kept_03: 111
      non_player_excluded_03: 4454
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2480
      cam0_far_covered_03: 2621
      cam1_far_covered_03: 2645
      cam2_far_covered_03: 2804
    coco_fullframe_0.30:
      player_kept_03: 19843
      player_kept_05: 19842
      player_wide_kept_03: 123
      non_player_excluded_03: 4451
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2477
      cam0_far_covered_03: 3107
      cam1_far_covered_03: 3078
      cam2_far_covered_03: 3364
    union_fullframe_0.30:
      player_kept_03: 19787
      player_kept_05: 19684
      player_wide_kept_03: 143
      non_player_excluded_03: 4446
      adjacent_court_excluded_03: 395
      off_court_excluded_03: 2472
      cam0_far_covered_03: 3268
      cam1_far_covered_03: 2919
      cam2_far_covered_03: 3295
  coco_030_pair_f1_iou03: 0.9629686697483307
  union_030_pair_f1_iou03: 0.9553266383683611
artifacts:
  run_dir: knowledge/runs/run-i964-fullframe-selection-r6-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-fullframe-r6-20260929
  video: /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-fullframe-r6-20260929/best_source_review_3cam.mp4
  report: knowledge/runs/run-i964-fullframe-selection-r6-20260930/report.md
parents:
- run-i964-coco-fullframe-r5-20260929
- run-i964-court-selection-r5-20260929
relations: []
papers: []
tags:
- dev-only
- cpu
- pre-roi
- selection
date: '2026-09-30'
repro:
  commit: f83bb2fa
  branch: campaign930/i964-2-tracking
---

## 推薦と適用範囲

**固定した現行の追跡・コート選別に渡すsourceとして、COCO全画面0.30を暫定推薦する。既定変更はしていない。** 全7条件で選手unit保持が最多の19,843/20,558（96.52%）、全8 identityを保持し、既知非選手の残存は3/4,454 unit。4 dev clip全てで既存CLIP対応が決定し、成功全4本のpair TP/FP/FNを合算したF1は0.962969（IoU .3）。1 detectorで済むこともunionに対する利点。これは検出器そのものの一般的な優劣ではなく、今回の共通追跡・固定選別との組み合わせに限った判断である。

**参照boxは旧COCO/旧tracker由来でCOCOに有利。** 旧box一致を検出recallと呼ばず、未ラベル予測をFPとしない。4本とも既存devで、完全な全人物ラベルではない。sideは既存の注釈ballによる判定を固定した。独立な未見・会場・ダブルス・全pipelineでの優位は未確認。学習を含まずTensorBoard曲線は無い。

union（#937≥.30＋COCO≥.30）は有力な代案。wideを143/143、cam0 farを3,268/3,270保持する一方、COCO .30に対してcam1 farは3,078→2,919、cam2 farは3,364→3,295へ低下した。全選手は19,787（56 unit少ない）、既知非選手残存8、合算pair F1は0.955327。unionでは2 detectorの実行が必要で、ここでその壁時計速度差は測っていない。wide・cam0の観測を最優先する場合はunionを選ぶ合理性がある。動画とcamera別表を見たユーザー判断に委ねる。

## 全表と固定規則

[日本語レポート](../../runs/run-i964-fullframe-selection-r6-20260930/report.md)を、camera×near/far・clip・identity・候補/track負荷・対応成功/失敗の入口とする。[comparison.json](../../runs/run-i964-fullframe-selection-r6-20260930/comparison.json)と同名CSVが数値の正本。主表は選別直後、IoU .3、unit=(clip,camera,frame,person)で重複boxをまとめる。選手ID保持はその集計範囲のunitの50%以上、非選手ID完全除外は残存0。非選手は全体除外と追跡hitに条件付けた除外を分ける。全体除外の改善を幾何による除外だけの成果とはしない。near/farは画像内の2選手のbox下端順位で、不明はunknownに残す。

選別規則の実装正本は`src/tasks/person_tracking/court_linking.py`、実行時の全値は[selection_protocol.json](../../runs/run-i964-fullframe-selection-r6-20260930/selection_protocol.json)。singles core（|x|≤4.115m、|y|≤16.885m）内の実観測distinct frameを連結後に25%以上集約し、その後group上限6。**領域はmembershipだけを決め、選択済み断片の全実観測を保存する。** ダブルス幅は診断値。gap/jumpによるidentity区間、安全な短いhandoff、1秒未満の曖昧断片除外は維持した。全84 source camera-clipで、全選択断片の観測が無削除・非観測を追加しない・上限6を確認した。COCO .30では診断corridor外（横幅またはbaseline+5m外）の実観測669件も選択済みのまま保持している。

全sourceは800/1333・ROI前で、同じUltralytics BoT-SORT（追加score gate/fusionなし）を使う。CLIPはCPUで既定on。全raw trackで遮蔽を検査し、coreに入るtrackを同じ条件でsampleする。完全一致するcrop入力と重み/hashだけを再利用する。farの小crop等で特徴が無い区間は明示missing。COCO .30の連結14件のうち両側CLIPありは6件、unionは11件中3件で、外観が常に使えたとは主張しない。対応用group timelineだけhandoff重複を1box/frameにまとめ、元rowを保存する。既存v3の安全策を経て、未決定は未決定として記録した。

## wide修正の効果と残る欠落

ラベルboxをz=0へ投影すると、|x|>5.485mは143/20,558 unit（cam0 far49、cam1 far71、cam2 near23）。参照足元が無効な選手unitは0。このxは推定校正/box下端による値で、独立な3D正解ではない。

前回のtrack/保存CLIPを固定した補助auditでは、wide保持がFT .01で46→141、旧ROI unionで5→123、旧経路で0→123へ回復した。選手全体も18,116→18,323、19,320→19,624、19,444→20,036へ回復し、隣コート395/395除外を維持した。これは空間mask削除の効果確認で、ROI差が残る補助表を今回の公平な比較に混ぜない（`wide_audit_*`）。

新しい公平な比較ではFT .01/.02/.05のwide保持141/135/121、COCO .05/.10/.30は49/111/123、unionは143。COCO .30の20欠落は全てclip_007/cam0の冒頭断片（raw track52、frame[0,165)）で、core121 frameが必要163 frameに届かない。**選択された断片内でwide観測を切ったためではない。** [wide_losses.json](../../runs/run-i964-fullframe-selection-r6-20260930/wide_losses.json)に全条件の原因を事後照合し、再現用`wide_losses.py`を同梱した。

隣コートは全7条件で395/395参照unitが残存0。追跡hitに条件付けてもFT .01/.02、COCO .10/.30、unionは395/395、FT .05は391/391、COCO .05は394/394を除外した。COCO .30にはコート外X1（clip_001/cam1、963–965）の3 unitが残る。動画の追加区間とpreviewで確認できるようにした。ここでは選手BとX1の参照box同士もIoU 0.571/0.504/0.430で重なっている（`overlapping_reference_units.json`）。IoUによる対応の曖昧さがあるため、この3 unitを実際の誤人物選択と断定しない。参照ラベルと集計は事後に修正していない。

## 失敗から分かること

COCO .05は検出段で全20,558参照選手unitに一致していたが、追跡/選別後は14,280へ低下し、保持identityも7/8。累計raw trackは3,930（COCO .30は120）。4/4本で対応が決定しても各clipのpair F1は0.45–0.64で、決定数だけでは選べない。FTは.01→.02→.05でcam0 far保持が2,538→1,598→1,310へ低下した。選別後の全体保持・非選手残存・camera別保持を一緒に見る必要がある。規則の追加調整はこの比較中に行っていない。

COCO .30にも715参照unitの選別欠落、wide20欠落、既知非選手3残存がある。対応後の保持は19,841、pair FPは6、FNは1,436。pair F1は追跡boxに一致したunit間の条件付き指標で、主表の未追跡を含む分母とは違う。モデルに外観/poseを入れた最終追跡を評価した結果ではない。

## 検証・再現と後続

[192 archiveの直接SHA-256検査](../../runs/run-i964-fullframe-selection-r6-20260930/source_tracking_hash_audit.json)（raw24/source84/track84）、両検出重みの実hash、CLIP重み/hash、全選別/対応archiveと保存maskの不変条件を確認。18 focused tests、ruff/mypy/commit hooks、実データ84 camera-clip、全28対応判断を実行した。validatorは指定なしで0回。[run.json](../../runs/run-i964-fullframe-selection-r6-20260930/run.json)にcommit・command、`select-*.log`にCPU完了と最大RSSを保存。新規GPU jobは0、未見評価試行も0。

動画は`outputs/person_tracking/evaluate/court_selection/i964-fullframe-r6-20260929/best_source_review_3cam.mp4`。4clip各最大誤り窓、wide、隣コート、残存非選手を含み、重複する対応失敗窓は重ねて描かない。ラベルは事後の窓選択とoverlayだけに使用する。[最終manifest](../../runs/run-i964-fullframe-selection-r6-20260930/video.json)・previewは同bundleに保存。28秒、1920×820、20fps、全560frame読戻し成功、代表frameを目視確認した。SHA-256は `ea5979d9cad02074c558bd6a1ba1069f902ba6bf36de73c40d0b849355100ed0`。CPU出力と補助auditは合計約123 MB、knowledgeは約11 MBで5 GB以内（詳細`resource_usage.json`）。

ユーザーが既定sourceを選んだ後に、2–3追跡方式と両baseline、BoT-SORT-style derivative、SOLIDER/KPR（KPRは推論portのみ・HL3・全weight hash）を比較する。共通pipeline/#935接続、clip_000全pipeline、調整凍結後一回の未見評価は未完了。#935/#936 branch/worktree、pipeline既定、#937重み/閾値には変更していない。
