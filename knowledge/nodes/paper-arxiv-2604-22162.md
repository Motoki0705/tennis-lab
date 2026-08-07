---
id: paper-arxiv-2604-22162
type: paper
title: "SAMIDARE: Advanced Tracking-by-Segmentation for Dense Scenarios"
curator: chatgpt-schedule
date: 2026-08-07
status: reviewed
external_ids:
  doi: null
  arxiv: "2604.22162"
  openreview: null
published_at: 2026-04-30
reviewed_at: 2026-08-07
evidence_level: fulltext-code
tasks: [multi_object_tracking]
repo_paths:
  - src/submodules/models/tracker/yolo_tracker.py
  - src/submodules/models/tracker/common.py
  - src/tennis_scene/pipeline/components/player_association.py
sources:
  - kind: paper
    url: https://openaccess.thecvf.com/content/CVPR2026W/CVsports/html/Hirano_SAMIDARE_Advanced_Tracking-by-Segmentation_for_Dense_Scenarios_CVPRW_2026_paper.html
  - kind: code
    url: https://github.com/ZabuZabuZabu/SAMIDARE
relations: []
tags: [literature, tracking, segmentation, sports]
---

## 要約

SAM2MOTへ密度依存のmask再生成、信頼frameだけを使うmemory更新、track状態を考慮したassociationを加え、遮蔽とframe-outが多いsports MOTでID維持を改善する。

## 主要な主張と根拠

著者はSportsMOT validationで基盤法よりHOTAを2.5、IDF1を4.2 point改善したと報告する。CVPRW公式本文と公式実装を確認した。結果は密集sports sceneを中心とし、通常2選手のtennis条件は直接評価されていない。

## tennis-labへの適用可能性

現行trackerにはmask memoryの選択的更新とstate-aware associationがない。選手交差、周辺人物との近接、frame-out復帰を固定clipで比較し、IDF1・HOTA・ID switchと計算資源を同時に測る候補になる。

## 制約・失敗条件

SAM2依存により遅延とVRAMが増える。小さい遠景人物ではmask driftが誤IDを長時間保持し得る。IDF1改善が1 point未満、ID switch増加、または資源2倍超なら導入根拠は弱い。

## コード・データ・ライセンス

公式codeを確認したが、レビュー時点でrepository rootの明示LICENSEを確認できなかった。直接取り込み前に利用条件とSAM2等の上流licenseを確認する。
