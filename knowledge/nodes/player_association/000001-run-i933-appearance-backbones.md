---
id: run-i933-appearance-backbones
type: run
task: player_association
sequence: 1
recorded_at: '2026-09-27'
title: camera間人物対応の外観backbone比較（chat-player-v1とMeijiラベル）
issue: 933
provider: claude
date: '2026-09-27'
status: done
config:
  encoders: [osnet_ain_x1_0_msmt17, osnet_x1_0_msmt17, clipreid_vitb16_market1501, dinov3_vits16, dinov3_vitb16]
  input_size: [256, 128]
  meiji_sampling: {min_height_px: 48, border_px: 4, max_overlap_iou: 0.1, max_samples: 16}
  meiji_min_purity: 0.9
  chat_sampling: {bbox_source: observed, occluded: false, truncated: false, min_height_px: 48, samples: 16}
  device: cpu
metrics:
  meiji_auc_players: {osnet_ain_x1_0_msmt17: 0.870, osnet_x1_0_msmt17: 0.723, clipreid_vitb16_market1501: 0.926, dinov3_vits16: 0.418, dinov3_vitb16: 0.413}
  meiji_top1: {osnet_ain_x1_0_msmt17: 0.814, osnet_x1_0_msmt17: 0.674, clipreid_vitb16_market1501: 0.930, dinov3_vits16: 0.302, dinov3_vitb16: 0.326}
  chat_test_auc: {osnet_ain_x1_0_msmt17: 1.000, osnet_x1_0_msmt17: 0.999, clipreid_vitb16_market1501: 1.000, dinov3_vits16: 0.997, dinov3_vitb16: 0.998}
  chat_test_rank1: {osnet_ain_x1_0_msmt17: 1.000, osnet_x1_0_msmt17: 0.958, clipreid_vitb16_market1501: 1.000, dinov3_vits16: 0.917, dinov3_vitb16: 0.917}
repro:
  commit: 499a1be5
  command: >-
    PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_appearance.py --repo $R --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset
    --observe $R/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927 --labels-dir tests/benchmarks/labels/player_association/meiji_3cam
    --report $R/outputs/player_association/evaluate/meiji_appearance/i933-appearance-v1-20260927;
    PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_chat_reid.py --repo $R --store $R/data/player_detection/chat-player-v1
    --report $R/outputs/player_association/evaluate/chat_reid/i933-chat-reid-v1-20260927
artifacts:
  run_dir: knowledge/runs/run-i933-appearance-backbones
  meiji: knowledge/runs/run-i933-appearance-backbones/appearance.json
  chat: knowledge/runs/run-i933-appearance-backbones/chat_reid.json
parents: []
relations:
- {to: run-scene-component-meiji-idstitch-20260925, rel: compares}
papers: []
tags: [player_association, reid, backbone_comparison, real_clip]
---

## 要約

camera間の人物対応に使う外観特徴を、学習なし（公開重みのまま）の5候補で比較した。
Meiji の人手ラベル（#933、4 clip・3 camera）の camera 間 track 対応では **CLIP-ReID（ViT-B/16、Market-1501）が最良**（AUC 0.926、top-1 0.930）で、
OSNet-AIN（0.870 / 0.814）が次点。DINOv3 の CLS は Meiji で偶然以下（AUC 0.41）だった。
chat-player-v1（単視点の放送映像）はどの候補もほぼ飽和（test AUC 0.997〜1.000）し、候補の差を測れなかった。

## 評価の条件

- 重み: OSNet 系は著者の公開重み（Hugging Face `kaiyangzhou/osnet`、MIT）、CLIP-ReID は `occurra/person_vit_clip_reid` の PyTorch 版（CLIP-ReID の Market-1501 学習済み、MIT）、DINOv3 は repo 既存の `third_party/dinov3/checkpoints`。
  重みは `ckpt/player_association/` に置いた（sha256 は PR に記載）。SOLIDER-REID と KPR/PRTreID は比較していない（下の「比較しなかった候補」）。
- Meiji: 観測 run `i933-observe-v1-20260927` の track をラベルと IoU≥0.5 で照合し、90% 以上が同じ人物の track だけを使う。
  track ごとに、観測済み・高さ 48 px 以上・画面端に接しない・他の box と IoU 0.1 以下の box から最大 16 frame を等間隔に取り、埋め込みの平均を track の外観とした。
  指標は camera の異なる track の組の cosine で、`auc_players` = 同じ選手 vs 別の選手、`top1` = 選手 track ごと・他 camera ごとに最も似た track（非選手を含む）が同じ選手である割合。組は同じ選手 25、別の選手 25。
- chat-player-v1: `observed` かつ遮蔽・見切れなしの box だけ。track を前半・後半に分け、前半が自分の後半を当てる rank-1 と、正例（同じ track の前後半）vs 負例（同じ clip の別 track、別 source の track）の AUC。同じ source の別 clip は同一人物の可能性があるので組にしない。val・test とも 24 track。

## メトリクスの解釈

| encoder | Meiji AUC（選手） | Meiji AUC（非選手込み） | Meiji top-1 | chat val AUC | chat test AUC | chat test rank-1 |
|---|---|---|---|---|---|---|
| osnet_ain_x1_0_msmt17 | 0.870 | 0.857 | 0.814 (35/43) | 0.993 | 1.000 | 1.000 |
| osnet_x1_0_msmt17 | 0.723 | 0.696 | 0.674 (29/43) | 0.998 | 0.999 | 0.958 |
| clipreid_vitb16_market1501 | **0.926** | **0.889** | **0.930 (40/43)** | 0.995 | 1.000 | 1.000 |
| dinov3_vits16 | 0.418 | 0.487 | 0.302 (13/43) | 0.963 | 0.997 | 0.917 |
| dinov3_vitb16 | 0.413 | 0.529 | 0.326 (14/43) | 0.964 | 0.998 | 0.917 |

- 観測: chat-player-v1 は試合ごとにウェアが大きく異なり、放送の大きな crop なので、どの候補でも容易だった。本 issue の難しさ（同じクラブの似たウェア、camera ごとに大きく違う見え方・解像度）は Meiji にしか無い。
- 観測: Meiji で CLIP-ReID は同じ選手の組の cosine が 0.78〜0.97、別の選手の組の最大が 0.835 で、分布は重なる。top-1 の失敗 3 件は clip_007（1件）と clip_001（2件）。
- 観測: cam0 の遠い選手（box 高さ 約 35 px）は高さ 48 px 未満で標本が0になり、外観を持たない。この track の対応は幾何に頼るしかない。
- 仮説: DINOv3 の CLS は、背景（コートの色・フェンス）と姿勢の類似を拾い、人物の同一性を表さない。Re-ID 用の学習なしでは使えない。
- chat-player-v1 の val で当てはめた `sigmoid((cos - c0) / tau)` は CLIP-ReID で c0 = 0.897。Meiji では同じ選手の組の多くがこれより低いので、単視点のデータで決めた閾値は camera 間には移らない。

## 既存実験との比較

PLCS の pose-only Re-ID（[run-scene-component-meiji-idstitch-20260925](../tennis_scene/000019-run-scene-component-meiji-idstitch-20260925.md)）は、Meiji clip_000 で同じ camera 内の別人の cosine が 0.967/0.850 に潰れ、cam1 の2人を入れ替えた。
RGB の Re-ID 重みを使えば、同じ clip_000 で CLIP-ReID は 8 組中 8 組、OSNet-AIN は 5 組の top-1 が正しい。

## 比較しなかった候補

- SOLIDER-REID: Re-ID で fine-tune した重みの配布が Google Drive / Baidu で、無人実行では取得を確認できなかった。Hugging Face には事前学習の backbone だけがある。
- KPR / PRTreID: keypoint prompt と専用の依存（torchreid の fork）が必要で、移植の手間が大きい。
- 判断は #933 の【要判断】に記録した。

## 次に有効な実験

- 採用: CLIP-ReID を既定の外観 encoder にする。camera 間の類似度の閾値は単視点のデータから移らないので、ラベルの無い Meiji clip で幾何から作った擬似ラベル（cam0–cam1 の足元距離が小さい組）で当てはめるか、clip 内の相対的な順位で使う。
- 外観と幾何を統合した対応付け（`cluster_multiview`）を、同じラベルで pair F1・group accuracy として評価する。
- 余力があれば、chat-player-v1 の train で CLIP-ReID を fine-tune し、Meiji で効果を測る（学習は training queue）。
