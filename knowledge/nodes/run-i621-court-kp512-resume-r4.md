---
id: run-i621-court-kp512-resume-r4
type: run
title: court kp512 学習再開（epoch 21 deploy）
issue: 621
provider: claude
date: '2026-07-08'
status: done
config:
  loss: kp
  data: court_kp
metrics:
  val_mean_dist_px_best: 2.23
  test_mean_dist_px: 1.708886
repro:
  commit: 41b318b39f757e09897964a32f28148a777b87f5
  branch: feat/issue-618-ball-subpixel-retrain
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data=court_kp loss=kp model.num_classes=14
    data.augmentation.train_scales=[512] data.augmentation.val_short_side=512 data.num_workers=2
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/mean_dist training.early_stopping.mode=min
    run.resume=outputs/court_detection/kp/logs/version_0/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i621-court-kp512-resume-r4
  predictions: knowledge/runs/run-i621-court-kp512-resume-r4/pred_test.npz
  log: .training_queue/logs/1783481398399240801_20792_i621_court_kp512_resume_r4.log
  output_dir: outputs/court_detection/kp/logs/version_1
  checkpoint: ckpt/court_detection/run-i621-court-kp512-resume-r4-epoch21.ckpt
  curves: knowledge/runs/run-i621-court-kp512-resume-r4/curves.png
  tb_logdir: outputs/court_detection/kp/logs/version_1
parents: []
relations: []
tags:
- court_detection
- kp
- resume
- deployed
---

## 考察 / Findings

### 要約
7/5 に途中停止していた court kp512 学習（best ep17, val/mean_dist 13.45px）を
`version_0/checkpoints/last.ckpt` (ep19) から再開したところ、**2 エポックで 13.45 → 2.23px（ep21）** に急改善。
ep22-27 は 2.3-2.6px で頭打ちのため意図的に打ち切り。best ckpt (ep21) を
`ckpt/court_detection/run-i621-court-kp512-resume-r4-epoch21.ckpt` にデプロイし、pipeline court_kp の既定を
`mode: model` に切替（PR #623）。

### アーキテクチャ詳細
元 run（outputs/court_detection/kp/logs/version_0、hierarchical KP14, 512 入力,
val/mean_dist 監視）の full-state resume。コード差分が本質: 元 run は 7/5 コード、
再開は **PR #613（court KP aug の constrained-retry visibility 修正）適用後**のコードで
学習した。ハイパーパラメータ・データは同一（model.num_classes=14 の明示が必要に
なった点のみ config ドリフト）。

### メトリクスの解釈
val/mean_dist（512x904 入力空間 px）: ep20 2.38 → ep21 **2.23**（best）→ ep22-27
2.3-2.6 で plateau。640x360 の tennis_clip 換算で約 1.1px。run 打ち切り時には test が
未実施だったため、登録時に ep21 checkpoint を固定して 2,211 サンプルを再評価し、
`test/mean_dist=1.708886px` と予測配列を保存した。このタスクの `test_dataloader` は
`data_val.json` を使うため、独立した held-out test ではなく再現可能な validation 再評価値である。

### アーキテクチャ⇄メトリクスの因果考察
1 エポックで 13.45 → 2.38px という不連続な改善は、通常の継続学習では説明しにくい。
最有力仮説は **#613 の augmentation visibility 修正が学習信号の質を変えた**こと
（旧コードでは aug 後に画面外へ出た KP の可視性/ターゲットが壊れており、
near 側コーナー(画面端)の学習を阻害していた — I1 で観測した「near 側だけ 19-29px」
の構造誤差とも整合する）。仮説であり、コード起因の切り分け実験は未実施。

### 既存実験との比較
- ep17 ckpt（旧 best）: val 80 枚で subpixel+homography 10.13px → **ep21 では 1.50px**。
  tennis_clip 手動 GT 比 5.91px → **0.76px**（後処理成功 80/80）。
- E2E（court のみ差し替え、ball/gvhmr/assoc 固定）: BLCS 3D は手動コートを
  全指標で上回り（jerk_mean 6014→5979、az_med −6.43→−6.74）、PLCS 位置差は
  median 6.9cm / p95 11.8cm、速度スパイク同数 — **全自動が手動クリックを超えた**。

### 次に有効な実験
- #613 効果の切り分け（旧コードで同 resume を行う対照）— 優先度低（デプロイ判断には不要）。
- DINOv3+DPT court（Colab, issue 未付番）と kp512 の比較。SSL backbone 検証と合流。
- 後処理パラメータ（min_score/ransac 閾値）は ep21 の score 分布で再スイープ推奨
  （PR #623 の残課題）。
