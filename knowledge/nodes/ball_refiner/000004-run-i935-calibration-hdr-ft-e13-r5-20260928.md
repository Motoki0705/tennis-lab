---
id: run-i935-calibration-hdr-ft-e13-r5-20260928
type: run
task: ball_refiner
sequence: 4
recorded_at: '2026-09-28'
title: '文脈なしMDNの未較正validation診断: 裾のcoverage不足と点精度の退行'
issue: 935
provider: codex
session: 01a0e7ce-fb4f-7891-bdd9-2fd526d5b391
date: '2026-09-28'
status: done
config:
  training_run: run-i935-detector-only-ft-e13-s42-r4-20260928
  best_epoch: 9
  partition: calibration
  covariance_scale: 1.0
  hdr_samples_each: 2048
  hdr_levels:
  - 0.5
  - 0.9
  - 0.95
  mc_seed: 1729
  bootstrap_repetitions: 2000
  bootstrap_unit: meiji/video/clip (all cameras)
  bootstrap_confidence: 0.95
metrics:
  observed/area_px2_mass_0.9: 77283.46423014154
  observed/coverage_mass_0.5: 0.48989061982101423
  observed/coverage_mass_0.9: 0.8126450116009281
  observed/coverage_mass_0.95: 0.8594630427577064
  observed/mean_error_px: 135.49748920716215
  observed/median_error_px: 46.17137336730957
  observed/p95_error_px: 568.0451568603515
  observed/position_nll_px: 10.918557383265762
  observed/recall_20px: 0.32905203844879016
  evidence_gap/area_px2_mass_0.9: 125970.66762141643
  evidence_gap/coverage_mass_0.5: 0.46555727554179566
  evidence_gap/coverage_mass_0.9: 0.8080495356037152
  evidence_gap/coverage_mass_0.95: 0.8591331269349846
  evidence_gap/mean_error_px: 156.13453637681692
  evidence_gap/median_error_px: 91.60045623779297
  evidence_gap/p95_error_px: 522.3872192382812
  evidence_gap/position_nll_px: 12.209332143928245
  evidence_gap/recall_20px: 0.04063467492260062
repro:
  commit: 7195911befde371c39f375981cb637bd117f5861
  branch: campaign930/i935-5-evaluation
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-5-evaluation
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-5-evaluation/.venv/bin/python
    -u -m src.tasks.ball_refiner.scripts.evaluate_pilot paths.artifact_root=/home/kamimura/projects/tennis-lab/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs evaluate.training_run=ball_refiner/train/detector_only/i935-ft-e13-s42-r4-20260928
    evaluate.partition=calibration run.output_dir=ball_refiner/evaluate/detector_only/i935-ft-e13-calibration-r5-20260928
artifacts:
  run_dir: knowledge/runs/run-i935-calibration-hdr-ft-e13-r5-20260928
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790597271266337994_560715_i935-calibration-hdr-ft-e13-r5-20260928.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/detector_only/i935-ft-e13-calibration-r5-20260928
  predictions: knowledge/runs/run-i935-calibration-hdr-ft-e13-r5-20260928/predictions
  audit: knowledge/runs/run-i935-calibration-hdr-ft-e13-r5-20260928/evaluation_audit.json
parents:
- run-i935-detector-only-ft-e13-s42-r4-20260928
relations: []
papers: []
tags:
- calibration-diagnostic
- validation-only
- gmm-hdr
- evidence-gap
- negative-result
---

## 結論と比較条件

文脈なし時間MDNは較正用validationでもdetectorの平均・裾誤差を減らす一方、
中央値と20px recallを悪化させた。未較正GMMの90/95%領域はobserved・人工欠損の両方で
名目coverageを下回る。生成分布をそのまま較正済みとして下流へ渡す根拠は得られていない。
checkpointとdetector deployの判断は維持し、この診断をtestの採否や補正後の性能とは呼ばない。

[学習run](000003-run-i935-detector-only-ft-e13-s42-r4-20260928.md)のbest epoch 9（2,500更新）を固定した。
Meiji video_000の較正側6時刻clip群×3camera、計18 camera-clip。
位置教師はobservedだけで、全観測12,068 frameと人工欠損内2,584 frameを採点した。
checkpoint選択に使った6群とは重ならないが、同じvideoなので独立収録のtestではない。
pose/courtは無効。gapは検出器の候補・patch証拠だけを消し、RGBは加工していない。

全GMMの50/90/95% HDRはR²上の条件付き領域。閾値用と面積用に独立した各2,048 sample/frame、
seed 1729からclip/条件ごとのseedを固定した。区間は時刻clip群を全cameraごと復元抽出する
2,000反復・frame加重のpercentile bootstrap 95%区間。独立群は6つだけで探索的な区間であり、
MC閾値推定の誤差は含めない。面積のMC標準誤差も推定閾値に条件付けたframeごとの値である。

## 観測frameの点推定

refinerの点は最大重み成分の平均。全位置既知frameを採点し、detector scoreと存在確率で除外しない。
同じframeでgate前detector argmaxと比較した。

| 指標 | detector | 未較正MDN |
|---|---:|---:|
| mean px | 187.259 | 135.497 |
| median px | 22.405 | 46.171 |
| p95 px | 649.956 | 568.045 |
| 20px recall | 49.44% | 32.91% |

paired mean error差（MDN−detector）は−51.761px、95%区間 [−79.744, −12.678]。
平均の改善と細かい位置合わせの退行が同時にある。NLLで学習した大きな混合分布と
最大重み成分の点化の関係は未切り分けであり、原因を時間平滑化と断定しない。
選択側validationの絶対値と異なるのは評価群が違うためで、学習の再実行による差ではない。

## 分布のcoverageと広がり

| 条件 | frame数 | NLL px² | 50% coverage | 90% coverage [95%区間] | 95% coverage [95%区間] | 90%面積 px² |
|---|---:|---:|---:|---:|---:|---:|
| 全observed | 12,068 | 10.919 | 48.99% | 81.26% [71.94,89.32] | 85.95% [76.90,93.25] | 77,283 |
| 人工証拠欠損内 | 2,584 | 12.209 | 46.56% | 80.80% [71.83,88.43] | 85.91% [78.62,92.37] | 125,971 |

NLLはsource画素座標の密度を使い、uv²単位ではobserved −3.625、gap −2.334。
欠損時に領域は広がるが、90/95%の裾のcoverage不足は残る。50%近辺の値だけから較正良好とは言えない。
areaは画面外tailを含むR²の領域であり、画面内の占有率ではない。

以下は各gap長について**同じframe**を無欠損/欠損で比較した値。
異なるgap長の行は別frame集合なので、長さの因果効果を分離した比較ではない。

| gap frame数 | 採点frame数 | NLL px² 無欠損→欠損 | 90% coverage 無欠損→欠損 | 全混合分散 px² 無欠損→欠損 |
|---|---:|---:|---:|---:|
| 1 | 95 | 10.861→11.862 | 81.05→83.16% | 17,626→22,665 |
| 4 | 388 | 11.306→12.387 | 74.74→73.20% | 20,237→23,871 |
| 8 | 699 | 11.003→12.243 | 83.69→82.26% | 20,715→23,904 |
| 16 | 1,402 | 10.617→12.167 | 85.45→82.03% | 16,620→22,763 |

各層で欠損により分散が増える一方、欠損長の増加に沿った単調な拡大は観察できない。
存在BCEはobserved 0.00903 / gap 0.01069、Brierは0.000160 / 0.000129。
いずれも確定負例0件の正例だけなので、これを不存在も含めた存在較正の証拠にしない。

## 保存と監査

[再現bundle・生予測](../../runs/run-i935-calibration-hdr-ft-e13-r5-20260928)へ
36 NPZ、config、manifest、metrics、queue log、reproを保存した。
[audit](../../runs/run-i935-calibration-hdr-ft-e13-r5-20260928/evaluation_audit.json)と
[CPU監査script](../../runs/run-i935-calibration-hdr-ft-e13-r5-20260928/verify_evaluation.py)で次を確認した。

- 入力checkpoint/学習設定/manifest/cacheのhash、36 NPZのhash、重複・欠落なし、選択/較正群の分離。
- 全frame/PTS/実秒/教師/固定gap長、全GMM契約、元cacheからのdetector点誤差の一致。
- 全10層の点/NLL/coverage/面積/分散とbootstrap区間を再計算し、保存JSONと完全一致。
- 全採点frameの密度を`torch.distributions.MixtureSameFamily`で照合し、HDR包含判定と一致。
- 各NPZの先頭最大32採点frame、計1,151 frameのMC閾値・面積・条件付きSEをCPUで再生成。
  GPU chunk 32に対しCPU chunk 7でも一致した。残りのMC推定値の全再生成はしていない。
- 実clip_010/cam0（270 frame）の両条件をcheckpointからCPU再推論し、成分平均の座標差は最大0.0475px。

新しい学習は実施していないためTensorBoard曲線はない。学習曲線は親runを参照する。
元media/JPEG/注釈の再hash、dense detector密度、最終test、RGB遮蔽、実amodal教師、
文脈ablation、pipeline保存/load-only、3D精度は未検証。

## 次の実験と判断

分散scale等の補正を試す場合はこの未較正結果を固定し、較正側だけで別runとしてfitする。
同じ較正側での補正後coverageを汎化性能とは扱わず、最終testで設定を選び直さない。
点推定の中央値退行は分散scaleでは直らないので、文脈生成と同一母数・同予算のfull/ablation比較も必要。
検出器の密度比較にはnative heatmapを生成し、RGB遮蔽対照・疑似amodal教師の品質検査を別途行う。
README設計のユーザー合意は依然未取得で、issueの【要判断】を継承する。
