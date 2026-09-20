---
id: run-court-hybrid-downstream-migration
type: run
title: KP＋LINE共通推論とcamera_view_v2下流移行のCPU検証
provider: codex
date: '2026-09-20'
status: done
config:
  model: court_hierarchical_dinov3_vitb16_lora_r8_residual_dense_heads
  loss: trimmed_KP_max8_plus_bidirectional_LINE_inference
  data: 4_external_photos_and_B00_B03_saved_GS_renders
  short_side: 256
  checkpoint_sha256: dd3a396841097e60ff1bc0eabcf7b911e97685e251bf8cc441c100b17276e816
  device: cpu
metrics:
  inputs: 8
  homography_accepted: 3
  photos_accepted: 2
  photos_count: 4
  renders_accepted: 1
  renders_count: 4
  max_selected_kp: 4
  line_adapter_max_abs_difference: 0.0
repro:
  commit: e002a6a801b2cffd654e54a0cd790e5ae973adeb
  branch: codex/court-hybrid-inference
  command: bash knowledge/runs/run-court-hybrid-downstream-migration/repro.sh /tmp/court-hybrid-new-audit
artifacts:
  run_dir: knowledge/runs/run-court-hybrid-downstream-migration
  log: knowledge/runs/run-court-hybrid-downstream-migration/metrics.json
parents:
- run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1
tags:
- court-detection
- hybrid-homography
- downstream
- camera-view
- inference-audit
task: court_detection
sequence: 31
recorded_at: '2026-09-21'
---

## 考察 / Findings

### 要約
指定epoch17を共通predictorへstrict loadし、写真4枚・保存済み3DGSレンダリング4枚で実推論した。H採用は写真2/4、render1/4であり、このcheckpointへの移行を「完全なホモグラフィ推論」とは評価しない。採用点は各成功例で4点、棄却例はゼロ座標＋全不可視として下流へ渡す。B00〜B03は再publicationしていない。

### アーキテクチャ詳細
学習runを親ノードとして参照し、短辺256、DINOv3 ViT-B/16＋LoRA r8、residual dense heads（KP14・SEG7・binary LINE1）＋pose10を保存構成から復元した。648個の全model tensorをstrict loadする。KPの信頼度順PROSACと4点候補をLINEで比較し、幾何・LINEゲートを通った最大8点だけを共同最適化のKP項に入れる。ホモグラフィはcheckpointのcamera-view順の規格コートmetre座標から原画像pixelへの写像である。推定失敗時にraw KPや拒否Hへ切り替えない。

### メトリクスの解釈
LINE支持はモデル内部の整合率で、正解annotationに対する精度ではない。表のKPは0始まりの採用index、支持はforward / reverse、時間はCPUのforward＋幾何処理である。閾値・探索予算は全画像で共通とし、この8画像に合わせた調整は行っていない。各リンクは実RGB・raw LINE・hybrid推定の比較画像。

| 入力 | H状態 | 採用KP | LINE支持 | 秒 |
|---|---|---|---|---|
| [01_ennai011_1](../../runs/run-court-hybrid-downstream-migration/figures/01_ennai011_1.png) | ok | 1,4,11,13 | 0.806 / 0.694 | 2.507 |
| [02_images_1_](../../runs/run-court-hybrid-downstream-migration/figures/02_images_1_.png) | ok | 0,2,5,13 | 0.747 / 0.985 | 1.998 |
| [03_images_2_](../../runs/run-court-hybrid-downstream-migration/figures/03_images_2_.png) | joint_optimization_failed | — | — | 1.431 |
| [04_images](../../runs/run-court-hybrid-downstream-migration/figures/04_images.png) | joint_optimization_failed | — | — | 2.061 |
| [05_B00_000720](../../runs/run-court-hybrid-downstream-migration/figures/05_B00_000720.png) | joint_optimization_failed | — | — | 2.388 |
| [06_B01_001403](../../runs/run-court-hybrid-downstream-migration/figures/06_B01_001403.png) | joint_optimization_failed | — | — | 1.196 |
| [07_B02_000808](../../runs/run-court-hybrid-downstream-migration/figures/07_B02_000808.png) | ok | 5,7,10,13 | 0.706 / 0.998 | 1.606 |
| [08_B03_001112](../../runs/run-court-hybrid-downstream-migration/figures/08_B03_001112.png) | no_jointly_supported_candidate | — | — | 0.846 |

[metrics.json](../../runs/run-court-hybrid-downstream-migration/metrics.json)にcheckpoint・入力・コードのSHA、全raw score、候補数、失敗理由、B00〜B03のowner metadata 16ファイルの前後一致を保存した。native LINE mapとraw/fitted KPは各NPZに保存している。これは推論検証なので新たな学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
5例は共同最適化または支持候補の判定で棄却された。LINE-only adapterの確率gridは共通hybrid APIのraw LINEと完全一致（最大絶対差0.0、256×384）し、合成アライメントへHを描き直したmaskを渡していない。[adapter記録](../../runs/run-court-hybrid-downstream-migration/alignment-adapter.json)を参照。失敗原因を学習不足やSfM driftと断定するGTはこのrunにはない。

### 既存実験との比較
親runのKP/LINE教師評価と今回のH採用率は別指標で、数値を直接比較できない。幾何実装は論文branchの`5b9f8d1d497d2be71dea4cccf62c1c15976c2eb4`から移植したが、その評価用checkpointとは異なる。今回の重みは指定sourceから`ckpt/court_detection/hybrid/`へ実コピーし、SHA-256の前後一致を確認した。

下流はユーザー指定に従いcamera_view_v2へ移行した。既定PLCS `real-rgb-meiji-foot-e60-v1.ckpt`（epoch57）とBLCS `real-rgb-meiji-e60-v1.ckpt`（epoch58）はcheckpoint本文の契約が一致する。CPUで実load・controlled tensor forwardを実施し、有限出力と不可視点・全不可視フレームのvisibility伝播を確認した。[downstream-smoke.json](../../runs/run-court-hybrid-downstream-migration/downstream-smoke.json)にSHAと結果がある。これは実動画全pipelineの精度評価ではない。重み選定は各既存runのvalidation最良保存結果に基づき、Meiji以外の汎化は未検証である。

### 次に有効な実験
複数会場の人手正解Hで採用率と誤採用率を評価し、LINEの誤検出・幾何候補・最適化の寄与を分離する。設定変更の評価にはこの8枚と独立なholdoutを使う。既存B00/B01/B03のheatmap v2を現行v3 loaderが読めない制限は今回も残る。保存成果物の形式やcheckpoint履歴は書き換えていない。

推論時のcommitはrebase前の`e002a6a8`。最新mainへの載せ替え後の同一推論実装は`d3fd5f80`であり、`metrics.json`の推論コードSHAでも照合できる。validator対応では保存入力契約とsemantic可視化設定を変更し、幾何推論の測定結果は書き換えていない。

PR #901のmain取り込み時に、下流の最大8点制限を推論APIとpipeline設定の検証へ移した。論文の共有幾何コードは保存証拠のハッシュと一致する版を保持する。このrunの`metrics.json`のコードSHAは移動前の測定版の記録であり、保存済みの推論出力・測定値は変更していない。
