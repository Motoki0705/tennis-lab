---
task: plcs
sequence: 98
recorded_at: 2026-09-14
date_source: experiment_date
papers: []
id: run-plcs-foot-residual-hard-ft20
type: run
title: 幾何トークンと足元残差・難例重点samplingの追加学習
provider: codex
session: 01a09b38-6993-7e23-9122-9482895fe0b4
date: '2026-09-14'
status: done
config:
  model: plcs_multiview_axial_foot_residual; geometry 77->512->512; hidden512; pos/rot
    trunks6
  loss: position=1, rotation=1, canonical/reprojection=0; same as baseline
  data: single_object_camera_view_v2; train-only 0.5 natural + 0.5 top20% prior-error
    scene sampling
  training: epoch19 init, lr=5e-5, batch4, max_epochs20, early_stop_patience5, bf16-mixed
metrics:
  position_error_m: 0.201928
  angular_error_deg: 5.15625
  position_accuracy_0.5m: 0.945438
  angle_accuracy_15deg: 0.978653
  canonical_mpjpe_m: 0.530381
  canonical_pck_0.1m: 0.012877
  best_val_position_error_m: 0.17953810095787048
  paired_best_position_error_m: 0.200130830547628
  paired_best_position_xy_error_m: 0.18665872014813445
  paired_best_angular_error_deg: 6.001400470733643
  paired_best_position_accuracy_0.5m: 0.9546749749498998
  real_root_reprojection_px: 38.69087074866504
  hard_ge_1m_xy_error_m: 0.8736667876709351
  hard_ge_2m_xy_error_m: 1.254946228855298
repro:
  commit: b8ccc6bac5beac082b97e94c8b42edd9da8d2f40
  branch: experiments/plcs-foot-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python -m src.tasks.plcs.scripts.train
    --config-path /home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-foot-residual/outputs/plcs/foot_residual
    --config-name train
artifacts:
  run_dir: knowledge/runs/run-plcs-foot-residual-hard-ft20
  predictions: knowledge/runs/run-plcs-foot-residual-hard-ft20/pred_test.npz
  output_dir: outputs/plcs/foot_residual/training/logs/version_0
  curves: knowledge/runs/run-plcs-foot-residual-hard-ft20/curves.png
  tb_logdir: outputs/plcs/foot_residual/training/logs/version_0
parents:
- run-plcs-foot-baseline-epoch19-eval
relations:
- to: run-plcs-foot-baseline-epoch29-eval
  rel: compares
tags:
- plcs
- foot-residual
- geometry-embedding
- hard-sampling
---

## 考察 / Findings

### 要約
幾何特徴の埋め込み、足首のコート面疑似位置を基準にしたXYZ残差出力、疑似位置誤差の大きいtrainシーンの重点抽出を実装し追加学習した。合成test全体・指定実クリップの再投影は改善したが、疑似位置誤差1m以上の集計では既存最良モデルより悪化した。ゼロ残差への崩壊は観測されなかったものの、難例の精度改善は未達である。

### アーキテクチャ詳細
既存の部位別UV埋め込みに、体中心相対XY34・human可視性17・court可視性14・足首UV2・body extent2・view別ground XY2・validity1・融合anchor XYZ3・view間差分2の77特徴をMLPで加える。位置と回転の6層ずつのaxial trunkを維持する。

観測courtの対応14点からimage→ground homographyを推定し、可視足首の平均UVを投影する。有効viewのground XYを平均しZ=0とする。出力はanchor + XYZ残差で、Z残差にはrootの高さも含まれる。rank・fit残差・分母・範囲で退化を検出し、無効priorは0と明示的validityで表す。GTは推論入力に使わない。

既存epoch19の互換重み223個を転用し、位置/補助headの最終層と追加geometry MLPの最終層をゼロ初期化。lr=5e-5、batch4、bf16 mixed、warmup250、最大20epoch、patience5。位置/回転lossは元と同じでcanonical/reprojection lossは0。canonical headの精度は今回の成果に含めない。

train 7998シーンを最大64等間隔フレーム・4viewのclean観測で事前評価し、平均水平prior誤差上位20%（閾値0.617641m）から50%、元分布から50%を復元抽出する。難シーンの期待割合は20%→60%、seed固定の事前抽出例では61.14%。誤差1m以上のフレーム割合の推定は8.26%→22.80%。これらは学習epoch中の実測ではなく、実際の3–4view選択・128frame crop・augmentationとは分布が異なる。val/testは変更していない。重み・全scene統計・設定をrun bundleに保存した。

### メトリクスの解釈
10epochで早期停止。validation位置誤差は0.491→0.325→0.299→0.225→0.180mと低下し、その後0.186/0.246/0.208/0.218/0.191m。採用は最小値のzero-based epoch4。train誤差は0.901→0.320mへ下がり続け、後半のval改善は停止した。`curves.png` に推移を保存。

frontmatterの無接頭辞metricsと `pred_test.npz` / `metrics.json` はtraining runnerによる**最終epoch9、bf16**の公式test（3D 0.201928m、yaw5.15625°）。`paired_best_*` と `residual_test.npz` / `residual_test_metrics.json` は**val最良epoch4、float32**で比較専用に再実行した結果（3D0.200131m、yaw6.001400°）。異なるcheckpoint・推論条件の指標を混同しない。選択はvalidationだけで行い、testや指定実映像を選択に使っていない。

疑似位置誤差1m以上の6796フレームでは平均水平補正2.785005m、補正5cm未満は0.0147%。2m以上の3463フレームでは補正平均4.577466m、5cm未満0%。ゼロ残差崩壊ではない。一方2m以上では必要な補正を取り切れず水平誤差1.254946mが残る。全testの5cm未満補正は2.58%。詳細は `paired_statistics.json`。

### アーキテクチャ⇄メトリクスの因果考察
仮説: image UVからコート座標への幾何変換を既知処理に分け、モデルが人体姿勢・観測不一致に応じた補正へ容量を使えたことが、全体誤差改善に寄与した。ただし幾何埋め込み・残差出力・重点抽出・追加学習を同時に変更したため個別の寄与は確定できない。

仮説: homographyは足首が地面より高いと奥行き誤差を拡大し、単純なview平均が大きな外れ値を残す。難例の抽出を増やしても、scene平均の順位だけではフレーム内の極端例を十分学習できず、priorへの依存が残る可能性がある。ジャンプや遮蔽のラベル別解析は今回しておらず、この機序を実証したわけではない。

実クリップのepoch0中間評価は16.906px、採用epoch4は38.691px（中間CPU計算38.680px）。合成validation改善と実映像整合性は一致しなかった。単一の実映像に合わせたcheckpoint選択はせず、今後は独立した実検証セットが必要。

### 既存実験との比較
主比較は指定configの既存最良epoch29。同じ998シーン・127744フレーム・GT・reference・seed・GPU float32を使用し、target/rotation/scene順/anchorの完全一致を確認した。比較表と条件・成果物は `group-plcs-foot-residual` に集約。実クリップでは保存済み2D検出・人物対応・手動courtとGVHMR/SMPLを固定し、本番PLCS predictorで全1010フレームを推論した。新位置/回転を標準SceneResultへ統合し、保存後の読戻し一致も検証した。

### 次に有効な実験
1. viewごとの幾何条件・足首高さの手掛かりを用いた信頼度付き融合と、直接位置headとの明示的gateを比較し、2m以上のprior誤差層を重点評価する。
2. scene平均だけでなくframe/windowのprior誤差binで抽出し、1–2mと2m以上を別々に管理する。今回のtestを学習側へ移さず、新たなtrain/val側の統計から作る。
3. 同じ初期値・学習量で「埋め込みのみ」「残差のみ」「抽出なし」を分離評価する。実データの独立validation（可能なら3D GT、少なくとも複数クリップ）を用意し、合成側だけの過適合を検出する。

再現情報: queue取得時点のcommitとpatchは元のまま保存した。完成コードは `43291cdb`。生成された `train.yaml`・sampling重みと統計もrun bundleに保存した。当時のコマンドとbundleの位置づけは `group-plcs-foot-residual` を参照。GPUでの再実行も共有training queueを使う。

実装検証: 163件の関連CPUテストが成功。射影復元・退化/遮蔽・view融合・ゼロ残差時の厳密prior一致・残差head勾配・checkpoint互換性・重み付き抽出とsplit整合性を含む。ruff/mypyも成功。
