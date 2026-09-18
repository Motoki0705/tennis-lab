---
id: run-slcs-ball-gradient-probe-v1
type: run
title: 'SLCS既存重み: 固定train窓のmode差・項別勾配診断'
provider: codex
date: '2026-09-18'
status: done
config:
  checkpoint: slcs/real-rgb-pilot-augmented-e60-v2.ckpt
  data: slcs/real_rgb_pilot_v2; first eligible train window per domain
  augmentation: false
  modes:
  - eval
  - train_dropout
  batch_size: 1
  precision: cpu float32
  seed: 42
  deterministic_algorithms: true
  optimization: false
metrics:
  domains: 2
  train_windows: 2
  optimizer_steps: 0
  model_state_unchanged: true
  broadcast_eval_ball_error_m: 5.202081203460693
  broadcast_train_mode_ball_error_m: 5.211867809295654
  broadcast_eval_ball_smooth: 1.041954237734899e-05
  broadcast_train_mode_ball_smooth: 0.05575551837682724
  broadcast_target_supported_smooth: 0.0027360152453184128
  broadcast_shared_player_over_ball_supervised_grad_norm: 13.866020159327835
  broadcast_shared_smooth_over_ball_supervised_grad_norm: 0.5281708085858934
  meiji_eval_ball_error_m: 6.787822723388672
  meiji_train_mode_ball_error_m: 6.786911487579346
  meiji_eval_ball_smooth: 7.523425301769748e-05
  meiji_train_mode_ball_smooth: 0.0555349662899971
  meiji_target_supported_smooth: 0.004004123620688915
  meiji_shared_player_over_ball_supervised_grad_norm: 8.164998081634726
  meiji_shared_smooth_over_ball_supervised_grad_norm: 0.5424362181091659
artifacts:
  run_dir: knowledge/runs/run-slcs-ball-gradient-probe-v1
  output_dir: outputs/slcs/analyze/ball_gradient_probe/s42-001
  diagnostics: knowledge/runs/run-slcs-ball-gradient-probe-v1/results.json
  predictions: knowledge/runs/run-slcs-ball-gradient-probe-v1/meiji_eval.npz
parents:
- run-slcs-rgb-pilot-augmented-selected-conditions-v2
relations: []
tags:
- slcs
- real-rgb
- cpu-diagnosis
- gradients
- dropout
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: bc2e8e00e1dff7746ac8c099dcf791e9431146fc
  branch: codex/slcs-real-rgb
  command: env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
    .venv/bin/python -B knowledge/runs/run-slcs-ball-gradient-probe-v1/probe.py --output-dir
    outputs/slcs/analyze/ball_gradient_probe/REPRO_NEW_RUN_ID
  checkpoint_sha256: b925fc2a8cc2eb3dc80aacb7e2e6f8389d40ab48f8afb3c74408b97cbd17d30c
---

## 考察 / Findings

### 要約
既存augmented選定重みのCPU診断。入力と重みを固定してeval/train modeを比較すると、ball平滑化項はbroadcastで0.0000104→0.05576、Meijiで0.0000752→0.05553へ増えた。train modeの局所勾配では共有層に対するplayer群のnormがball supervisionの13.87倍/8.16倍。2つの窓・選定後checkpointの結果であり、学習履歴全体の支配や縮退原因を証明しない。

### アーキテクチャ詳細
既存model_io・target adapter・loss関数を使用。checkpointの期待SHA、元configのmodel/data/loss一致、入出力有限性、重み・buffer・入力の不変を検査。train splitの先頭適格120frame窓をdomainごとに1つ選んだ（broadcast_shanghai/clip_004 cam0 start0、video_000/clip_000 cam0 start0）。入力augmentationは明示的に無効、同じbatchでseed42をresetしてeval/train modeを切替。model.trainのdropout等のmode依存動作を比較し、optimizerは作らない。CPU FP32・batch1・deterministic algorithms有効は本学習のBF16/batch16と異なる。

### メトリクスの解釈
ball教師有効frameはbroadcast96/120、Meiji118/120。ball誤差はeval5.202/6.788m、train mode5.212/6.787mとほぼ変わらないが、予測の軸別stdはeval約0.003–0.011mからtrain mode約0.158–0.217mへ増えた。教師の長い移動を再現した分散ではなくmode依存の揺らぎである。妥当な連続教師区間だけのjerk penaltyは0.002736/0.004004で、同じmaskのtrain-mode出力は20.52倍/13.88倍。paddingのみの本番maskと教師有効maskを混同しない。学習曲線なし。

### アーキテクチャ⇄メトリクスの因果考察
損失値の大小からの推測を補うため、同じ共有axial trunkパラメータ順で勾配norm・dot・cosineを実測した。train modeのball smooth normはball supervisedの約0.53/0.54倍であり、この局所状態でsmoothが全てを圧倒するとは言えない。player群はnormが大きいがball群とのcosineは約−0.060/−0.059と小さく、単純な正面衝突とも断定できない。ball headでのsmoothとsupervisionのcosineはbroadcast−0.370、Meiji+0.414で符号も異なる。dropoutと時間平滑化の相互作用、タスク間の勾配配分を次の仮説として扱う。

### 既存実験との比較
先行のTensorBoardで見えたtrain/val平滑化項の差を、同じtrain入力でmodeだけ変える比較へ進めた。追加学習・閾値変更・test選定はしていない。既存の小規模pilot教師（新Meiji v8ではない）を使う診断であり、全体教師完成に先立つSLCS再学習ではない。モデル・loss配列4組、全項の値、同一順パラメータ名、未使用勾配数を保存。入力DINO配列は出力先に保持し、gitへは識別hashを記録して重複コピーしない。

### 次に有効な実験
ユーザー優先順位に従いMeiji全体教師/QCを先に完了する。その後、既存pilot/seed/60epoch/validation選定を固定し、ball_position_smoothness_weight=0だけを変える比較を行う。成功・失敗いずれでも唯一の原因とは断定せず、train誤差・予測分散・教師との対応・定数baselineを併せて確認する。別施策としてdropoutの時間相関やheadへの適用を検討できるが、最初の比較へ混ぜない。

比較設定は [`train_real_rgb_pilot_no_ball_smooth.yaml`](../../src/tasks/slcs/configs/train_real_rgb_pilot_no_ball_smooth.yaml) に分離した。CPUの設定合成・型付きruntimeで、保存済みaugmented runとの変更が平滑化重みと出力先だけであることを確認した（絶対root表記はruntime解決して比較）。この記載時点では追加学習は未実施で、Meiji全体QC後に共有queueへ投入する。

後続の読み取り専用コード監査は、初回に誤ってmain（`4348077d`）を参照したため、従来の監査版表記（`c7728c49`）を訂正する。`pwd`・git root・HEADを確認し、正しい `.claude/worktrees/slcs-real-rgb` の `740e788877a00971a7ab3d913458c8328379815e` で再監査した。SLCSは基底のoptimizer構築を使い、登録された全パラメータをAdamWへ渡す。ball embedding/headは通常のmodule属性で、保存済み両pilot設定にfreezeの指定はなく、resume・init_weightsはnull、learning_rateは0.0003、min_lrは0.000001だった。ballがoptimizerから外れる、またはfreezeされる構造的原因は見つからなかった。同版の `data/augmentation.py`・dataset・datamoduleでは、学習用augmentationは入力観測に適用され、教師と時間軸の `padding_mask` は維持される（RGB dropout時の入力側 `dino_padding_mask` は変更される）。datamoduleはtrainにだけaugmentationを有効化し、overfit時もval/testには適用しない。このコード監査は実checkpointのoptimizer state、学習途中の状態、学習済みモデルの入力感度を確認したものではない。上記CPU勾配診断は当初から正しいworktreeで実行された別の根拠であり、その `repro.commit=bc2e8e00` と実測結果は変更しない。
