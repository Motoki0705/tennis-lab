# PLCS foot residual experiment

`prepare.py` は比較元checkpointから明示的に残差モデルを初期化し、学習splitだけの足首疑似位置の統計・重みを作る。`evaluate.py` は両モデルを同じseed・testシーン・本番観測・推論窓で比較する。`render.py` は保存した予測から比較動画と図を生成する。`finalize.py` はvalidation最良checkpointを選び、両モデルを同じGPUで評価し、標準 `SceneResult` archive（`residual_scene.npz` とmetadata sidecar）へ既存のGVHMR/SMPLと新しいPLCS位置・回転を統合する。実験の数値・考察は `knowledge/` に記録する。

```bash
# 専用worktree内、Pythonは共有 .venv を使用。
PYTHONPATH=. .venv/bin/python scripts/plcs_foot_residual/prepare.py \
  --baseline /path/to/plcs-multiview-axial-split-camera-view-v2-epoch19.ckpt \
  --source data/plcs/single_object_camera_view_v2 \
  --output outputs/plcs/foot_residual

# GPUコマンドは必ず元repoのtraining queue経由。
.venv/bin/python -m src.tasks.plcs.scripts.train \
  --config-path /absolute/worktree/outputs/plcs/foot_residual --config-name train

PYTHONPATH=. .venv/bin/python scripts/plcs_foot_residual/evaluate.py \
  --checkpoint /absolute/checkpoint.ckpt --label baseline_best \
  --baseline-config /absolute/baseline/config.yaml \
  --dataset data/plcs/single_object_camera_view_v2 \
  --clip data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000 \
  --output outputs/plcs/foot_residual/comparison
# validation最小の新checkpointでも --label residual として同じ評価を実行。

PYTHONPATH=. .venv/bin/python scripts/plcs_foot_residual/render.py \
  --comparison outputs/plcs/foot_residual/comparison --baseline-label baseline_best \
  --clip data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000
```

最終比較の合成推論は両モデルともfloat32を使う。合成評価は全testシーンの中心128フレーム、seed 1234、camera_0 referenceを固定する。実クリップは保存済み `human_kp_2d` / `human_kp_vis` と手動courtを使い、本番のstride=2・128/64 windowで全1010フレームを再構成する。人物対応と入力観測は両モデルで共通であり、モデル推論による既存の `player_position` はGTに使わない。

実映像の指標は、コート点だけで近似したカメラによるroot再投影とCOCO左右hipの中点の距離。3D位置の正解誤差ではなく、カメラ近似・2D検出・root定義の差を含む整合性指標である。カメラのfit誤差を一緒に保存する。baselineのcanonical headには教師損失がなかったため、canonical poseの再投影は比較指標に採用しない。

重みは各scene最大64等間隔フレーム・4カメラのclean観測から推定した水平誤差で作る。元分布50%とシーン平均水平誤差の上位20%から50%を混ぜ、各epochに復元抽出する。`sampling_summary.json` の抽出後割合はseed固定の事前シミュレーションであり、学習中の各epochの実測値ではない。訓練では従来どおり3–4 viewの選択・128 frame crop・augmentationを行うため、事前推定と実際の各batchの難易度分布は同一ではない。val/testのsplit・観測分布は変えない。
