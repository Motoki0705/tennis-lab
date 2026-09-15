---
id: run-meiji-convnext-l4-20260911
type: run
title: Meiji 3カメラのConvNeXtUNet学習（L4・勾配蓄積4）
provider: codex
session: 01a08d6c-c026-7420-be29-d217b791a69b
date: '2026-09-11'
status: done
config:
  model: conv_next_unet
  loss: focal_bce
  data: meiji_3cam
  config_name: train_meiji_3cam
  num_workers: 8
  accumulate_grad_batches: 4
  batch_size: 8
  effective_batch_size: 32
metrics:
  best_validation_epoch: 12
  best_val_loss: 0.0007240797276608646
  best_val_precision_at_loss_checkpoint: 0.034503329545259476
  best_val_recall_at_loss_checkpoint: 0.06830708682537079
  best_val_f1_at_loss_checkpoint: 0.045847922563552856
  best_val_mean_distance_px_at_loss_checkpoint: 2.5642449855804443
  best_f1_epoch: 17
  best_val_f1: 0.0507517009973526
  final_val_loss: 0.0007647858583368361
  final_val_precision: 0.036831825971603394
  final_val_recall: 0.079822838306427
  final_val_f1: 0.05040554702281952
  final_val_mean_distance_px: 2.461451768875122
repro:
  commit: 4e530309b022aee393cb264bf211407c1012f835
  branch: codex/meiji-ball-colab
  remote: https://github.com/Motoki0705/tennis-lab.git
  command: bash scripts/colab/run.sh run ball_meiji_3cam_l4 --gpu L4 --drive-mode
    mount --source snapshot --keep-on-failure --download-to ./colab-artifacts
  source_archive_sha256: c5b5fe30a707d25d3acd3207575064d4fc72020fc44afa78077d9b70cb742866
  source_content_digest: ad89faa565b2b7cb76e92c33bcd1b57af1807e30309696e93cad8df54624e361
  working_tree_dirty: true
artifacts:
  run_dir: knowledge/runs/run-meiji-convnext-l4-20260911
  output_dir: MyDrive/tennis_lab/colab-live/meiji-l4-bmp-acc4-20260911-r2/outputs/colab/ball_meiji_3cam_l4
  log: MyDrive/tennis_lab/colab-live/meiji-l4-bmp-acc4-20260911-r2/resume-attempt-5.log
  calibration: knowledge/runs/run-meiji-convnext-l4-20260911/batch_size.json
  config: knowledge/runs/run-meiji-convnext-l4-20260911/config.yaml
  validation_metrics: knowledge/runs/run-meiji-convnext-l4-20260911/validation_metrics.json
  checkpoint: MyDrive/tennis_lab/colab-live/meiji-l4-bmp-acc4-20260911-r2/outputs/colab/ball_meiji_3cam_l4/logs/version_1/checkpoints/meiji-convnext-epoch=12.ckpt
  checkpoint_sha256: 0ff29a4659fb823576fc35bd6b8862f5242ab5377256dcad50cf765e45e4ab03
  final_checkpoint: MyDrive/tennis_lab/colab-live/meiji-l4-bmp-acc4-20260911-r2/outputs/colab/ball_meiji_3cam_l4/logs/version_3/checkpoints/last.ckpt
  final_checkpoint_sha256: 53938c01c7a8ce5ee5a940d056c3c834b25f9088bef84fc0f11a7c07cce8c34b
  curves: knowledge/runs/run-meiji-convnext-l4-20260911/curves.png
  tb_logdir: outputs/colab_import/meiji-l4-bmp-acc4-20260911-r2/tensorboard/canonical
  inference_visualization: outputs/inference_visualizations/meiji-convnext-epoch12/video_002__clip_023.mp4
  inference_visualization_metadata: outputs/inference_visualizations/meiji-convnext-epoch12/video_002__clip_023.json
  inference_visualization_preview: outputs/inference_visualizations/meiji-convnext-epoch12/video_002__clip_023_preview.jpg
parents: []
relations: []
tags:
- ball-detection
- meiji
- colab
- l4
- convnext
---

## 考察 / Findings

### 要約
L4で最大physical batch=8を実測し、accumulate_grad_batches=4（effective batch=32）で20 epochsを完了した。
val/loss最小はepoch 12の`0.000724080`で、対応するcheckpointを最終候補とした。最終epoch 19はval/loss `0.000764786`、F1 `0.050406`だった。testは実行していない。
学習手順と設定の説明は[レシピ](../../scripts/colab/train/ball_meiji_3cam_l4.md)を参照する。

### アーキテクチャ詳細
既存ConvNeXtUNetと固定のBallDetectionDatasetを使用し、追加したMultiviewBallDataModuleで3カメラの動画とアノテーションを接続する。
同じ撮影クリップの3カメラは同じ分割に固定。各入力窓は1カメラの8連続フレームで、3カメラ同時入力ではない。
座標のあるobserved・interpolated・occlusion_estimatedを教師として使用し、unresolvedを含む窓は除外する。

### メトリクスの解釈
checkpoint monitorであるval/lossはepoch 12で最小になり、その後はepoch 19まで`0.000729144`～`0.000781720`の範囲で推移した。loss選択checkpointのprecisionは`0.034503`、recallは`0.068307`、F1は`0.045848`、対応済み検出の平均距離は`2.564245 px`だった。
F1自体の最大はepoch 17の`0.050752`で、loss最小epochと一致しなかった。最終epoch 19はprecision `0.036832`、recall `0.079823`、F1 `0.050406`、平均距離`2.461452 px`だった。全20 epochsの実測値は同runの`validation_metrics.json`、推移は`curves.png`に記録した。
testは`run.test_after_fit=false`で保留しており、frontmatterの値はすべてvalidationである。

学習完了後に追加された未注釈`video_002/clip_023`へbest val/loss checkpointをCPU適用し、3カメラ同期可視化を生成した。peak threshold 0.5以上はcam0で187/228、cam1で151/228、cam2で216/228フレーム、最大confidenceは順に0.9575、0.9504、0.9566だった。最大confidenceフレームの目視ではcam0・cam2の連続軌跡がボール付近に重なり、cam1は同時刻に閾値未満だった。ただし未注釈clipなので、この観察から正解率やtest性能は算出していない。

| physical batch | 結果 | peak allocated GiB | peak reserved GiB |
|---:|---|---:|---:|
| 1 | 3 optimizer更新・検証・可視化成功 | 1.70 | 2.06 |
| 2 | 同上 | 2.97 | 3.41 |
| 4 | 同上 | 5.51 | 6.55 |
| 8 | 同上 | 10.59 | 12.87 |
| 16 | CUDA OOM | — | — |

全成功試行は勾配蓄積4で12 training microbatchesを実行した。バッチ16では576 MiBの割り当てに対して空きが約409 MiBとなりOOMを検出した。
詳細は同runのbatch_size.jsonを参照する。本学習の保存済みconfig.yamlでもbatch_size=8、num_workers=8、accumulate_grad_batches=4を確認した。

全99動画・62,967フレームをColabのローカルディスクへBMP展開し、完了セッションの前処理は122.62秒、画像容量は27,857,986,074 bytesだった。
学習10,101窓、validation 1,270窓、test 788窓を確認した。

### アーキテクチャ⇄メトリクスの因果考察
train F1はepoch 3の`0.039060`からepoch 19の`0.079958`まで上昇した一方、val/lossはepoch 12以降改善しなかった。観測上は後半で学習側の適合が続き、validation lossが頭打ちになっている。軽い過適合、lossとpeak-based F1の選択基準のずれ、またはその両方が候補だが、この1 runだけでは原因を分離できない。

validation F1の絶対値が低い。仮説として、ランダム初期化、背景に対するボール画素の強い不均衡、`observed`に加えて`interpolated`・`occlusion_estimated`を同じ重みで使う教師構成、peak threshold 0.5、元画像座標4 px未満という対応条件が影響し得る。距離値は対応に成功した検出だけを集計するため、約2.4～2.6 pxだけを見て検出性能が高いとは判断できない。

BMPの事前展開は学習時の画像解凍負荷を減らす目的であり、モデル構造や教師を変更しない。完了セッションでは99動画の展開が約2分で、4 epochsの再開処理全体6067秒に対して小さい割合だった。

### 既存実験との比較
このMeiji分割・教師条件での比較対象は未設定なので、他runに対する改善は主張しない。
Colab runtimeの回収をまたいでepoch 3、13、15の完全checkpointから再開した。最終attemptはepoch 15のmodel・optimizer・scheduler状態を復元し、epoch 16～19を実行してreturncode 0で完了した。再開ごとにTensorBoard versionが増えたため、収束曲線はcheckpoint境界に従いversion 0のstep 1263まで、version 1の4423まで、version 2の5055まで、version 3の6319までを連結した。

ソースの一括転送はSSL接続エラーになったため8 MiBごとに分割転送し、source archive SHA-256の一致をVMで確認してから実行した。学習に用いたソースの正本はrun.jsonのsource digestで特定する未コミットsnapshotで、repro.commitだけでは追加実装を含まない。

### 次に有効な実験
testを開封する前にvalidation可視化で未検出・誤検出・遮蔽推定ラベルの挙動を確認する。その上で、既存checkpoint群を使ってpeak thresholdとcheckpoint選択指標をvalidation内で固定する。val/loss最小epoch 12とval/F1最大epoch 17が異なるため、次回は`val/f1`最大も保存対象にする価値がある。

モデル側の次候補は、(1) observedのみと座標付き3状態の教師アブレーション、(2) positive画素の不均衡を補うlossまたはheatmap sigma、(3)学習済みball detectorからの初期化である。変更は一度に1要因とし、同じclip split・seed・20 epochでvalidationを比較してから、固定した1条件だけをheld-out testへ適用する。
