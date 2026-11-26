# 3D Ball Refiner

三角測量した3D座標 `(B,T,3)` と欠損mask `(B,T)`（true=欠損）から、
観測区間を含む全frameの軌道1本と、各frameのショット・バウンド・ネット接触の確率
`(B,T,3)` をこの順序で推論する。同じframeで複数のイベントが成立する。
オフラインの双方向Transformer。confidence・court・camera・GTイベントはモデルへ渡さない。
`model.physics_heads=true` では、ラリーの場（風・k_drag・k_magnus・surface）と
飛行区間ごとの初期状態（位置・速度・スピン）も出力し、それらを共有の物理積分器で積分した軌道を併せて返す。

## 構成と依存

| ディレクトリ | 責務 |
|---|---|
| `configs/` | Hydraのdata・augmentation・model・loss・training・generation・visualization設定 |
| `configuration/` | 設定の厳密な検証、path境界、旧実験設定の明示的な読み取り |
| [data/](data/README.md) | NPZ読込、拡張、三角測量、イベント・物理教師、可変長窓サンプリング、DataModule |
| `generate_dataset/` | BLCS物理生成adapter、camera選択、保存、既存3D軌道の再投影 |
| [models/](models/README.md) | 共通trunk、回帰・Flow generator、物理head、軌道discriminatorのforward |
| `physics/` | 物理パラメータのネットワーク単位、`ball_physics.v1` からの教師、区間の積分による軌道再構成 |
| `model_io/` | 入出力型、正規化、adapter、factory、checkpoint読込 |
| [training/](training/README.md) | BaseTrainingRunner / BaseLightningModuleへの接続、loss、step schedule |
| `inference/` | 公開Predictor、クリップ全体の1回forward、予測イベントによる区間分割、Flow sampling、NPZ入出力 |
| `evaluation/` | 全ラリー評価、baseline、checkpointと対応付けた予測・指標の保存 |
| `post_training/` | SceneResult v3の選手動作入力、凍結Refinerへの融合、実映像での追加学習と評価 |
| [visualization/](visualization/dataset_review/README.md) | 静的plotとDataset Review WebUI |
| `scripts/` | 引数・path検証と上記APIの呼び出しのみ |

`models` はloss・sampling・dataset・WebUIを参照しない。
`model_io` がforward前にshape/device/maskを検証し、共通 `BoundModelIO` を通じてモデルを呼ぶ。
汎用Transformer部品は `src/utils/models/components`、学習基盤は `src/tasks/base` を再利用する。
シーンへの3D Refiner接続はこのtaskの学習・推論APIとは別の工程。

## 実行

```bash
# 既存datasetがあれば停止する。共有データの再生成は通常不要。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.generate_dataset
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.generate_dataset generation=net_contacts

# ローカルCUDAは必ずtraining queueへ登録して実行する。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train data.dataset=ball_refiner/single_object_net_v1
# Flowの比較実験は model=flow。回帰は model=regression。
```

GPU実行は[training queue](../../../.agents/skills/training-queue/SKILL.md)に従う。
学習設定・再開・保存先は[training README](training/README.md)を参照。

`generation=net_contacts` は `data/ball_refiner/single_object_net_v1/` を生成する。
既定では各splitの50%をネット接触に割り当て、白帯から相手コートへ越えて次の打球がある球と、
打者側へ落ちて終了する球に均等に配分する。残り半分を通常ラリーとして保持しつつ、
train 512件、val・testそれぞれ64件の接触を確保する設定。
割合は `generation.net_contact_fraction` で変更できる。split内で最も近い整数に丸め、
奇数の場合は越える球を1件多くする。採用した軌道は物理記録の再積分とfloat32のビットを照合する。
manifestに物理イベント件数と接触の内訳を全体・split別で保存する。

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.predict \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --input /absolute/input.npz --output /absolute/prediction.npz --device cpu

.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.evaluate \
  --checkpoint /absolute/run/logs/version_0/checkpoints/best.ckpt \
  --run-config /absolute/run/config.yaml \
  --dataset /absolute/data/ball_refiner/single_object_net_v1 \
  --output /absolute/new-evaluation --device cpu

.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.review_dataset
```

入力NPZは `coordinates`, `missing`, `fps` のみ。座標はm、欠損はbool。
公開API `inference.RefinerPredictor` は同じ契約で入力し、CPU上の座標とイベント確率を返す。
FPSは絶対差が `max(1e-6 Hz, fps_tolerance)` 以下の場合に受け付ける。閾値の境界では大きい方のFPSの1 ULPを丸め誤差として許す。
各クリップ全体（最大4096 frame）を1回のforwardで推論する。
物理headを持つcheckpointは、クラスごとに確率0.5以上かつ3 frame以上離れたピークを取り、
frame 0と全クラスのピークの和集合で区間を作り、
その区間で初期状態を出力して積分軌道を作る（積分定数はcheckpointの `flight_clock`）。
`ball_refiner_3d.prediction.v2` の出力NPZには `event_classes` と `event_probability` が含まれる。
物理head付きの `ball_refiner_3d.physics_prediction.v2` には `integrated_coordinates`、`wind_mps`、`k_drag`、`k_magnus`、
`surface_probability`、`segment`、`segment_{position_m,velocity_mps,spin_radps}` が加わる。
新しい評価先・推論出力は既存ファイルを上書きしない。

`inference.scene.predict_scene_events(predictor, scene_json, clip_id=..., checkpoint_sha256=..., seed=42, fps_tolerance=0.06)`
は、SceneResult v2・v3のコート座標 `ball_3d` をmのまま使い、`ball_3d_valid` の反転を欠損maskにする。
実測59.94005994 fpsと公称60 fpsの差を0.06 Hzまで明示的に許容する例。
元frameをそのまま推論し、`ball_event_predictions.v1` のイベント時刻には実測fpsを使う。

## 評価指標

`evaluation/evaluator.py` は座標RMSE・イベント確率に加え、checkpoint評価で物理整合性 `physics_eval.v2`
（`evaluation/physics.py`）を計算する。GTの飛行区間ごとに共有の力モデルを当てはめた残差、
区間内有限差分の速度・加速度誤差とjerk比、非物理加速度率、地面下率、イベントのピーク照合を、
欠損・観測・イベント近傍・バウンド後・ショット後のframe群と、surface・ラリー長・バウンド数別に出す。
`diagnostic_metrics.json` の `physics.events.by_kind` は、ラリー内・クラス内でピークを真値の±2 frame以内に1対1照合する。
一致数を最大化し、同数の割当の中で絶対フレーム誤差の合計を最小化する。適合率・再現率・F1と平均絶対誤差・平均符号付き誤差をframe・秒で記録する。
`metrics.json` には `test_event_net_f1` などの代表値を保存する。
当てはめは区間全体の残差を最小化するため、欠損区間の非物理性は同じ区間の観測frameの残差にも現れる。
学習中のvalidationは物理評価を行わない。
物理headのモデルは、積分軌道（予測イベントで区間分割）の座標誤差と物理指標、GT区間分割での積分軌道（上限）、
場・surface・区間初期状態のパラメータ誤差（`parameters`）も報告する。
このとき `pred_test.npz` には次の配列が加わる。
- frame単位：`integrated`、`integrated_segment`、`integrated_truth_segments`。
- ラリー単位：`physics_field`（風x・風y・k_drag・k_magnus）と `physics_surface_probability`。`physics_rally_id` でラリーに対応付ける。

dataset reviewはこれらの配列を表示する。

## 互換性

3クラスのcheckpointは `ball_refiner_3d.events.v3` とイベントの順序を持つ。
旧1チャネルの `events.v1`・`events.v2` は読み込み時に拒否する。
学習で保存するLightning checkpointにはoptimizer・scheduler・乱数状態も含まれる。
共有dataset・学習済み重み・過去の実験記録は変更しない。
旧CLI・flat moduleは廃止したため、過去実験の再実行には記録されたcommitを使用する。

## 実映像でのモーション付きイベント追加学習

Meiji `multiview-v3` の公開SceneResult v3を使う。事前学習済みの3クラス回帰checkpointから球embedding・Temporal Transformer・座標head・イベントheadを読み込み、evalモードで固定する。1フレームにつき1トークンを保ち、球embeddingへ選手の残差を加える。

| `--mode` | 入力融合 |
|---|---|
| `shared` | 共有MLP（161→128→256、RMSNorm・SiLU）で符号化し、有効な選手を加算 |
| `per-player` | 同じ構造のMLPを4枠それぞれに持ち、有効な選手を加算 |
| `set-attention` | 共有MLP、4-head SAB 1層、球embeddingをクエリにしたattention pooling |
| `ball-only` | 事前学習済みモデルで評価のみ実行 |

残差への最終線形層はweightとbiasをゼロで初期化する。Set Attentionは球と選手の種別embeddingを持ち、無効な選手をkey/valueから除外する。有効な選手がいないフレームでは全方式が球embeddingをそのまま使用する。[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)

選手枠は負のY側2枠、正のY側2枠。各側はmanifestとSceneResultの安定IDを昇順に配置し、欠員をmaskする。各選手の161次元は関節の回転6D（126）、root位置（3）、yawのsin/cos（2）、root速度（3）、左右手首のroot相対位置（6）とその速度（6）、球のroot相対位置（3）と左右手首相対位置（6）、root・heading・SMPL・球・root速度・手首速度の有効性（6）。XYZと球相対位置は既存の `[10,20,5]` m、手首相対位置は1 m、速度は10 m/sで割る。速度は有効な連続2フレームの後退差分と元FPSから求める。SMPL-Xの中立形状で順運動学を行い、SceneResultに記録された体スケール、Y-up→Z-up変換、yawと描画補償回転の逆を合成してコート座標に配置する。root単体の観測はSMPL欠測時にも使用する。

```bash
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.post_train prepare \
  --dataset "$BENCHMARK" --manifest-sha256 "$MANIFEST_SHA" \
  --smplx "$SMPLX_NEUTRAL" --reference-labels "$REFERENCE_LABELS" \
  --reference-pins "$REFERENCE_PINS" --reference-pins-sha256 "$REFERENCE_PINS_SHA" \
  --output "$PREPARED"

# CUDAを指定するコマンドはtraining queueに登録する。
.venv/bin/python -m src.tasks.ball_refiner_3d.scripts.post_train run \
  --dataset "$PREPARED" --manifest-sha256 "$MANIFEST_SHA" \
  --prepared-sha256 "$PREPARED_SHA" \
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --mode shared --seed 42 --device cuda --output "$RUN"
```

パスは絶対パス、出力は新規ディレクトリを指定する。`prepare` はmanifestの47/11/11分割、sceneと注釈のhash、元動画区間、FPS、選手IDを照合する。元動画区間の分割間重複はエラーになる。独立ラベルは #2542 のrun bundleの `inputs.json` と採点6本の `video_000/clip_NNN/labels.json` を指定する。`prepare` が出力する `inputs.json` のSHA-256が `PREPARED_SHA`。

学習は1クリップ1更新で全長を使い、AdamW（既定12 epoch、lr 0.0003、weight decay 0.01、勾配norm上限1.0）。checkpointのGaussian幅でcertain・probableを教師にし、ポイント外の注釈も含める。coverage外は全クラス、uncertainの前後2フレームは該当クラスをBCEから除く。別の確定・有力ラベルが重なる場合はその正例中心を保持する。validationのmacro-F1、同点はvalidation BCE、さらに同点は先のepochで選ぶ。

testでは確率0.5以上・同クラス3フレーム間隔でピークを抽出し、クラスとクリップごとに元フレーム±2の1対1対応を最大化する。ポイント内はTP/FNをラベルで分類し、FPは連続するポイント内注釈の最初から最後までで判定する。ポイント外の注釈と次のサーブは区間の境界になるため、このFPは注釈からの推定に基づく。条件別の判定時刻はTP/FNがラベル時刻、FPが予測時刻。ネット付近は観測球の `abs(y) ≤ 1 m`、SMPL欠測は割当済みの選手に対するmaskを使う。ボレーは注釈subtypeごとの再現率・時刻誤差を報告する。正解イベントのない条件は指標を `null` とする。

`run.json` と `history.json` に設定・入力hash・学習曲線を、`best.pt` に選択した融合重みと事前学習checkpointのhashを保存する。`predictions/` に元フレーム単位の予測、`matches.json` に照合内訳、`metrics.json` に両ラベル源の精度・追加パラメータ数・推論時間・CUDAピーク割当を保存する。計測は各クリップ1回のwarmup後に同期付きforwardを3回行い、入出力の転送と採点を計測から除く。
