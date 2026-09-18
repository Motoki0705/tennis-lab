# PLCS Inference UI

PLCS のチェックポイントとシーンを選んで GPU 推論を実行し、テニスコート上に
GT と推論結果を同時に描画するローカル Web UI。読み取り専用で、データセットと
チェックポイントは変更しない。フロントエンドのビルド・外部 CDN は不要。

起動コマンドと操作手順は[PLCS利用ガイド](../README.md)を参照してください。

ブラウザで `http://127.0.0.1:8771` を開く。`--data-root` は既定で
`<repo>/data/plcs`、`--checkpoint-root` は既定で `<repo>/outputs/plcs`。
`--extra-checkpoint-root` は繰り返し指定でき、追加 root のチェックポイントは
`<parent>/<name>/...` 接頭辞つき（例 `ckpt/plcs/axial/logs/...`）で一覧に並ぶ。
root ごとの解決は外部パスを受け付けず、既知の root 配下に限定する。

GPU要求は[共有GPUキュー](../../../base/visualization/README.md#web-uiのgpu実行)を通す。

## 画面

- 左：チェックポイントのサジェスチョン一覧（`outputs/plcs` を走査）と、
  シーン形式・シーンの一覧。チェックポイント未選択では利用可能な形式を
  すべて表示し、選択すると**そのチェックポイントが使える形式だけ**に絞る。
  非対応の形式は折りたたみセクションへ移り、選択できない。
- 中央：コートを自由視点で見る 3D ビュー。GT と推論を同時に描画する。
  ドラッグで回転、Shift＋ドラッグで移動、ホイールで拡大、ダブルクリックで
  視点リセット。下部のトランスポートで再生／一時停止、フレーム送り、
  スクラブ、速度（0.25×〜2×）を操作する（Space、←→ も可）。
- 右：推論設定。device、使用カメラ、reference camera、推論するフレーム窓
  （開始位置と長さ）、canonical pose の由来（GT／推論）。実行後は位置誤差・
  yaw 誤差・関節誤差を表示する。チェックポイント未選択でも、選択中シーンの
  GT とカメラ位置・frustum を GPU なしで表示する（`/preview`）。

## チェックポイントとシーン形式の対応

チェックポイント側の保存済み config（`model.name` と
`court_keypoints.selector`）だけから、使用できるシーン形式を決める。
ファイル名からは推論しない。

| モデル | selector | 提示するシーン形式 |
|---|---|---|
| single-object モデル | `physical_v1` | `single_object`, `single_object_broadcast` |
| single-object モデル | `camera_view_v2` | `single_object_camera_view_v2` |
| `plcs_track_query` | `physical_v1` | `multi_object`, `multi_object_broadcast` |
| `plcs_track_query_reference` | `camera_view_v2` | `multi_object_camera_view_v2` |

学習時の `data.scene_dir` が対応表にある場合、その形式を先頭に並べる。
single-object 系は標準 profile（`frame` / `sequence` / `multiview`）、
multi-object 系（`plcs_track_query` / `plcs_track_query_reference`）は
`track_query` profile として推論する。両者は UI 上も別モード（`single` /
`tracking`）として扱い、`mode` をペイロードに含める。

メタデータは **checkpoint 本体を正本**とする。`torch.load(..., mmap=True)` で
保存済み Hydra config だけを読み、隣接する `hparams.yaml` / `config.yaml` は
本体が読めないときのみ使う。両方が存在して `model.name` / `selector` /
`input_profile` / `max_views` / `max_seq_len` / `num_queries` / `data.scene_dir`
のいずれかが食い違う場合、その sidecar は古いものとみなしてチェックポイントを
**拒否**する（信用しない）。数百 MB の checkpoint を並べても一覧表示は軽い。

## 推論の契約

- 入力は `data/plcs/<family>/scenes/<scene_id>` を `court_keypoints.selector`
  に対応する contract で読み込む。reference（`camera_view_v2`）では
  `reference_camera_id` が必須で、選択カメラに含まれていなければならない。
  モデルが 3〜4 カメラを要求する場合（axial reference）はその範囲を強制する。
- モデルに `max_seq_len` がある場合、フレーム窓の長さはそれを超えられない。
  シーン全長より短い窓を推論した場合は警告を返す。
- 出力は世界座標メートル（右手系、+Z up、X=コート幅、Y=コート長、ネット y=0）。
  ブラウザは正規化表現や reference frame を知らない。
- GT の骨格は保存済み `human_kp_3d`（COCO17・メートル）を使う。無い場合は
  推測せずエラーにする。
- モデル構造上の view 制限を強制する。standard モデルは `max_views`
  （axial reference モデルは 3〜4 カメラ）、track-query は保存済み config の
  `data.camera_candidates` を守らせる。学習サンプリング用の
  `data.num_views_range` の範囲外は、動的 view モデルでは拒否せず警告する。
- device は `utils.device.resolve_device` で厳密に解決する。CUDA を明示して
  利用できない場合は CPU へ黙って落ちず、事前検証の時点で失敗する。

### multi-object（track-query）

single-object 用の `PLCSPredictor.predict_scene` は使えない。`build_sample` が
observation association と lifecycle slot packing を行った後にしかモデル入力が
完成しないため、UI は `PLCSTrackingDataset` の窓・カメラだけを request で固定した
薄い subclass（`tracking.SingleWindowTrackingDataset`）を使い、checkpoint の
保存 config と single scene の一時 split file から `(B=1)` バッチを組む
（`augment=False`）。association や slot packing はモデル側と同一の実装を通る。

推論は `PLCSTrackingPredictor.predict` に typed `reference_metadata` を渡して
物理座標へ戻す。reference（`camera_view_v2`）では `reference_camera_id` が必須で、
選択カメラに含まれていなければならない。

## 構成

`checkpoints.py` がチェックポイント走査と形式の絞り込み、`loader.py` が
推論専用のチェックポイント復元、`service.py` がシーンカタログと推論、
`web.py` が HTTP API、`static/` がブラウザ実装を担当する。正規化↔物理の
変換は `visualization.api.predict` と `court_keypoint_contract`、姿勢合成は
`utils.geometry.court_pose` の共有関数を使い、UI 独自の再実装はしない。

### チェックポイントの復元（`loader.py`）

`PLCSPredictor.load_from_checkpoint` / `PLCSTrackingPredictor.load_from_checkpoint`
は Lightning の checkpoint 復元を通るため、`PLCSTrainingConfig` 全体
（`run.artifact_store` など学習時にしか存在しない section を含む）を要求する。
学習されていない推論専用 checkpoint はこの section を持たないので、UI は
`loader.py` でモデル生成に必要な設定だけを厳密に再構成する。

- 復元手順は `PLCSTrainingConfig.from_config` のうち training section 検証より
  前に走る部分と同じ関数を使う（`load_and_validate_checkpoint` →
  `prepare_plcs_checkpoint_court_keypoint_config` → `PLCSModelConfig.from_mapping`
  → `PLCSPathConfig.from_config` → `PLCSDataConfig.from_mapping`）。
- 得た 3 値（`model` / `data` / `court_keypoint_contract`）を既存の
  `build_plcs_model_io` factory にそのまま渡す。factory の引数はこの 3 プロパティ
  だけを要求する読み取り専用 Protocol（`PLCSModelIOConfig`）に変更してあり、
  `PLCSTrainingConfig` は構造的に適合する（学習側の挙動は不変）。
- Lightning module の `on_load_checkpoint` と同じ marker 検証を行い、
  `model.` prefix の重みだけを **`strict=True`** で復元する。過不足や
  architecture 不一致は拒否し、旧 architecture の補完・キー変換はしない。
- checkpoint root の境界は `service._checkpoint_resolver` を通す。
- この loader は推論専用で、optimizer / scheduler / loss / metric state は
  一切読まない。

API は次のとおり。

- `GET /api/catalog`：シーン形式の一覧と checkpoint のサジェスチョン
- `GET /api/scenes?family=&split=&query=&limit=`：split 内のシーン一覧
- `GET /api/scenes/{family}/{scene}?checkpoint=`：シーン詳細と窓・カメラ条件
- `GET /api/scenes/{family}/{scene}/preview?cameras=&window_start=&window_length=`：
  GPU なしで GT とカメラ frustum を返す（predict と同じフレーム付きバイナリ）
- `POST /api/validate`：`/api/predict` と同じ事前検証だけを行い、GPU 待ちの前に
  不正な checkpoint/format/cameras/window/device を 422 にする
- `POST /api/predict`：推論を実行し、JSON ヘッダー + float32 ペイロードの
  フレーム付きバイナリを返す（`uint32` ヘッダー長 → UTF-8 JSON → float32）
- `GET /static/{name}`：`style.css` / `app.js` / `court_scene.mjs` のみ
- `GET /shared/{name}`：共有の Three.js モジュール（`mount_scene_assets`）

## バイナリの読み方（frontend 向け）

`/api/predict` と `/api/scenes/{family}/{scene}/preview` は同じ形式で返す。
先頭 4 バイトが `uint32`（リトルエンディアン）のヘッダー長、続く JSON が
ヘッダー、残りが float32 の平坦なペイロードである。ヘッダーの `tracks` は
各要素が `position` / `rotation` / `presence` / `joints` を持ち、それぞれ
`{offset, count, shape}` で flat ペイロードの位置と論理形状を示す。`offset` は
要素数（バイトではない）で、`shape` の順に `Float32Array` を切って使う。

- `mode`: `"single"`（1 人・index 比較の metric）／`"tracking"`（多人数・
  slot）／`"preview"`（checkpoint 未選択の GT のみ）。
- `tracks[*].kind`: `"gt"` か `"pred"`。`object_index` は同 kind 内の連番。
- `tracks[*].presence`: `(T,)` float32（0/1）。multi GT／pred はこれを尊重し、
  0 のフレームは描画しない。
- `tracks[*].rotation`: `(T,2)` の `(cos, sin)` yaw。
- `tracks[*].joints`: COCO17 の `(T,17,3)`。multi の **pred** は query slot が
  時間方向で人物を再利用するため `null`（`has_joints=false`）。single のみ
  予測骨格を出す（canonical pose を出力しないモデルでは `null`）。
- `header.cameras`: preview のみ。`index` / `id` / `center` / `frustum`（5 頂点）/
  `image_size`。`header.court` と `header.skeleton` は単一の描画スキーマ。
- `header.metrics.scope`: `"single_object"` か `"multi_object_tracks"`。
  multi の metric はフレームごとの Hungarian 最小コスト割当で、学習時の
  lifecycle metric とは定義が異なるため直接比較しない（`metrics.note` に明記）。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tasks/plcs/visualization/inference
node --test tests/unit/tasks/plcs/visualization/inference/court_scene.test.mjs
```

`test_tracking.py` と `test_preview.py` は `data/plcs` の全 6 形式に対し、実データ
からサンプルを構築して CPU 上の tiny な track-query モデルで推論とペイロードを
検証する（データが無い環境では skip）。
