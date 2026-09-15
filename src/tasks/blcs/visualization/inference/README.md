# BLCS Inference UI

`src/tasks/blcs` のチェックポイントと生成済みシーンを選び、GPU 推論を回して
GT と予測のボール軌道を 3D コート上で見比べるローカルWeb UI。カタログと
シーンの読み込みは読み取り専用で、モデルの重みやデータは変更しない。

起動コマンドと操作手順は[BLCS利用ガイド](../README.md)を参照してください。

ブラウザで `http://127.0.0.1:8770` を開く。外部CDNやフロントエンドのビルドは
不要。`--outputs-root` と `--checkpoints-root` は再帰的に `*.ckpt` を探索し、
`ckpt/blcs` を既定に含めることで curated な重みをそのまま使える。

既定の実行デバイスはCUDA。GPU要求は[共有GPUキュー](../../../base/visualization/README.md#web-uiのgpu実行)を通す。

## 画面

- 左：チェックポイント（学習データから導出した `allowed_forms`）と、フォーム・
  スプリット・検索語で絞り込むシーン一覧。
- 中央：選択シーンの GT 軌道と、選んだカメラで実行した推論結果を、物理コート
  座標系（右手系・+Z up・メートル）の3Dビューに重ねて表示する。
- 右：推論設定と結果。single-object は GT の第1トラック（メートル）に対する
  平均位置誤差・終端誤差・0.3m 以内のフレーム率を表示する。

## 推論の流れ

`service.py` はチェックポイントの Lightning config を正本として読み、
`model.name`・`court_keypoints.selector`・`model.io.input_profile`・
`model.num_queries`・`model.max_num_cameras`・`data.scene_dir`・
`data.num_views_range`・`data.seq_len_range` からフォーム・参照カメラ要否・
カメラ本数・窓長を決める。`allowed_forms` は `data.scene_dir` の basename では
なく `model.name` と `selector` の整合（`*_reference` は `camera_view_v2`、
それ以外は `physical_v1`）から決め、矛盾する checkpoint は `runnable=false`
として理由付きで一覧に出し、実行時も明示的に失敗させる。

- **single-object（trajectory）**: `load_scene_bundle`（camera_view 系は
  `camera_view_v2`、それ以外は `physical_v1` 契約）で読み、`model.max_seq_len`
  以下の重複しない窓（既定は学習クリップ長）に切って
  `BLCSPredictor.predict_scene` を回し、窓ごとの予測をフレーム順に連結する。
  予測は `blcs_trajectory_prediction_to_physical` でメートルに戻す。
- **multi-object（tracking）**: scene→input のビルダーとして
  `src/tasks/blcs/data/tracking_dataset.py` の `BLCSTrackingDataset` と
  `collate_blcs_tracking_batch` をそのまま使い、観測 association・reference
  selection・query packing は再実装しない。UI は
  `visualization/inference/tracking.py` の固定 window/camera サブクラスで
  `select_window`/`select_cameras` を上書きし、`augment=False`・単一シーンの
  一時 split file・保存済み checkpoint config を用いて 1 窓だけ推論する。
  `BLCSTrackingPredictor` の出力を `blcs_track_query_prediction_to_physical`
  で物理座標に戻す。tracking は `model.max_num_cameras` を持たないため窓長の
  上限は学習クリップ長（`data.seq_len_range[1]`）で、窓はフレーム 0 から
  1 本だけ切り、シーンより短い場合は warning を返す。

request の妥当性（unknown checkpoint/form、存在しないシーン、モデル構造上の
カメラ本数制限、重複・範囲外インデックス、参照カメラ未指定・
非選択、窓長が非正、明示 CUDA が利用不可）は
`InferenceService.validate_inference_request(**infer kwargs)` が **モデル読込
前に** 検証する。web層はここで `ValueError` を返し、GPU 待ちに入る前に
HTTP 422 として応答する。明示的な CUDA指定は `resolve_device` により
サイレントに CPU へフォールバックしない。学習サンプリング用の
`data.num_views_range` の範囲外は、動的 view モデルでは拒否せず警告する。

checkpoint の列挙・メタデータ読取は `torch.load(mmap=True)` を使い、重み
本体をメモリに展開せずに設定だけを取り出し、mtime/size でキャッシュする。

## multi-object のメトリクス

query slot と GT object の index は一致しない前提とし、各フレームで
present な予測と present な GT を最小コスト（Hungarian）で 1 対 1 対応させて
から位置誤差を測る。`position_error_m` / `endpoint_error_m` / `accuracy_0p3m`
に加え、`matched_pairs`・`predicted_present`・`gt_present`・
`presence_precision`・`presence_recall`・`presence_f1` を返す。GT は
`ball_pos_world`（物理座標・メートル）を窓分だけ使い、scene GT API は
single-object と同じ形（`tracks` と `presence`）で返す。

## API

- `GET /`：`static/index.html`
- `GET /static/{name}`：`index.html` / `style.css` / `app.js` / `scene.mjs` のみ
- `GET /api/catalog`：探索ルート、チェックポイント、シーンフォーム、コート形状、
  世界座標契約
- `GET /api/scenes?form=&split=&query=&offset=&limit=`：シーン一覧（ページング）
- `GET /api/scene?form=&scene=`：GT とカメラ一覧（モデル不要）
- `POST /api/infer`：窓推論・予測・メトリクス

`ValueError` は 422、`RuntimeError` は 409、`FileNotFoundError` は 404 として
`{"detail": "..."}` を返す。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tasks/blcs/visualization/inference -n 0
.venv/bin/python -m pytest tests/unit/tasks/blcs/model_io -n 0
```

テストは 6 形式すべてのサンプルデータを構築し、メタデータ分類・`allowed_forms`・
request 検証・窓推論・tracking の canonical dataset 構築・slot 非依存メトリクスを
CPU で検証する。実チェックポイントを読むテストは `@pytest.mark.local_data` を
付け、ローカル資産が無ければ skip する。
