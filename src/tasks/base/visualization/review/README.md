# Dataset Scene Review (shared core)

BLCS / PLCS の生成データセット（`data/<task>/<form>/scenes/<scene_id>/`）を、
ローカルWeb UIで3D再生する共通基盤。読み取り専用で、データは変更しない。

このパッケージはタスク非依存の中核だけを持ち、各タスクが薄いサービスを束ねる。

## 起動

リポジトリのPython環境から実行する。`--data-root` は `plcs` と `blcs` を
含むデータディレクトリを指定する。

コピー可能なコマンドは[PLCS利用ガイド](../../../plcs/visualization/README.md)・[BLCS利用ガイド](../../../blcs/visualization/README.md)を参照。

PLCSは `http://127.0.0.1:8772`、BLCSは `http://127.0.0.1:8773`。
`--port` で変更できる。左のディレクトリからシーンを選択し、中央のコートを
回転・平行移動・ズームできる。カメラ位置・視錐台・カメラ視点、再生・シーク、
軌跡・追従の切り替えに対応する。Three.jsはローカル配信する。

形式は `single_object` / `multi_object` と、それぞれの `_broadcast` /
`_camera_view_v2` の全6形式。

## 構成

- `catalog.py`: `data/<task>/*/scenes` を自動検出する `DatasetCatalog`。
  scene一覧はディレクトリ名のみ。revisionは選択シーンのJSON・npyと形式の
  metadataの `st_size`+`st_mtime_ns` のsha256先頭20桁。
- `camera.py`: `parse_cameras()` が `cam_<i>_params` を `CameraRecord` に変換。
  `camera_to_world` / `intrinsics` / `frustum_vertices`（`camera_geometry` の
  `camera_frustum_corners` を再利用、5頂点、OpenCV規約）を提供する。
- `court.py`: `src.utils.schema.court` を薄く包み、20点コート・線分・アプロン・
  ネット形状を返す（定数の二重定義なし）。
- `payload.py`: `ScenePayload` とエンティティバイナリのコーデック。
- `service.py`: `DatasetSceneReviewService`。パス解決・revision検証・コート/
  カメラのJSON化・エンティティ整形・CourtKP検証の呼び出しを共通化する。
- `web.py`: `create_review_app(service, title=..., task=...)`。FastAPI アプリ。

## API

- `GET /` : `index.html`
- `GET /static/{name}` : 許可リストのみ
- `GET /api/catalog` : `{task, root, forms[], entity, skeleton}`
- `GET /api/scenes?form=<form>` : `{form, scenes:[...]}`（名前のみ）
- `GET /api/scene?form=<form>&scene=<id>` : scene JSON（`revision` を含む）
- `GET /api/scene/buffer?form=<form>&scene=<id>&revision=<rev>` :
  `application/octet-stream`。revision不一致は409。

## エンティティバイナリ

`float32 (slots, frames, joint_count, 3)`（C順）→
`float32 (slots, frames, 2)` orientation（PLCSのみ）→
`uint8 (slots, frames)` presence（multiのみ）、の順に連結する。
`float32` を先に置くのは typed-array view の4バイト境界を保つため。
要素 `(s,t,j,k)` は `((s*frames+t)*joint_count+j)*3+k`。
`presence=false` のフレームは位置0で、UI側で非表示にする。

## 座標系

保存済みの3Dデータ（ボール・選手関節・`position`）とカメラ `C`/`R` は
全形式で physical court frame に統一されている。`camera_view_v2` は
カメラ毎の CourtKP20 ラベル順（`semantic_to_physical` /
`canonical_from_physical`）だけが異なり、幾何は同一なので、コート描画は
`court_keypoints_3d` をそのまま使う。

## 検証

```bash
./scripts/run_in_repo_venv.sh pytest -n auto tests/unit/tasks/base/visualization \
  tests/unit/tasks/plcs/visualization tests/unit/tasks/blcs/visualization
node --test tests/unit/tasks/base/visualization/review/scene.test.mjs
```

`tests/unit/tasks/{blcs,plcs}/visualization/review/test_dataset_review.py` は
実データで `proj(cam_i, court_keypoints) == cam_i_court_kp_uv`（`semantic_to_physical`
で並べ替え）とボール/選手の再投影不変条件を6形式すべてで検証する。
