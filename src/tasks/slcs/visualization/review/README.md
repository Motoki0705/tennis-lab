# SLCS Dataset Review

SLCS の構造化データセットに保存された疑似ラベルを、ローカルWeb UIで3D再生する。
PLCS / BLCS と同じ[共有review基盤](../../../base/visualization/review/README.md)を使う。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.review_dataset \
  --dataset-root /home/kamimura/projects/tennis-lab/data/slcs/real_rgb_v1
```

ブラウザで `http://127.0.0.1:8774` を開く。`--dataset-root`（別名 `--data-root`）は
`dataset.json` を含むディレクトリを指定する。`--video-id <id>` は繰り返し指定でき、
一覧を対象videoへ限定する。接続先は `--host` / `--port` で変更できる。
GPU、checkpoint、DINO tokens、split fileは不要で、データは変更しない。

## 表示と操作

- 左のvideo一覧を展開してclipを選ぶ。検索、再生／停止、フレーム送り、シーク、
  速度変更、俯瞰などの視点切替、軌跡、先頭選手への追従に対応する。
- 選手はルート位置のマーカーとyaw矢印、ボールは位置と軌跡として同時に描く。
  選手は学習と同じ平均court-Yによる手前→奥の順。表示単位はメートル。
- 対象は3D疑似ラベルで、RGB動画や2D pose overlayは表示しない。SLCSの標準教師は
  選手の位置・yawであるため、SMPLや3D骨格は生成しない。
- 入力カメラIDは凡例に表示する。現行manifestに3D校正情報がないため、
  カメラ位置・視錐台・撮影カメラ視点の操作は無効。
- `configs/data/default.yaml` の品質設定と `data.dataset.load_clip_arrays()` を使う。
  無効ラベルは非表示とし、軌跡も欠損区間を結ばない。品質閾値・有効フレーム数は
  scene APIの `quality` に含まれる。学習用windowの抽出・間引きは行わない。

## 構成

`dataset_service.py` が正規manifestによるvideo/clip一覧、strict annotation読込、
単位復元、選手とボールのpayload生成を担当する。アーカイブはclipを選ぶまで読まず、
最近の2件だけキャッシュする。未完了annotation、不正なdigest・shapeはエラーになる。
scene JSONとbufferの間でmanifest・annotation・NPZ・metadata sidecarが変わると409を返す。

`dataset_web.py` は共有FastAPIアプリへの接続だけを持つ。共有APIの `form` はvideo ID、
`scene` はclip名に対応する。scene JSONの `entities` とbinary layoutは共有基盤の正本を参照。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tasks/slcs/visualization/review
node --test tests/unit/tasks/base/visualization/review/model.test.mjs
PLAYWRIGHT_MODULE=/path/to/playwright \
  SLCS_REVIEW_DATASET_ROOT=/path/to/dataset \
  node tests/e2e/tasks/slcs/dataset_review_browser.cjs
```
