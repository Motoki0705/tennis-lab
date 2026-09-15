# PLCS Review

生成データセットの閲覧は `dataset_service.py` / `dataset_web.py` が提供する。
起動・操作・API・座標契約は[共有review基盤](../../../base/visualization/review/README.md)を参照。

## ACCAD Motion Review

`data/ACCAD` の生モーション（AMASS / SMPL-H）を、PLCS の世界座標系で目視確認する
ローカルWeb UI。読み取り専用で、npzやモデルは変更しない。

```bash
.venv/bin/python -m src.tasks.plcs.scripts.review_accad_motion \
  --data-root /home/kamimura/projects/tennis-lab/data \
  --port 8769
```

ブラウザで `http://127.0.0.1:8769` を開く。GPU・フロントエンドのビルド・外部CDNは
不要。`--accad-root` は既定で `<data-root>/ACCAD`、`--smplh-root` は
`<data-root>/smplh`（`<gender>/model.npz` を置くディレクトリ）。

## 画面

- 左：`data/ACCAD` 配下のディレクトリ（被験者）とモーションを検索する一覧。
  検索語は空白区切りのANDで、被験者名・モーション名・genderに一致する。
  ディレクトリは折りたたみでき、選択すると中央に読み込む。
- 中央：選択モーションを世界座標系で再生する3Dビュー。ドラッグで回転、
  Shift＋ドラッグで移動、ホイールで拡大、ダブルクリックまたは「視点」で
  全体へ戻る。下部のトランスポートで再生／一時停止、フレーム送り、
  スクラブ、速度（0.25×〜2×）を操作する（Space、←→ も可）。
- チップは gender・fps・フレーム数・長さ、右上のHUDは現在フレーム・時刻・
  ルート位置[m]を示す。「追従」はカメラをルート位置に追従させ、「軌跡」は
  ルートの世界軌跡の表示を切り替える。
- 色は左半身=teal、右半身=coral、体幹=slate。地面グリッドは世界座標の
  1m（広いモーションでは5m）、軸は X=赤 / Y=緑 / Z=青。

## 座標系と関節

`poses`（156 = 52関節 × 3軸角）は AMASS の並び
`[global_orient, body, left_hand, right_hand]` のまま使い、pose mean は加えない。
形状ブレンド → ルート相対の剛体変換 → `trans` 加算という、repo の SMPL-H
linear blend skinning（`src/synthetic_data_generation/dataset/plcs/smplh.py`）と
同じ手順で `(T, 52, 3)` の世界関節を得る。座標系は `PLCSCoordinateContract`
（`plcs_amass_smplh_z_up_v1`：右手系・+Z up・メートル）で、`trans` を加えるため
ビューは世界座標をそのまま見る。

関節名と骨格は `src/utils/schema/player.py` の52関節定義
（`SMPLH_JOINT_NAMES` / `SMPLH_FULL_SKELETON`）をAPIで配る。指の並び
（index, middle, pinky, ring, thumb）は公式モデルの rest 骨格長で検証している。

## 構成

`service.py` がカタログ（npzメタデータ）と世界関節の再構成、`web.py` が
読み取り専用API、`static/scene.mjs` がCanvasの3D投影と操作、`static/app.js` が
選択・再生の配線を担当する。APIは次の3つ。

- `GET /api/catalog`：被験者・モーション・関節スキーマ・世界座標契約
- `GET /api/motions/{subject}/{motion}`：メタデータ（`?stride=` で間引き可）
- `GET /api/motions/{subject}/{motion}/joints`：`float32` の `(T, 52, 3)` バイナリ

`joints` はカタログが持つ `revision` を要求し、読み込み中にnpzが変わった場合は
409を返す。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tasks/plcs/visualization/review
node --test tests/unit/tasks/plcs/visualization/review/scene.test.mjs
PLAYWRIGHT_MODULE=/path/to/playwright ACCAD_REVIEW_URL=http://127.0.0.1:8769 \
  node tests/e2e/tasks/plcs/accad_review_browser.cjs
```
