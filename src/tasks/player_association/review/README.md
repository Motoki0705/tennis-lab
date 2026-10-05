# Player Association dataset review

保存済みのcross-cameraデータを読取専用で確認する。3cameraの同期全景とcrop、
camera-local raw ID → 匿名人物ID、ラベル推移、足元、参照boxの被覆を同じframeで辿れる。
全景のbox上には小さいraw IDだけを表示し、人物・cameraの対応はcropと選択パネルで読む。
新規推論・学習・注釈編集・player自動処理の再開は行わない。

## 起動

専用worktreeのカレントディレクトリから、共有venvのPythonで実行する。
`--dataset-root`と`--artifact-root`は絶対パス。`--sides`と繰返し可能な
`--score-report`はartifact root相対で指定する。後者は省略可能で、欠測を明示する。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.tasks.player_association.scripts.review_dataset \
  --dataset-root /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset \
  --artifact-root /home/kamimura/projects/tennis-lab/outputs \
  --sides court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json \
  --score-report player_association/evaluate/meiji_association/i933-evaluate-full-v4-20260927/evaluate.json \
  --score-report player_association/evaluate/meiji_association/i933-evaluate-geometry-v3-20260927/evaluate.json \
  --port 8897
```

`http://127.0.0.1:8897`でclip・frame・raw trackを選択する。全景/cropをクリックすると
同じ匿名**選手**がcameraを跨いで強調される。raw timelineの色付き区間と選択パネルの
切替frameからシークできる。URLの`clip/frame/camera/track/report`で表示条件を再現できる。
画像取得は指定frameをCPU decodeするため、大きくシークした際は待ち時間がある。

## 現物と現行reader（2026-10-04調査）

| family / schema | 用途・入力 → 参照 | 出自・単位 | split / size・現物 | 現在の位置付け |
|---|---|---|---|---|
| Meiji dataset/clip manifest v2・labels v1 | 同期3cam RGB、観測box → 選手/除外対象/曖昧box | 既検出boxを目視した部分参照。boxはsource px、匿名IDはclip内 | 57clips・29,624 source clip frames。7 labelled clips・7,205frames・63,279boxesがローカルに存在 | 現行のdataset注釈reader。50clipsには対応ラベルがなく負例にしない |
| 初期dev参照4clips / person_tracks v3 | 旧観測runのraw tracks、mask → boxラベル | `i933-observe-v1-20260927`・目視review。tracker/model出力 | `000/000`, `000/007`, `001/001`, `002/013`。設計・評価に使用済み、未見testではない | 歴史的なraw入力として読める。main標準trackingの現行出力はv5 |
| 追加raw参照3clips / person_tracks v5 | 全raw実観測 → 部分box参照 | `i964-unseen-r16-20261001`のrun18 Codex目視。`blind_to_association=true`、第2注釈者なし | `000/002`, `001/003`, `002/003`。事前予約clip、参照作成にassociation出力は使わず | 現行schemaの保存raw store。独立した人手の全人物GTではない |
| classical保存評価 / evaluate v1 | CLIP-ReIDと足元 → 保存pair証拠・MILP診断 | 距離m、geometry/appearanceはLLR、cosineは外観類似度 | 旧full-v4 / geometry-v3各4clipsのreportが存在。採用尺度Aのrun14 dev証拠も存在 | **mainの現在の方式はclassical＋採用尺度A**。UIが比較する旧reportは歴史的尺度であり現在設定の再推論結果ではない |
| learned view association研究 | 合成camera-local pose/ball/court → 合成人物ID | 合成truth、正規化2D・30fps格子 | 当時のPLCS/BLCS各1,000scenes（train800/val100/test100）というmanifest/検査記録、学習metrics/checkpointsは保存。旧multi_object入力の現物なし | 研究・歴史的artifact。main推論と区別する。削除済み形式を復活せず、RGBスクリーンショットの代用にしない |

正本は[`dataset_labels.py`](../evaluation/dataset_labels.py)、[`labels.py`](../evaluation/labels.py)、
[`metrics.py`](../evaluation/metrics.py)、[dataset/clip manifest reader](../../../tennis_scene/generate_dataset/manifest.py)、
[store descriptor reader](../../../tennis_scene/pipeline/storage/scene_index.py)と
[配列codec](../../../tennis_scene/pipeline/storage/codec.py)。main標準のschemaとraw/group IDの関係は
[pipeline README](../../../tennis_scene/pipeline/README.md)に従う。

現物は`data/tennis_multivew/processed/meiji_3cam/dataset/{dataset.json,videos/*/clips/*/clip.json}`、
clipごとの`annotations/player_association/{labels.json,review.yaml}`。
各reviewの`observe_run`が指定する観測runだけを開き、候補directoryを探索して代替しない。
初期runの`observe.json`は`stores/<clip>/scene.json`、追加参照の`person-execute.json`は
`<clip>/store/scene.json`を指定する明示的なlayoutで読む。後者はラベルのreceipt checksumも照合する。
旧footpointは保存court calibrationとreviewed-ball side、追加参照は保存court/side receiptに基づく。

learned研究の根拠は既存`association-learning-recovery` worktreeの
`src/tasks/{plcs,blcs}/configs/data/_association.yaml`、`src/tasks/base/data/association_dataset.py`と
`outputs/association_analysis_20260922/training_dataset_feasibility.json`。
[`c3d51a86d`](https://github.com/Motoki0705/tennis-lab/commit/c3d51a86d)が旧multi_object系とlearned入口を整理し、
[`601754149`](https://github.com/Motoki0705/tennis-lab/commit/601754149)がclassical採用尺度Aを既定化した。
ラベルのdataset移管は[`6e324f8c9`](https://github.com/Motoki0705/tennis-lab/commit/6e324f8c9)。

## 読み取りと欠測の意味

- `player`: clip内の匿名選手。`non_player`: 除外対象で、camera間同一性の採点・保証はない。
  `null`: 複数人物や背景を含む曖昧boxで採点対象外。未対応raw boxは別状態として表示する。
- raw観測がないframeは空白とする。人物の不在やout-of-frameを推定しない。
  v5のGSI補間boxを実観測やcropへ混ぜない。raw store欠測時はラベルboxのみを示し、raw IDと被覆は不明とする。
- 被覆はIoU≥0.5で照合した**参照box数 / 保存ラベルbox数**。元観測への照合100%でも、
  一度も検出されなかった人物はラベルに存在しないため、全人物のrecallは保証しない。
  曖昧・raw未対応・参照未照合の数と分母は折りたたみで確認できる。
- 足元はbox下端中央のz=0逆投影による推定。足切れ・光線不交差・side/校正欠測を理由付きで無効にする。
  無効点を原点へ描かず、表示範囲外の点も境界へ丸めない。poseや校正はmodel推定でGTとは扱わない。
- 保存scoreは同じ旧run・v3入力・camera/raw ID・segment区間/実観測数・時間格子を照合する。
  異なるrun/世代のrawへ同じ番号だけで結合しない。旧reportにbox内容hashは未保存という限界も表示する。
  保存segment全区間のscoreで、選択frameの再推論ではない。score欠測は0や確率へ変換しない。
- descriptor/配列checksum、schema、source RGB checksum（保存されている場合）、camera・source size・frame格子を確認し、
  不一致時は停止する。`ClipStore`の初期化・writer/runnerを使用せず、storeにlockやcacheを作らない。
  必要なraw/校正配列だけを読み、メモリcacheは最大2clips / 6RGB framesに制限する。

## 構成・検証

`reader.py`がv3/v5の明示readerと欠測状態、`diagnostics.py`が保存score照合、
`service.py`が被覆・timeline・同期frame/crop、`web.py`と`static/`がGETのみの画面を所有する。
CLIはtask-localにあり、共通のPathBoundary/inventory/package-data契約に従う。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/kamimura/projects/tennis-lab/.venv/bin/python -m pytest -n 2 \
  tests/unit/tasks/player_association/review \
  tests/unit/tasks/player_association/test_dataset_labels.py \
  tests/unit/utils/configuration
node --check src/tasks/player_association/review/static/app.js
```

テストfixtureは契約検証専用。公開画像はMeiji現物を実ブラウザで表示したものだけを使う。
