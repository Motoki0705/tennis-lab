# SLCS Dataset Review

保存済みSceneResult v2の疑似教師を、元clip RGB・2D観測・教師qualityと同じframeで確認する読み取り専用Web UIです。
[共有review基盤](../../../base/visualization/review/README.md)の3D再生とscene/frameイベントを使います。

```bash
./scripts/run_in_repo_venv.sh python -m src.tasks.slcs.scripts.review_dataset \
  --dataset-root /home/kamimura/projects/tennis-lab/data/slcs/meiji_one_clip_scene_v2 \
  --port 8901
```

`http://127.0.0.1:8901` を開きます。`--dataset-root`（別名 `--data-root`）は
`dataset.json` を含むroot、`--video-id` は繰り返してvideoを限定できます。
GPU・checkpoint・DINO tokens・splitはレビューに不要です。注釈の生成・修正は行いません。
対応形式・学習契約は[SLCS README](../../README.md)、scene publicationの正本は
`src.tennis_scene.generate_dataset.manifest` と `pipeline.storage.scene_index` を参照してください。
旧scene archiveや旧PLCS/BLCS経路は読みません。

## 見るもの

左からclipを選び、3D再生を停止・seekして右panelのcameraを選びます。
選手はreaderと同じnear→far順のrootとyaw、球は位置・軌跡で、単位はmです。
3Dの品質mask外は非表示で、軌跡は欠損区間を結びません。
初期画角・プリセット・リセットはSLCS内でコートと有効教師の全軌跡にfitします。
「全体fit」ボタンでも同じ操作ができます。cornerの向きは保ち、距離とtargetを調整します。
右panelは同じexportのRGB・pose17・ball UV・court14と教師qualityを確認します。

- RGB上のposeと球は小さい点、疑似教師の投影は紫の輪です。各overlayを切り替えられます。
- P0/P1のcropで遠い選手のposeを拡大できます。枠内2D観測がないとcropは利用不可です。
- 有効/無効、weight、全cameraのcoverage、root/heading/reconstruction gateの棄却理由を表示します。
  無効教師のゼロ座標を有効位置として描きません。
- 教師maskのtimeline（P0/P1/ball）はclickまたは矢印キーでseekできます。
  前/次の無効frameボタンは少なくとも1教師が無効なframeへ移動して再生を停止します。
- source detailsでcameraの元source、同期offset、元frame範囲、letterbox、教師representation、
  元player slot、ID scope、座標系を確認します。元sourceそのものは開かず、clip mediaだけを表示します。
- DINO完成markerとsplit fileの存在は表示しますが、内容検証・学習準備の完了判定はしません。

再生中のRGB取得はCPU負荷を抑えるため間引きます。画像・2D観測・教師は同じ取得frameで
一緒に更新し、3D clockより遅れる間は表示中のcamera/frameと読込状態を明示します。
停止してseekすれば右panelのreadoutで一致を確認できます。RGB欠落時は座標だけを暗背景に
描き、media欠落を明示します。別clip読込中や取得失敗時に旧overlayを正常表示として残しません。

## 教師と投影の意味

この画面のteacherはモデル・幾何処理から得た**疑似ラベル**です。実測GTを含みません。
2Dの不可視は「観測なし」であり、物体不在は未確認です。観測が枠内/枠外かも区別します。
coverage閾値を満たしてもreconstruction/heading maskが無効なら教師は採用されません。
qualityは`configs/data/default.yaml`と`data.quality.build_label_masks()`の正本を使います。
学習用windowの抽出・augmentationは実施しません。

保存`court_reference.camera_fits`にK/R/tがあるときだけ、
`PinholeCamera.project()`で高さを含む教師を投影します。近似単一平面pinhole fitの
残差・校正frameを表示し、歪み補正なし・独立校正GTなしと明記します。
K/R/tがなければ3D投影は利用不可です。地面homographyを3D投影の代用には使いません。
球の観測↔教師投影差は再構成に使用した観測との整合であり、独立な3D精度評価ではありません。
courtの`static_first_frame_broadcast`など保存temporal policyと観測frameも表示し、
毎frameの新規検出と混同しないようにします。

## 構成と整合性

`dataset_service.py`はmanifestを遅延参照し、`load_clip_arrays()`で読んだ標準教師を
`inspection.py`へ接続します。近い2clip分の小配列だけをcacheし、SMPL verticesは保持しません。
RGBは要求frameのみCPUでdecodeしてJPEGを最大8枚cacheし、capture handleは毎回解放します。
`dataset_web.py`のtask専用assetsとAPIが共有3D UIへpanelを追加します。

revisionはmanifest、完成marker、scene index、現行component descriptor、export/sidecar、
clip media、DINO marker、split fileのstatから計算します。変更があればchecksum・component
lineageを再検証し、旧revisionのframe/buffer/image要求には409を返します。
root外の参照やmedia symlink、不正digest・shape・publicationはエラーです。

追加APIは以下の2つです。すべてGETで、`form`=video ID、`scene`=clip名です。

- `/api/inspection/frame?form=...&scene=...&camera=...&frame=...&revision=...`
- `/api/inspection/image?form=...&scene=...&camera=...&frame=...&revision=...`

## 検証

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ./scripts/run_in_repo_venv.sh python -m pytest -n 2 \
  tests/unit/tasks/slcs/visualization/review \
  tests/unit/tasks/base/visualization/review/test_web_extensions.py
node --test tests/unit/tasks/slcs/visualization/review/overlay.test.mjs
PLAYWRIGHT_MODULE=/path/to/playwright-core \
  SLCS_REVIEW_URL=http://127.0.0.1:8901 \
  node tests/e2e/tasks/slcs/dataset_review_browser.cjs
```

unit fixtureはテストにだけ使います。レビュー画像の証拠は実データの稼働画面から取得します。
