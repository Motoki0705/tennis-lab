# Court Review

公開済みCourt datasetを目視確認する、読み取り専用のローカルWeb UI。
版・生成元・採否・split・target courtを確認し、3Dカメラと生成画像／保存教師を連動して検品する。

```bash
/home/kamimura/projects/tennis-lab/.venv/bin/python -m src.synthetic_data_generation.scripts.review_court_dataset \
  --data-root /home/kamimura/projects/tennis-lab/data \
  --scenes-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes \
  --port 8905
```

ブラウザで `http://127.0.0.1:8905` を開く。GPU・フロントエンドのビルド・外部CDNは不要。
版・教師・保存形式の正式契約は [Court dataset contract](../README.md) を参照。

## 検品の流れ

1. 「データ体系」で、設定したscenes root内の公開ownerを確認する。
   geometryとstorageの版を分け、source videoは保存された`run.json`から読む。
   現在の設定のvideo名を既存publicationの生成元として代入しない。
   sourceや生成状態がない場合は「未記録」。scene artifactだけではdataset採用と扱わない。
2. 左のシーン／splitから軌道を選択する。中央でコートと生成カメラを3D表示し、
   右に選択軌道の全採用画像をframe・view順に表示する。
   train／validation／testを色分けし、3D上の軌道クリックでも選択できる。
   画像のhover／focusで対応カメラとtarget courtの可視性を確認する。
3. 「教師ラベル」「生成RGB」「RGBと教師を比較」を切り替える。
   描画courtは標準で「target限定（学習教師）」とし、現行consumerのKP14対象と一致させる。
   「全court参考」は生成scene全コートの疎な投影を表示する。target可視数はtargetだけの集計。
   画像クリックで元解像度表示を開き、frame／view、camera、target、保存KPの物理index、
   画面内／画面外／カメラ後方とrenderer可視性を確認する。点の行のtooltipはpixel座標。
   左右キー／ボタンで前後、Escapeで閉じる。長いIDと数値は選択パネルに置く。
4. 「統計」でtarget court×splitを確認する。0セルはそのtargetへの採用sampleがないことを表す。
   背景に映るコートの数とは異なる。偏りが意図したものかは分割方針と照合して判断する。
5. reject件数・理由と候補の疎な記録を確認する。候補クリックでは専用dialogにcamera、
   target、理由、投影の保存状態を表示し、採用画像は表示しない。
   候補詳細が保存されていなければ「未記録」。reject画像は未保存。
   pre-render候補の`renderer_visible=null`と不可視`false`を区別する。

画像はsample storeのRGBと疎なlabelsに結び付ける。塗りつぶし点はrenderer-visible、
輪郭点は画面内・renderer不可視。画面外の点は画像には描画しない。
生成truthはNHT再構成とalignmentに依存し、人手GTや独立QA済み画像ではない。
completedは生成時ゲート通過を表す。画像・教師・splitの編集は行わない。

軌道の線は採用カメラ位置をframe順に結ぶため、未採用位置を含む完全な計画曲線ではない。
同一位置に複数viewがある場合も全画像を並べる。3Dは先頭コートのm座標系、方向線は2m。
ドラッグで回転、Shift＋ドラッグで移動、ホイールで拡大。「視点を戻す」／Homeで全体へ戻る。

## 実装とキャッシュ

`records.py`が保存値の可視性・target／split・rejectを要約し、`service.py`が版別readerで
publication／3D要約／画像を読む。`web.py`は読み取り専用API、`static/scene.mjs`は3D操作、
`static/app.js`は選択連動を担当する。catalogはpacked metadataを読み、全画像をdecodeしない。

画像は表示付近のみ遅延読み込みし、合成は同時2件、画像キャッシュはraw／overlay合計128件、
シーンキャッシュは2件。一覧は最大480pxのJPEG、全画面は元解像度のJPEG。
画像APIの既存呼出しは全court表示を保持し、UIは`label_scope=target`を明示する。
targetの保存投影がない場合は取得エラーとし、全courtへの自動切替はしない。
RGB表示は保存RGBのdecode結果を表示用JPEGへencodeする。rendererや推論は呼ばない。

manifest／alignment／packed index／source run／resolved configのrevision変更は取得を拒否して
再読み込みを要求する。切替中の古いscene／sample応答は破棄する。
正式ownerを変更せず配列ファイルだけを書き換える運用は想定しない。
この画面は全配列の完全性validatorや学習開始画面ではない。

## 検証

```bash
.venv/bin/python -m pytest -n 2 tests/unit/synthetic_data_generation/dataset/court/review tests/unit/synthetic_data_generation/visualization/test_overlays.py
node --test tests/unit/synthetic_data_generation/dataset/court/review/scene.test.mjs
# 公開実データのサーバーに対する読取専用ブラウザ検証
PLAYWRIGHT_MODULE=/path/to/playwright-core COURT_REVIEW_URL=http://127.0.0.1:8905 \
  node tests/e2e/synthetic_data_generation/court_review_browser.cjs
```

## 明示したデータ版で方式を比較する

train専用scale profileは、固定scene ownerを探索する既存モードとは別に、
データ版manifestまたは生成器が書いたcatalogを明示して起動する。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.review_court_dataset \
  --catalog tmp/court-scale-review/run-03/review-catalog.json --port 8906
# 個別指定でもよい（同じsceneの異なる方式・recipe・版を同時に渡せる）
.venv/bin/python -m src.synthetic_data_generation.scripts.review_court_dataset \
  --dataset-version /path/to/output/AAA.dataset.json /path/to/output/BBB.dataset.json \
  --port 8906
```

`http://127.0.0.1:8906` の左上「方式・データ版」を切り替えると、同じ実sceneを
維持して比較できる。選択keyと画像cacheはデータ版・sceneに結び付き、元のscene名を
別名として保存しない。版切替中の古いscene／sample応答は既存のsequence guardで破棄する。

scaleモードの3D表示は **点線＝計画曲線、点＝採択camera、赤×＝棄却camera**。
採択点だけを結んで棄却区間を跨ぐ線は描かない。開線分・開Bézierは始点へ閉じず、
未撮影の終点も計画曲線に含める。採択0件の軌道も一覧・計画線・診断を残し、
画像欄に「採択0件・画像なし」を表示する。曲線自体が境界検査で落ちた場合は理由のみを示す。

「比較設定・採否診断」でmethod・recipe・seed・密度段・候補／描画／再利用／採択／棄却数、
品質設定、対象KP≥4／ROI穴≤1%の感度集計を確認する。この2つは、hard gateを明示指定
していなければ診断だけである。画像hoverでは対象renderer KP数とROI穴率も表示する。
RGBとtarget限定教師を見て、線・地面・ネットのぼけ、未観測領域の破綻、target切替、
局所方式の近接画像の反復を確認する。ROI穴率0だけでは画質合格を意味しない。
保存されていないreject RGBに他の画像を代入しない。

scene alignmentはハッシュでpinした資産snapshotを使う。storeは版manifestの明示rootから
解決し、複数の共有storeを1sceneとして表示する。起動時にschema・hash・membership・
train専用の計数・v3幾何を検証するが、全JPEGのdecodeは行わない。既存の画像cache128件、
scene cache2件、overlay並行2件を共有する。版・request・curve plan・資産・QA・descriptor・
index・JPEG shardの変更はrevisionを変え、古い取得を拒否する。

合成版・保存・生成コマンドの正本は[scale profile](../README.md#explicit-scale-comparison-profile)。
方式やD2/D5/D6の最終採用はユーザーが比較後に親issueへ記録し、UIの閲覧から自動採用しない。

ブラウザでのscale profileの受入検証は、生成済みの4scene×5方式を含むcatalogを指定した
上記サーバーに対して実行する。GPUは使わない。target限定の14KP、採択0件の表示、
開曲線の終点、method/scene/version切替、古い応答の破棄を検査し、B00の各版を撮影する。

```bash
PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chrome \
  COURT_REVIEW_URL=http://127.0.0.1:8906 \
  COURT_SCALE_SCREENSHOT_DIR=tmp/court-scale-review/run-03/screenshots \
  node tests/e2e/synthetic_data_generation/court_scale_review_browser.cjs
```

比較データを生成したworktreeから起動すること。`src`はcwdからimportされる。
`review-catalog.json`、`*.dataset.json`、JPEG store、スクリーンショットはgit管理外の
同じoutput rootに残し、catalogの絶対path参照を保つ。密度追加版やjitter 0m版を同じ
catalogへ渡すと、左上のn/k/jitter/seed/版で識別できる。採用を決める際は、
候補／採択枚数や層の空きを含めて確認し、親issueへ採用方式とD2/D5/D6を記録する。
