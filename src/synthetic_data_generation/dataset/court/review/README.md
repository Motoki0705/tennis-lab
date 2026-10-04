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
