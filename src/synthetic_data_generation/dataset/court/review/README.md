# Court Review

公開済みCourt datasetを目視確認する、読み取り専用のローカルWeb UI。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.review_court_dataset \
  --data-root /home/kamimura/projects/tennis-lab/data \
  --scenes-root /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes \
  --port 8778
```

ブラウザで `http://127.0.0.1:8778` を開く。GPU・フロントエンドのビルド・外部CDNは不要。

- 左：シーン切替とスクロール可能な軌道一覧。
- 中央：コートと生成カメラ軌道を3D表示。train／validation／testを色分けし、一覧または3D上の軌道クリックで選択する。ドラッグで回転、Shift＋ドラッグで移動、ホイールで拡大。視点を戻すボタン（canvas上ではHomeキー）で全体へ戻る。
- 右：選択軌道の**全採用画像**をフレーム・view順に表示。約5.5枚が見えるスクロール一覧で、画像と保存教師ラベルを一枚に合成する。クリックでブラウザ画面全体に表示。左右キー／矢印ボタンで前後、Escapeで閉じる。サムネイルにカーソルを置くと3D上の対応カメラ位置と向きを表示する。
- ページ下部：画像数、軌道数、コート数、採用率、split・形状・coverageの統計。画面上部は目視確認に使う。

軌道の線は採用カメラの位置をフレーム順に結んだ表示で、未採用位置を含む完全な計画曲線ではない。同一位置に複数viewがある場合も、右側にはすべての画像が並ぶ。3Dの座標は先頭コートのメートル座標系。カメラ方向線は見やすさのため2mで表示する。

画像は学習用の `rgb.npy` と、そのsampleに紐付く保存済みlabelsを使う。既存のバージョン別Court overlayを再利用し、画像上のクラス名・コート線・点は保持、ヘッダーや凡例パネルは省く。塗りつぶし点はrenderer-visible、輪郭点は不可視。全画面は元解像度、一覧は最大480pxへ縮小したJPEG表示。教師や元データは変更しない。これは目視確認用であり、全配列の完全性検査や学習開始を行う画面ではない。

画像は表示付近のみ遅延読み込みし、合成の同時実行は2件、画像キャッシュは128件、シーンキャッシュは2件に制限する。シーン切替中の古い応答は破棄する。manifest/alignmentのrevisionが変わった場合は古い画面からの取得を拒否し、再読み込みを要求する。正式ownerを変更せずに配列ファイルだけを書き換える運用は想定しない。

## 実装

`service.py` が保存データの読み込み・3D要約・画像合成、`web.py` が読み取り専用API、`static/scene.mjs` がCanvasの3D投影と操作、`static/app.js` が画面間の選択連動を担当する。

## 検証

```bash
.venv/bin/python -m pytest -n 4 tests/unit/synthetic_data_generation/dataset/court/review tests/unit/synthetic_data_generation/visualization/test_overlays.py
node --test tests/unit/synthetic_data_generation/dataset/court/review/scene.test.mjs
# 実データを公開したサーバーに対するブラウザ検証（読み取り専用）
PLAYWRIGHT_MODULE=/path/to/playwright COURT_REVIEW_URL=http://127.0.0.1:8778 \
  node tests/e2e/synthetic_data_generation/court_review_browser.cjs
```
