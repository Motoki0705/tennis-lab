# BLCS Dataset Review

既存の `data/blcs/single_object` / `physical_v1` を読み取り専用で検品するWeb UI。
保存2D ball・CourtKP20・可視性、3D位置・速度、camera、hit/bounceの保存metadataを
同じフレームで確認する。物理シミュレーションの合成truthであり、RGBは保存されていない。
起動は[BLCS利用ガイド](../README.md)、共有API・座標契約は
[共有review基盤](../../../base/visualization/review/README.md)を参照。

## 検品画面

- 共有3Dビューの再生・シークに同期する6カメラ比較。カメラ選択で1台を拡大できる。
- 保存点とメートル単位の教師からの再投影を別記号で重ね、pixel差・可視性の不一致を表示する。
  画像領域を破線で示し、有限な画像外座標も保持する。camera後方では再投影UVを未定義として扱う。
- 保存CourtKP20をすべて表示する。学習readerが選ぶ `num_court_kp`（既定14）は
  保存点数とは別の設定で、検品時に切り詰めない。
- 3Dは保存 `ball_pos_world` / `ball_vel_world` を表示し、保存normalizedペアを
  [正規化契約](../../../../utils/README.md)で復元して照合する。数値誤差の検品閾値は `1e-4`。
- `shots.t_*` はoutputの0始まりframe。秒は `frame / fps_out` で算出する。
  `hit` は `t_start`（shot開始）、`return` は保存 `t_return`、`net` は保存 `t_net`。
  `return_type=none`でもreturn時刻が保存されるため、次hitの実行を保証する印ではない。
  次shot以降のbounce候補・scene範囲外・未記録 `-1` を区別し、候補は再生イベントに入れない。
  全保存metadataは折りたたみ表で確認できる。
- ファイル欠損、未知のvisibility、非有限UVを明示する。2D欠損時にも3D教師から観測を補完しない。
  配列shape・型・CourtKP契約違反は読取エラーになる。シーン切替／失敗時には前の検品内容を消す。

現行対応はsingle_objectのみ。chunkedは同じ契約のtrain供給方式であり、別datasetではない。
生成・augmentation・学習・推論は行わず、元データを書き換えない。

## 構成・API

| ファイル | 責務 |
|---|---|
| `dataset_service.py` | 現行形式の制約、共有3Dサービスとの接続、revision照合、split表示 |
| `inspection.py` | 保存配列の読取、再投影と正規化診断、保存イベントの区間分類 |
| `web.py` | 共有extensionへtask-owned assetを登録し、検品APIを公開 |
| `static/blcs-inspection.mjs` / `.css` | scene/frameイベントに同期する検品パネル |
| `static/blcs-observation.mjs` | visibilityと画像外座標の表示ロジック |

`GET /api/inspection?form=single_object&scene=<id>&revision=<3Dと同じrevision>` は
保存観測・教師・診断を返す。`revision`は必須で、不一致は409。共有のパス制約を維持する。
2D・速度・normalizedペアの欠損はJSON `null`、非有限座標の成分も `null`と件数で返す。
可視性のfalse・未知・欠損を区別し、保存UVをゼロや再投影に置き換えない。

## 検証

```bash
./scripts/run_in_repo_venv.sh pytest -n 2 tests/unit/tasks/blcs/visualization/review
node --test tests/unit/tasks/blcs/visualization/review/observation.test.mjs
PLAYWRIGHT_MODULE=/path/to/playwright-core \
BLCS_REVIEW_URL=http://127.0.0.1:8900 \
node tests/e2e/tasks/blcs/dataset_review_browser.cjs
```

Unit testsの小さな配列fixtureは欠損・外れ・revision変更等の契約検証専用。
`test_dataset_review.py` とbrowser regressionは実在するローカルdatasetを検査する。
画面証拠は実在datasetで撮影し、全データの品質合格やモデル精度の証明とは扱わない。
