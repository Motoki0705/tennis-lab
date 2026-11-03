# 統合比較の実装前テスト設計

対象は #1568、境界commit `cecb863797d3e5468f26005d1286e29c7eaf997a`。
既存loader・runner・scorer・error_samples・描画部品を再利用し、調整はしない。

| 守る契約 | 見逃したくない不具合 | 層・既存node | 予算 |
|---|---|---|---|
| 明示したnative peakごとに既存局所decoderを適用し、元格子で座標補正。paddingは0、状態を保存 | 全候補を全画面argmaxへ置換、XY/格子と元画素の混同、paddingへのaffine offset | decoder unitを拡張。既存 `test_known_centres_include_rotation_anisotropy_and_saturation` と `test_default_matches_main_decoder_arrays_and_preserves_camera_time_axes` を再利用 | 新規6ケース（既知2峰1、棄却2、軸/範囲/legacy拒否3）、各1秒以内、共有composition1回 |
| 明示的な候補座標をtop-Kへ渡せる。元evidenceは不変、overrideなしは既定配列と一致 | overrideを無視、camera/frame取り違え、元heatmapの複製や偽装 | top-K unitの既知非対称camera fixtureを再利用。既存 `tests/integration/tennis_scene/test_ball3d_topk.py` のmain回帰も実行 | 新規4ケース（既知平行移動1、camera/shape/範囲拒否3）、各1秒目標 |
| 組合せは選択viewだけのBA cameraを使い、camera-only比較は時刻0。held-outへ変更cameraを漏らさない | 全view BAの再利用、camera順逆転、offsetの混入、旧cameraへ戻る | run専用composition integration。制御したBA出力と実top-K/runner/scorerの接続を検証し、solver自体は既存 `test_court_ball_ba.py` の実solver/LOVO契約を再利用 | 新規1ケース（4fold）、5秒以内目標、GPU/実資産なし |
| 共通/追加/喪失の定義・20px境界・505境界・空集合・camera/frame joinと検算 | joinからcameraを落とす、片側の残差取り違え、採点総数不一致の黙認 | 既存 `test_topk_cohorts.py` とその実装を直接再利用。全行の実資産検算も停止条件にする | 既存5ケース、新しい重複テストは作らない |
| HTMLは相対動画リンク、画像の参照切れ0。全frameをstreamで処理 | 動画base64埋込、持ち運び時の絶対path、全frame読込 | 既存 `test_ball3d_media.py`、run専用reportの軽い公開入口テスト | 新規1ケース、1秒以内、重い動画fixture追加なし |

変更は既存fit/探索の接続だけ。srcの既定config/solver/基盤採点は変更しない。
新規ケースは手計算または独立pinhole式の値を期待値とし、実装の式の写しにしない。
テストの削除・統合はない。pytestは `-n 2` 以下、collect-onlyと全phase durationsを保存する。
既存974件程度を再利用し、追加テストの測定が5秒/1秒を超えれば理由を実測とともに記録する。
