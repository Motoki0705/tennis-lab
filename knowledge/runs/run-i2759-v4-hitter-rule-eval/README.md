# V4ルールベース打者推定の映像監査と評価

明治4カメラの frozen V4 に採用されたshotは330件。固定splitはtrain 47クリップ219件、val 11クリップ60件、test 11クリップ51件。元の試合映像は3分割で共通する。clip ID、元映像の時刻区間、各カメラの元フレーム区間、クリップMP4のSHA-256を照合し、分割をまたぐ重複はいずれも0件だった。 [valレポート](val_evaluation.json) · [testレポート](test_evaluation.json)

## 映像による打者の確認

各shotの中心とその前後をcam0〜cam3で同期して表示した。前後幅は8フレームと注釈の時刻誤差幅の大きい方で、クリップ境界では範囲内に収めた。各画像の骨格と `index/ID` はV4の三角測量関節とローカルtrack ID。白円は投影した3Dボール、黄円は該当カメラの2D観測。打者はラケットの準備、振り出し、振り抜きと相手・ペアの動きから判断し、映像中の選手をIDに対応付けた。

Codexの同一作業者が推定結果を閲覧する前に全111件を確認し、GTと監査目録をコミットした。108件は打球近傍の投影ラベルで確認した。残り3件は打球近傍で該当選手の投影ラベルが欠けていたため、同じクリップの別shotで服装とコート位置を照合し、`confirmed_with_track_gap` とした。 [GT](visual_ground_truth.csv) · [監査目録](visual_audit.json) · [GT固定コミット](https://github.com/Motoki0705/tennis-lab/commit/d64228b92d179e56b1758d6110fc2585d45808bf)

監査目録は監査対象の `clip.json`、実際にデコードした4本のMP4、SceneResult、イベントcomponentと画像パネルのSHA-256を記録する。元動画4本も個別にハッシュを再計算した。V4の `SHA256SUMS` のダイジェストは配布時の検証記録と一致した。 [元動画のハッシュ](raw_videos.sha256.txt) · [V4配布時の検証](../run-i2733-meiji-event-labels-v4/verification.json)

## 閾値の固定と結果

valの60件で手首到達距離4値とスコア差4値の16通りを比較した。誤確定数が最小、正しい確定数が最多、既存の手首到達距離に近い、次点差が大きい、の順に選んだ。 [val比較](val_calibration.json)

レビュー後の `court-hitter-v2.217835f4f9d0` は、球の観測が不足し、両手首欠測の同側選手が到達圏内にいるときに `same_side_competitor_wrists_missing` で保留する。ネット際では順位にかかわらず到達可能な反対側・側不明の候補を調べる。既存の映像GTを使ったvalの再比較では手首到達距離1.30 m、次点差0.18が再び選ばれた。全閾値を引き継ぎ、ロジックを固定したコミットでtestを再評価した。 [ルール固定コミット](https://github.com/Motoki0705/tennis-lab/commit/4ec290e9758880374b3a9bff7cb8d5d90d596c73)

| Split | 全shot | 評価可能 | 確定 | 曖昧 | 未確定 | Coverage | 確定時正解率 | 誤確定率 | 全体正解率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 60 | 60 | 59 | 0 | 1 | 98.33% | 100% | 0.00% | 98.33% |
| test | 51 | 51 | 49 | 0 | 2 | 96.08% | 100% | 0.00% | 96.08% |

Coverageは確定数÷全shot。確定時正解率は正しく確定した数÷GT付きの確定数。誤確定率は誤確定数÷GT付き全shot。全体正解率は正しく確定した数÷GT付き全shot。誤確定は両splitとも0件で、誤確定事例一覧は空。 [valレポート](val_evaluation.json) · [testレポート](test_evaluation.json)

| Split | Clip / source event | Frame | GTのlocal ID | 保留理由 |
|---|---|---:|---:|---|
| val | `clip_022 / e001` | 155 | 20 | `too_far`。本人の手首とrootの有効観測が窓内で0。別選手の最上位部位は球から4.87 m |
| test | `clip_004 / e001` | 70 | 22 | `too_far`。本人の手首とrootの有効観測が窓内で0。別選手の最上位部位は球から10.28 m |
| test | `clip_019 / e009` | 471 | 4 | `too_far`。本人の手首とrootの有効観測が窓内で0。別選手の最上位部位は球から7.62 m |

評価は同一試合の固定カメラ条件とV4のイベント時刻・コート側を所与とする。GTのID対応にもV4の投影を使用しており、投影と追跡品質の影響が残る。別の注釈者による一致率は未計測。 [監査目録](visual_audit.json)

## 再現

リポジトリルートで実行する。画像はクリップごとの `<source_event_id>-wide.jpg` と `<source_event_id>-close.jpg` に出力される。行がカメラ、列が時刻。監査目録のファイル名とSHA-256から個別パネルを照合できる。特定のクリップだけ描画する場合は `--clip-id video_000/clip_022` を追加する。

```bash
run=knowledge/runs/run-i2759-v4-hitter-rule-eval
benchmark=data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4
manifest_sha=$(jq -r .manifest_sha256 "$run/visual_audit.json")
audit_out=$(mktemp -d)

.venv/bin/python -m src.tennis_scene.analytics.hitter_eval \
  --benchmark "$benchmark" --manifest-sha256 "$manifest_sha" \
  render --output "$audit_out"

PYTHONPATH=. .venv/bin/python "$run/calibrate.py" \
  "$benchmark" "$manifest_sha" "$run/visual_ground_truth.csv"

for split in val test; do
  .venv/bin/python -m src.tennis_scene.analytics.hitter_eval \
    --benchmark "$benchmark" --manifest-sha256 "$manifest_sha" \
    evaluate --ground-truth "$run/visual_ground_truth.csv" --split "$split"
done
```

画像生成にはPython 3.11.15、NumPy 2.3.5、OpenCV 5.0.0を使った。描画環境を変更するとJPEGのバイト列が変わり得る。

PostgreSQL 17.12で #2767 の `ingest` CLIから凍結V4を取り込み、`hitter --benchmark "$benchmark" --ingest-run 1 --owner-team 1` で全69クリップの330件を保存した。確定321件、曖昧0件、未確定9件。照会で330件を取得し、全件で取込時のイベントcomponentのSHA-256と推定行の記録が一致した。再実行の更新は0件。実PostgreSQL結合テストでは、欠落クリップ、shotのstatus不一致、clipとshotの出典ハッシュ不一致で推定行が残らないことと、CLIのJSON/CSV往復・一意制約を確認した。 [取込コードの固定コミット](https://github.com/Motoki0705/tennis-lab/commit/521bc7a76da04296e7a0a0ef971313a76459ddd6) · [DB CLI](../../../src/tennis_scene/analytics/README.md) · [結合テスト](../../../tests/integration/tennis_scene/analytics/test_hitter_postgres.py)
