# 3D骨格の時間追跡と選手選別

[`multiview_pose`](../multiview_pose/README.md) の `frame_pose_candidates` v1をCPUで追跡し、コート座標系でclip内の人物IDを割り当てます。入力はフレーム別3D骨格と、その元観測・競合の情報です。

```bash
.venv/bin/python -m src.tennis_scene.multiview_tracking \
  --candidates "$POSE_STORE" --candidate-artifact "$POSE_ARTIFACT_ID" \
  --players-per-side 2 --output "$TRACK_STORE"
```

入出力はどちらも1クリップのstoreです。 `--players-per-side` はシングルスで1、ダブルスで2を明示します。 `--candidate-artifact` は省略すると入力storeの現在の参照を使います。実際に読み込んだ成果物ID・記述子SHA-256、設定、実装digest、選別領域を出力記述子に保存します。同じ入力と設定で実行すると同じ内容アドレスを採用します。

## 時間窓内の割当

元FPSで1.2秒の窓を0.5秒ずつ進めます。候補を頂点、時間順の許容リンクを辺とするパス被覆を、各候補に前後それぞれ高々1本の辺を許す疎な二部マッチングで解きます。前半0.5秒に到着するリンクを確定し、残りの候補は次の窓で再評価します。確定済みの末端はリンク可能期間中、同じIDの接続先候補として引き継ぎます。

位置の差と、共通する肩・腰の平行移動を除いた形状差で最初の割当を解きます。支持カメラの変化、とくに2視点を含む場合は位置差の余裕を広げます。最初のパスで同じカメラ集合が支持した0.16秒以内の骨格から前向き・後ろ向きの速度を求め、両側の予測残差と速度差を入れた割当を解き直します。候補のスコア、競合の相対差、欠測間隔もリンクの重みを変えます。最大0.4秒の欠測を1本のリンクでつなげます。窓、確定するフレーム数、欠測上限は元FPSに丸めます。重なりは少なくとも欠測上限と速度推定に必要なフレーム数を含みます。

時間リンクの曖昧さは、リンクの削除と、他のリンク1〜2本との交換で得た代替解のうち、最小の損失で判定します。この値は窓全体の次善解や確率を表しません。v1の代替割当はスコア差だけが保存されるため、時間窓の変数は採用候補の連結と分離です。設定根拠とMeijiでの計測は[評価記録](../../../knowledge/runs/run-i2445-multiview-tracking/README.md)を参照してください。

## 選手枠への断片の統合と選別

肩・腰の有効な3D座標の平均を胴体中心として、 `|x| ≤ 5.485 m` かつ `|y| ≤ 16.885 m` に実測があるフレームを数えます。領域内で滞在が多い側を各断片の側とします。各側の順位は、その側での滞在数に、約0.25秒の水平移動量の75パーセンタイルを使った最大10%の加点を付けて決めます。同順位ではclip内IDの昇順です。

各枠は時間窓で得た断片を核にします。単独で25%の滞在を満たして採用された断片は優先して保持し、空き枠には滞在・骨格の有効率・観測の確度から候補を選びます。側が同じで時刻が重ならない断片間に辺を張ります。最大1.5秒の欠測について、両端0.2秒の頑健な位置と速度、3つ以上の共通する骨長の中央値を使って連結を絞ります。断片の実測滞在と品質による利得からリンクの費用を引き、核を通る各側の全枠を整数計画で同時に最適化します。同じ断片は高々1枠に使います。

統合した枠の採用にはclip全長の25%を切り上げた領域内実測フレーム数が必要です。短い欠測も1.5秒までの統合による欠測も、採用された前後の実測を使って関節を補間します。足りない側では所定人数との差が `selection.missing_per_side` に出ます。アレーに滞在した実測数を含め、各IDの採否の根拠を保存します。 [閾値と実データの前後比較](../../../knowledge/runs/run-i2445-player-slot-integration/README.md)

## clip単位の成果物

出力は `multiview_tracking` ノードの `multiview_pose_tracks` v1です。 `storage.read_tracks(Path(...))` が記述子・配列・時間軸を検査して `PoseTracks` を返します。 `scene.json` のsourceがclip ID、カメラ順、元映像のFPSとSHA-256を示します。

| フィールド | 軸と意味 |
| --- | --- |
| `frame_offsets`, `track_ids` | 元 `T+1` と `S`。元frameのslice内をclip内IDの昇順で格納。空フレームはoffsetが同一 |
| `candidate_indices`, `source_rows` | `S` と `S × V`。上流のフレーム別候補を通し番号で参照し、元DINO行をカメラ順に保持。欠測は `-1` |
| `positions_m`, `measured_joints`, `predicted_joints` | `S × 17 × 3` と `S × 17` の2 mask。実測骨格は上流の値。遮蔽でつないだフレームは両端が有効な関節だけ線形補間。両maskが偽の座標は0 |
| `torso_m`, `velocity_mps`, `velocity_valid` | `S × 3` と `S`。胴体中心と局所速度。遮蔽中は補間区間の速度。速度が未確定の行はmaskが偽 |
| `event` | `S`。 `BORN=1`・ `OBSERVED=2`・ `OCCLUDED=3`・ `REAPPEARED=4`。 `observed` は候補を持つ行のmask |
| `frame_competing`, `frame_gap`, `frame_ambiguous` | `S`。Cのフレーム内割当の競合・相対差・接戦 |
| `temporal_margin`, `id_ambiguous` | `S`。連結・分離の代替損失とその接戦。比較対象なしは `-1`。遮蔽行には再出現リンクの値を入れる |
| `end_frames`, `end_reasons` | `K`。IDごとに末尾実測の次のフレーム。終了理由の観察期間は時間窓の欠測上限で、期間を満たす場合は `DISAPPEARED=1`、clipが先に終われば `CLIP_BOUNDARY=2` |
| `selection` | 人数設定、必要滞在数、ID別の観測数・領域内数・側別数・アレー内数・移動量・順位スコア・採否理由。採用IDは `selected_track_ids` |

統合版も `multiview_pose_tracks` v1で読み戻せます。記述子の `provenance.config.fragment_integration` が統合設定、 `provenance.fragment_integration.slots` が断片から出力IDへの対応です。各枠に核、時系列順の断片IDと、連結点の前後の実測frame・欠測フレーム数・ `long_gap`・位置残差・速度差・共通骨の数と長さの差を保存します。 `confidence` は幾何・時間の費用から求める0〜1の適合度で、 `objective_margin` はその枠のリンクを除いた場合の全体最適値との差です。差が0.08以下のリンクは `ambiguous` を立て、到着した実測行と隙間の補間行の `temporal_margin` と `id_ambiguous` に反映します。

たとえば選別済みIDを1フレーム分取り出すときは、 `frame_offsets[f:f+2]` で行を絞り、 `track_ids` が `selection.selected_track_ids` に含まれる行を使います。観測を使う処理では `measured_joints` と `source_rows`、軌跡の描画では必要に応じて `predicted_joints` も使います。最後の実測より後ろは終了理由に残ります。真のIDが区別できない接近や長い全視点遮蔽では断片化や誤連結が生じることがあります。
