# 全人物のフレーム別3D骨格と観測割当

保存済みの全画面DINOと全検出行のViTPoseから、同じ元フレームの観測をコート座標へ復元するCPU入口です。出力はフレームごとの骨格候補と観測行の対応で、人物IDの有効範囲はそのフレームです。

## 実行

入力clip storeの `scene.json` が指す `person_detections` v2、`person_pose_features` v2と `local_court_calibration` v1を使います。DINOは全画面・元行をすべて保持した版を使い、poseのcamera・`frame_offsets`・`source_rows` を検出と照合します。poseは同じ版の検出、校正は共有 `court_observations` v3から生成したものに限定します。上流に指定外の依存がある実行は停止します。

`--side-digest` でボール由来の動画共有判定を指定します。判定からコート・driftのdigestを解決し、判定済みclipの集合・カメラ順・レイの基準となるカメラ行列・元動画の実ファイルSHA-256を照合します。検証済み動画内で単独判定が未決のclipは警告を残して進み、共有判定との矛盾、判定なし、digest・映像・校正の不一致では停止します。meiji_4cam の各storeは #2241 の記録から取得できます。 [保存済み共有判定とコートの配置](../../../knowledge/runs/run-court-side-video-shared-20261118/reproduce.md)

```bash
.venv/bin/python -m src.tennis_scene.multiview_pose \
  --clip-store "$CLIP_STORE" \
  --side-store "$SIDE_STORE" --side-digest "$SIDE_DIGEST" \
  --court-store "$COURT_STORE" --drift-store "$DRIFT_STORE" \
  --output "$POSE_STORE"
```

判定がdriftを参照する場合は `--drift-store` が必須です。出力には専用のclip storeを使います。同じ入力と設定で再実行すると同じ内容アドレスの成果物を採用します。

## 推定

肩・腰のうち3関節以上のレイが整合する別視点の検出行を結び、全ペアに辺がある多視点の組を候補にします。COCO17の各関節について、2視点で作った点のうち最も支持視点の多いものを選び、支持レイで最小二乗の再推定を行います。近接するカメラレイ、カメラ後方、範囲外、再投影不一致は無効にします。確信度の高い関節の大きな再投影不一致、または復元した骨長の上限超過は、その骨格候補を棄却します。

同じ観測行を取り合う候補群ごとに、支持と残差から計算したスコアをMILPで最大化します。各行の使用は高々1回で、人物数は推定結果に従います。競合群では、最良解に無い候補を少なくとも一つ採用する代替解も求めます。最良解から外れる候補について、最良解のスコアで正規化した差と接戦maskを保存します。閾値と骨長上限の正本は `PoseConfig` と `geometry.py` です。 [観測割当の考え方](https://arxiv.org/html/2002.12625v1#S4)

## 3D追跡への受け渡し

成果物は `multiview_pose` ノードの `frame_pose_candidates` v1 です。`storage.read_candidates()` は記述子、配列、公開契約を検査します。`scene.json` のsourceにclip ID・カメラ順・元動画のFPSとSHA-256、ノードのprovenanceに校正・side・入力artifact・閾値の版を保存します。

| 配列 | 軸と意味 |
|---|---|
| `frame_offsets` | 元 `T+1`。空フレームは同じoffsetが続き、候補IDはこのslice内の位置 |
| `source_rows` | `M × V`。採用したカメラごとの元DINO行。未使用は `-1` |
| `positions_m`, `joint_valid`, `joint_rejection_codes` | `M × 17` 関節のコート座標(m)と実測の有効性。無効座標は0 |
| `inlier_mask` | `M × V × 17`。各関節の復元を支持した観測。各有効関節に2視点以上 |
| `reprojection_px`, `reprojection_valid` | `M × V × 17`。有効な3Dと観測の残差(px)。支持外でも残差を測れた観測を含む |
| `scores`, `competing_assignment`, `relative_score_gap`, `ambiguous` | 採用候補のスコア、競合解との差と接戦mask |
| `detection_offsets`, `row_rejection_codes` | カメラごとの元検出フレーム軸と各行の診断。採用、視点不足、退化、幾何不一致、混入、競合 |

行に複数の棄却候補がある場合、診断には最も先まで進んだ棄却段階が残ります。骨格位置の有効性は座標値と独立してmaskで判定します。

## 制約

適用には、対象動画に対応するボール由来の共有判定が必要です。人物レイによる向き判定はmeiji_4camで旧判定68/68、共有判定69/69と一致した試作で、別収録への適用候補です。校正は単一平面のpinhole近似です。近接・遮蔽・校正ずれがある場合、レイと骨長が整合する別人の観測は曖昧になることがあります。 [設計試作の結果](../../../knowledge/runs/run-i2444-multiview-pose-probe/README.md)
