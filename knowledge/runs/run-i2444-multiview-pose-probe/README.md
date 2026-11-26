# Meiji の全検出行を使った人物レイと割当のCPU試作

[#2444](https://github.com/Motoki0705/tennis-lab/issues/2444) の設計調査で、meiji_4cam の保存済み入力を69 clip照合し、Meiji 3camの人手ラベル付き4 clipでフレーム別割当を比較した。試作は肩・腰の4関節を使う。全身骨格の検査とダブルスの割当精度は後続の実測に残る。

## meiji_4camの入力と向き

`#2363` の `outputs/i2363/i2363-all-v1/inputs/dataset/videos/video_000/clips/<clip>/annotations/tennis_scene/scene.json` は、各clipの採用済み成果物を指す。69 clip・276 camera/clipで `person_detection/<camera>` の `person_detections` v2、`person_pose_features/<camera>` の同名schema v2、`court_calibration` の `local_court_calibration` v1を確認した。4台それぞれの元46,228 frameに属する1,332,834検出行で、`frame_offsets` と `source_rows` が検出とposeの間で完全一致した。記述子と使用配列のSHA-256、pose記述子の検出依存、校正の依存閉包も確認した。旧経路で停止した14 clipでも上流の9成果物が揃っていた。旧 `court_side` は68 clip、旧 `camera_alignment` とscene exportは55 clipに存在した。 [#2363 の固定条件と全件記録](../run-i2363-trt-player-video/README.md)

元frameを12 frame間隔で計3,887本抽出した。元画像内かつheatmap peakが0.25以上の肩・腰を使い、共通する3関節以上で、2本のレイの最近接点の中点が前方、角度が1°以上、かつ両視点の誤差が1080p換算で24 px未満のペアを採点した。探索範囲は `|x|, |y| ≤ 40 m`、`-0.2 ≤ z ≤ 3 m`。2視点間は一対一の最大重み割当を使い、カメラ対ごとの相対half-turn 2通りの結果を合計して全8仮説を順位付けした。各仮説を同じ照合可能行数で正規化した。実験上の採否条件は、8 sample frame以上の支持を持つ辺で全カメラが基準に連結すること、正規化scoreが0.05以上、次点との差が0.02以上。 [試作コード](probe.py)

全69 clipで `[false, false, true, true]` が首位で、単独でも実験上の採否条件を満たした。旧 `court_side` がある68 clipとは68/68一致した。旧判定が絶対支持条件で止まった `clip_012` は人物レイでscore 0.942、margin 0.105。全clip集計のscoreは0.942、marginは0.146で、単独clipのmarginは0.082〜0.200だった。ボール由来の #2241 共有判定とも全69 clipで一致した。比較ではカメラ順、clip、全276元動画の保存SHA-256と時間軸、校正行列、共有コート・driftの出自を照合した。同一収録内での一致であり、別会場、画角変化、画面遮蔽や対称配置に対する再現性は未評価。 [全仮説と照合結果](side.json)・[#2241の判定](../run-court-side-video-shared-20261118/summary.json)

#2363 のstoreにボール点はなかった。#2241 は既存4 clipの出力を再利用し、残り65 clip×4 cameraのX3D/CSN+DPT T32 BF16検出とボール点・校正の補完を67分39秒、CPU集計・照合を16.74秒で実行した。フレーム数に比例する近似では、同条件の全69 clip分は約70分。保存済み動画判定をpinして使えるmeiji_4camでは追加GPU推論が不要になる。 [前段の計測](../../nodes/court_side/000007-run-court-side-video-shared-20261118.md)

## Meiji 3camのフレーム別割当

人手ラベルがある `000/000`、`000/007`、`001/001`、`002/013` の全3,497元frameを対象にした。#964 r6のCOCO DINO全画面 .30と、r7で計算したFP32・flip testありのViTPose-H全40,531行を使用した。r7のNPZでは `rows`、`offsets`、`boxes`、`scores`、`poses` と検証用の出自を読み、元検出との完全一致を確認した。外観配列へのアクセスは不要。校正は#933の `local_court_calibration` v1を読み、#932の人手確認済みボール点に基づく向き `[false, false, true]` を固定した。人物ラベルとの照合は全予測の確定後に行った。 [r7の保存条件](../../nodes/person_tracking/000007-run-i964-coco-person-features-r7-20260930.md)・[Meiji 3camのラベル契約](../../../src/tasks/player_association/README.md#meiji-3cam-のラベル初期v1観測)

肩・腰のレイが整合する2視点の行ペアを候補とし、3辺が整合する3視点の組も候補に追加した。3視点候補の重みは辺の重みの和。全く同じ候補と重みで、重み順に行を確保する貪欲法と、各検出行の使用が高々1回となる0–1整数計画を比較した。同じ行を同じframeで再使用していないことを両方式で検査した。ラベルは既存の採点器のIoU 0.5で照合した。表は既存の `pair F1`。 [試作コード](assignment_probe.py)・[採点器](../../../src/tasks/player_association/evaluation/metrics.py)

| clip | 元frame | 貪欲法 | MILP | 候補数の最大 |
|---|---:|---:|---:|---:|
| `video_000/clip_000` | 1,010 | 0.7202 | 0.7377 | 25 |
| `video_000/clip_007` | 652 | 0.9214 | 0.9292 | 26 |
| `video_001/clip_001` | 1,300 | 0.8802 | 0.8640 | 36 |
| `video_002/clip_013` | 535 | 0.8819 | 0.8933 | 16 |
| 全clipのTP/FP/FNを合算 | 3,497 | 0.8472 | 0.8485 | 36 |

既存採点器は `non_player` の視点間リンクもFPに数える。全人物を扱うCの診断として両方に選手ラベルがある組を採点すると、micro F1は0.8676→0.8728。既存採点器のTP/FP/FNは貪欲法15,568/999/4,615、MILP 15,683/1,102/4,500。選手の組でのFPは貪欲法135、MILP 70で、各方式のTP・FNは既存採点と一致した。参照25,147 boxのうち25,146を照合し、生検出側の未照合は15,385 boxだった。このラベルが覆う範囲は選手と、選手でないことが確定したbox。 [clip別の指標と被覆](pair-f1.json)

677 frameでMILPが幾何スコアを改善した。`video_001/clip_001` はそのうち372 frameを占め、選手の組に限った場合もF1が低下した。ダブルスでは支持視点数、骨格全体の品質、曖昧な競合を含む別の重みを評価する。再実行時のソルバー関数内のP95は、clip別で貪欲法0.010〜0.015 ms、MILP 7.96〜8.68 ms。候補生成・三角測量の本実装にかかる時間は別途測る。合成2視点の片側で手首を200 pxずらしても、肩・腰だけの試作スコアは同じだった。手足の混入検出には関節残差と骨長の検査が必要になる。

## 再現

リポジトリrootから、#2363・#2241の保存済みworktree outputs、Meiji 3camのdataset、#932・#933・#964の保存済みoutputsを読める環境で実行する。既定のパスと配置が異なる場合は `I2444_SHARED_OUTPUTS` に共有worktree outputsのroot、`I2444_ARCHIVED_OUTPUTS` に旧outputsのE側rootを指定する。

```bash
bash knowledge/runs/run-i2444-multiview-pose-probe/reproduce.sh
```

結果は `outputs/i2444/planning/replay/{side.json,pair-f1.json}`。環境はPython 3.11.15、NumPy 2.3.5、SciPy 1.17.1。CUDAを隠した再実行は、入力照合込みで向き29.68秒、割当24.35秒。初回はそれぞれ55.41秒、66.19秒で、時間欄以外の出力は再実行と完全一致した。
