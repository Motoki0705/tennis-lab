---
id: run-slcs-court-db-meiji-rgb-v1
type: run
title: Meiji RGB線によるCourtデータベース検索・ECCの初回CPU評価
provider: codex
date: 2026-09-19
status: done
config:
  model: numpy-HOG-retrieval-top5-truncated-distance-homography-ECC
  data: meiji_3cam-3clips-9views
  database_cameras_per_view: 96
  image_size: [512, 288]
  camera_prior: v9-H-conditioned-zero-roll
  manual_usage: post-selection-evaluation-only
metrics:
  views: 9
  ecc_returned: 45
  ecc_attempts: 45
  manual_static_layouts: 3
  manual_points: 42
  baseline_manual_mean_px: 7.312831695399613
  initial_manual_mean_px: 17.388358848990723
  refined_manual_mean_px: 58.72466436582349
  baseline_query_distance_px: 3.7397923933135138
  refined_query_distance_px: 3.2288459870550366
  baseline_temporal_distance_px: 4.049048953586155
  refined_temporal_distance_px: 3.578174418873257
artifacts:
  run_dir: knowledge/runs/run-slcs-court-db-meiji-rgb-v1
  output_dir: outputs/tennis_scene/analyze/court_db_meiji_rgb/s42-001
  log: knowledge/runs/run-slcs-court-db-meiji-rgb-v1/run.log
parents: [run-slcs-meiji-v9-court-v1, run-slcs-meiji-baseline-line-audit-v2]
tags: [slcs, court, meiji, rgb, cpu, negative-result]
---

## 考察 / Findings

### 要約
固定レシピの9視点・45候補でECCは全件推定値を返した。しかし手動注釈のある3視点すべてで誤差が増え、平均7.31px→58.72pxとなった。RGB線への適合スコア改善は幾何精度改善の証拠にならず、この設定はv9の置換・teacher生成に採用しない。

### アーキテクチャ詳細
Chen/Little系のカメラ事前分布・合成線検索という構成を使うが、学習済みSiamese特徴の再現ではない。既存NumPy HOGとECCライブラリを直接使用。各視点96個のカメラを512×288で生成し、HOG上位5件を100反復上限のECCで精密化した。

v9 Hから推定した近似K/R/tの中心±0.3m、光軸の地面交点±0.5m、水平FOV±1.5°を事前分布とした。DBはゼロロール・画像中心主点・正方画素・歪みなしを仮定する。cam2は世界XYの180°回転を明示的に適用し、手動比較時にカメラ局所14点順へ戻した。refined HからK/R/tを再推定したとは主張しない。

クエリは実RGBの白色top-hat（15px窓、強度差20以上、V100以上・S100以下）と細長い連結成分で抽出した。v9 Hの投影凸包を80px拡張した広いROIに制限するため、v9に条件付けられた実験である。予測点を線で結んだ画像をクエリには使っていない。光・影・ネット・人の混入や線の欠落を除去しきれない。

各clipの約1/3時点を検索、約2/3時点を時間別診断に使い、キャッシュの検出用9サンプルを除外した。候補選択は同一クエリ上の双方向truncated Chamfer平均（上限12px）最小のみ。全9視点の選択を `selection_frozen.json` に保存した後、手動注釈を初めて読み込んだ。manualはprior・ROI・抽出・候補選択に不使用。全設定は1回の実験で固定、manual結果を見た調整は行っていない。

### メトリクスの解釈
手動比較はvideo_000/clip_000の3カメラ、各14点、計42点。1010フレームに同一注釈が繰り返されるため独立標本は3静的レイアウトであり、時間汎化や3D精度の評価ではない。注釈の作成者・時刻はファイルから確認できず、検出器サンプルを避けても静的v9 prior自体はキャッシュに依存する。

| camera | v9 px | retrieved initial px | refined px |
|---|---:|---:|---:|
| cam0 | 10.18 | 27.14 | 71.70 |
| cam1 | 6.94 | 13.12 | 88.55 |
| cam2 | 4.82 | 11.90 | 15.92 |

同一RGBの双方向距離平均は3.740→3.229、別時刻は4.049→3.578（512px幅画像単位）。9/9で改善したが、同じ抽出器の系統誤差を共有する診断であり独立GTではない。ECC成功は有限推定値の返却のみで、意味的対応や幾何正当性を示さない。学習を伴わないため収束曲線はない。`metrics.png` が3種類の評価を分離する。

### アーキテクチャ⇄メトリクスの因果考察
v9近似姿勢に対するcam0のロールは約−6.76〜−7.46°。同じ中心・光軸・FOVをゼロロールに置き換えた名目投影とv9の14点平均差は25.0〜27.7pxとなった（近似ピンホール化も含む差で、純粋なロール効果の分離実験ではない）。cam1は約−1.23〜−1.69°、差10.0〜10.5px。ゼロロール事前分布の表現制約が観測された。

仮説: 部分的白線、影、ネット等の混入とコートの対称・平行線構造により、自由度の高いHが誤った線へ適合し、ChamferとECCを改善しても対応点を悪化させた。cam2のロールは小さいが精密化は手動誤差を増やしており、ロールだけでは失敗を説明できない。親エージェントもcam0のRGB・mask・比較図を目視し、retrieved段階のロール差とECC後のドリフトを確認した。全候補H・ECC・距離・選択IDを残しており、成功例のみの抜粋はしていない。

### 既存実験との比較
v9-courtを基準とし、baseline-line-audit-v2の「実RGBと独立に比較する」要件を全コートに拡張した。ただしaudit-v2の人間が確認した単一baseline線と異なり、今回のRGB成分は未確認の候補群。今回の全体距離を同等の正解線精度として扱わない。

### 次に有効な実験
ロールを持つカメラ事前分布と、線の所属を検証できるRGB抽出・対応付けを別々に検証する。現状のECC成功率を採用条件にせず、幾何妥当性と独立線評価で棄却できる仕組みが必要。次の設定を現在の3手動レイアウトに合わせて調整する場合、この3レイアウトは開発用となり、別の独立注釈が必要。
