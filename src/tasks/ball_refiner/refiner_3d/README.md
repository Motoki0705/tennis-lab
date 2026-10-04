# 3D Ball Refiner (#936)

mainの計算実装はCPUの入力分布比較までで、モデル・pipeline接続は未統合。
保存済みの実験datasetは下記のレビュー入口で読み取れる。
要件の正本は [#936](https://github.com/Motoki0705/tennis-lab/issues/936)。
2D契約は [親README](../README.md#2dモデルのapi) を参照する。

## 保存datasetのレビュー

各cameraのボール位置候補、3Dの合成真値とのずれ、欠損時の分布を同時刻で見る。
datasetの系列・ローカル棚卸し・実画面は [Dataset Review #990](https://github.com/Motoki0705/tennis-lab/issues/990) が正本。
開発中 [PR #969](https://github.com/Motoki0705/tennis-lab/pull/969) の保存schema
`ball_refiner_3d.synthetic.v1/v2` をtask内の `review/` で読む。
生成器・モデル・GPU推論は必要ない。mainの古い計画configを実体の代わりに使わず、
各datasetの `manifest.json` に保存された生成設定・登録rally・NPZ checksumを検証する。

```bash
# このworktreeをcwdにする。data-rootはdataset directory群の親を絶対pathで明示。
/home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.tasks.ball_refiner.scripts.review_3d_dataset \
  --data-root /home/kamimura/projects/tennis-lab/data/ball_refiner --port 8893
```

ブラウザで `http://127.0.0.1:8893/` を開き、dataset → split → rallyを選ぶ。
frame欄・slider・timelineを使い、全camera gap、画面外、hit/bounceへ移動できる。
URLにdataset/rally/split/frameと描画filterが残る。
3D画面はdragで回転、wheelで拡大。「分布周辺を拡大」は2D候補と真値のsource pxを拡大する。

- 緑はsimulatorの保存3D教師とそのtrue cameraへの投影。実測された球のGTではない。
  camera校正はMeiji由来。主レビュー系列の2D劣化には#935の保存推定と2D教師から作られた
  残差bankを使っており、そのbankは較正fit frameを再利用する。独立した実データ評価ではない。
  RGBを合成・代用しない。
- 青はcameraが寄与する保存分布、紫は空camera subsetのprior成分。
  各成分の2σは混合全体のHDRではない。描画filterで省略した成分数・確率質量を表示し、
  成分tableとAPIには全成分を保持する。重みを再正規化しない。
- 遮蔽/gapと画面外は別mask。遮蔽中も保存2D GMMが残り、amodal存在やprior-only massとは別である。
  物理球の不存在/unknownを示す独立した教師fieldはこのschemaにはない。
- 固定予算の積分は「収束未評価」、旧診断の未収束と診断未保存は別表示する。
  `complete` は生成完了であり、datasetの品質承認や最終holdout合格ではない。
- failed runは一覧だけ、stopped runは登録済みrallyのみ読める。未登録NPZを採用しない。
  手元の実体が無い/破損/入力変更の場合は明示的に停止し、fixtureや生成で補完しない。

`review/data.py` が保存契約と表示payload、`review/web.py` が読み取りAPIとtask専用static UIを担当する。
教師の再生成・三角測量の再計算・実験bankの再読込は行わない。

## 今回の比較

A（成分組合せのMahalanobis最適化＋Laplace）、B（適応voxel積分）、
C（2D分布の標本化＋三角測量）を同じ合成入力で比較する。
3D密度の単位は m⁻³、NLLは自然対数、coverageは混合全体のHDRで測る。
点推定の平均誤差だけで方式を選ばない。raw detectorは入力にしない。

位置prior・presence周辺化・Laplace evidenceの定義と限界は
[共有geometry API](../../../utils/geometry/probabilistic_triangulation/README.md)を正本とする。
`triangulation.frame_observations()` は既存の `pixel_moments()` を使って、
B軸を呼び出し側が同期させたcamera順のV軸へ写す。成分は平均しない。

比較用cameraは
[`fixtures/meiji_video_002_clip_010.json`](fixtures/meiji_video_002_clip_010.json) の
video_002/clip_010、cam0/1/2（1920×1080）。
既存court校正artifactのパスとSHA256を含み、実ボールの観測・評価には使っていない。
比較のCLIは `python -m src.tasks.ball_refiner.scripts.compare_triangulation`。
`--fixture` と未使用の `--output` を絶対pathで明示し、CPU/native threadを1に制限する。

## 当初の59.94fpsデータ生成計画

以下はmainへ入った時点の実装前計画であり、保存datasetの現況ではない。
数値・split・校正SHA・劣化・保存fieldの正本は
[dataset_plan.yaml](dataset_plan.yaml)。これは計画用configで、まだ生成CLIの入力ではない。

- BLCSの `RallySimulator` から240Hz原系列とイベントindexを得て、
  出力を正確な60000/1001Hz時刻へ線形補間する。現行実装の整数strideへ
  小数fpsを直接渡さない。イベント秒を保持して最近傍frameへ対応させ、
  打球/バウンスをまたぐ速度差分を物理lossから除外する。
- cameraはcourt校正のK/R/tとsource画像サイズだけを使用する。実2D/3D球座標は
  読まない。旧geometry replayのsplitを流用せず、#934/#935の録画splitを維持する。
  scene内で同じ摂動cameraを真投影と推定に使うclean条件を先に確認し、
  校正誤差の条件は真/推定cameraを分けて保存する。
- 合成3D→source pixel投影→refiner相当の `BallGMM2D` →
  `pixel_moments()` →確率的三角測量の順を必須とする。
  短欠損・全camera長欠損・持続する代替位置・相関共分散・時間相関誤差を含める。
  遮蔽だけでpresenceを下げず、画面外と遮蔽を別maskにする。
  劣化パラメータは#935 train/valから後で固定し、仮定した劣化を実測とは呼ばない。
- CPU smoke後にpilotを生成する。全camera集合の不在項と全3D成分を保存する。
  動画/RGBは生成しない。scene単位のseed/split、変長padding、重複窓のsplit禁止、
  イベント補間境界、実際のframe数とbytesを検証する。
  今回はcamera fixtureだけを生成し、rally/datasetの生成は未実施。

## 当初のGPU実験案

以下の予算は当初の案。実行済み実験の現況は上記Issue/開発PRと保存manifestで確認する。
このレビュー入口から学習・生成を投入することはない。
空間位置そのものをx0予測するflow matchingモデルと、同じbackboneの
1-step回帰対照を作る。サンプル間の分散をuncertaintyとして出す。
損失・評価・禁止事項はissueの要件をそのまま受入条件とする。

| 段階 | データ/計算予算案 | 時間・VRAM・出力の見積 |
|---|---|---|
| memory smoke | 12 rally、T=128、batch=2、幅128/4層、100 update、fp32 | 10分上限、4–6GB、0.2GB |
| diffusion pilot | train512/val64、T=128、batch=8、幅128/4層、10k update、bf16、1 seed | 2時間上限、6–10GB、1GB |
| 同backbone回帰 | 同一split/seed/batch、10k update | 2時間上限、6–10GB、1GB |
| 合成評価 | test64、16 samples×16 ODE steps、3方式平滑化対照 | 30分上限、6–10GB、0.5GB |

いずれも実測前の概算。12GBを超える構成は実行せず、smoke実測後にbatch/窓長を確定。
実MeijiのLOCO評価・pipeline統合は、学習済み#935と別runの許可を待つ。
