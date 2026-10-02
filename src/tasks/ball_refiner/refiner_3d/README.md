# 3D Ball Refiner (#936)

CPUでの入力分布の準備段階。モデル・pipeline接続は未実装。
要件の正本は [#936](https://github.com/Motoki0705/tennis-lab/issues/936)。
2D契約は [親README](../README.md#2dモデルのapi) を参照する。

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

## 59.94fpsデータ生成計画

実装前の計画であり、生成済みdatasetではない。
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

## 次のGPU実験（未承認・未投入）

上のCPU smokeと#935の契約/較正確定後にqueueへ申請する。
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
