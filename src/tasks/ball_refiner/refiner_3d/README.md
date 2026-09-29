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
`--fixture` と未使用の `--output` を明示し、CPU/native threadを1に制限する。

## 59.94fpsデータ生成計画

実装前の計画であり、生成済みdatasetではない。

- BLCSの `RallySimulator` をCPUで使い、`sim_fps=output_fps=240`、
  `physics.dt=1/240` として元のsimulation時系列とイベントindexを得る。
  現行実装は整数strideなので `output_fps=59.94` へ直接変えてはいけない。
  出力時刻を正確に `n*1001/60000` 秒として線形補間し、イベントは元の秒を
  保存して最近傍frameへ対応させる。打球/バウンスをまたぐ速度差分は物理lossから除外する。
- cameraは既存のcourt校正からK/R/tとsource画像サイズだけを取り出す。
  実2D/3Dボール座標は読まない。clip/camera、校正のSHA256、近似校正で歪み補正が
  ないことをmanifestに残す。旧geometry replayのsplitを流用しない。
  #934/#935に合わせ、video_002=train、video_000=val、video_001=test。
- scene単位でcamera中心の標準偏差0.15m、軸角0.5度、焦点距離1%、主点2pxの
  摂動を固定する。最初は真の投影と三角測量に同じ摂動cameraを使用。
  校正誤差を加える別条件では真/推定cameraを分けて保存する。
- 合成3D→source pixel投影→2D refiner相当の `BallGMM2D` →
  `pixel_moments()` →確率的三角測量の順を必須とする。
  観測/短欠損/全camera長欠損（1,4,8,16,32,64frame）、
  代替位置仮説、相関を持つ共分散、時間相関した誤差を作る。
  遮蔽だけでpresenceを下げない。画面外と遮蔽を別maskで保存する。
  GMMの較正パラメータは#935 train/valから後で固定し、仮定した劣化を実測とは呼ばない。
- 保存物は秒、3D教師、イベント秒/frame、欠損mask、camera/sourceサイズ、
  全2D GMM＋presence、全3D GMM、生成seed/設定/hash。
  camera集合の不在項も落とさず保存する。動画・RGBは生成しない。
- 最初のCPU smokeは各split 4 rally。次にtrain/val/test=512/64/64 rally、
  最大512frame、K=3、3cameraを候補とする。全64個の3D成分をfloat32で保存すると
  上限約1.1GB（2D・教師・metadata込みは1.5GBを計画上限）で、実測を報告する。
  sceneごとのseed分割、変長padding mask、重複windowのsplit禁止を検証する。

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
