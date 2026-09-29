# 3D Ball Refiner (#936)

CPUでの入力分布・合成系列の生成を提供する。pipeline接続は未実装。
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

## 59.94fps合成系列の生成

設定の正本は [dataset_plan.yaml](dataset_plan.yaml)。
`synthetic/` はBLCSの240Hz原系列を正確な60000/1001Hzへ線形補間し、
合成3D → source画素 → `BallGMM2D` → `pixel_moments()` → 方式Aの順で生成する。
全64成分とcamera集合、全共分散を保存し、点推定への置換はしない。
数値は仮定した劣化で、#935の実出力に較正した値ではない。

`timebase.py` は実際に残ったshot区間のイベントだけを採用し、native frame/秒と
最近傍frameを保持する。打球・bounceの±5frameとnet通過付近を物理lossから除外する。
fence時刻を返さないsimulatorに対しては、fence近傍も保守的に除外する。
`observations.py` は#934の録画splitを維持し、各sceneの摂動cameraを真/推定の双方に
使うclean geometry条件を作る。真/推定行列は別fieldで保存する。
校正誤差条件は未実装。実ボール座標や注釈は入力にしない。

遮蔽はpresenceを下げず、分布を広げて代替位置の重みを変える。
画面外は別maskと低presenceで表す。GMM headの範囲に合わせる明示的な画像境界clipを
適用し、その成分数をmetadataへ保存する。短すぎるrally、solver失敗、非SPDは停止する。
seedの引き直し、成分削除、jitter、自動resumeはしない。

生成CLIは `python -m src.tasks.ball_refiner.scripts.generate_synthetic_3d`。
`--project-root`（作業checkout）、`--data-root`（共有data）、
`--plan`（上記YAML）、`--output`（新しいDATA内directory）を絶対pathで指定し、
`--mode smoke` または `--mode pilot` を明示する。
OMP/MKL/OPENBLASのthread数を1にして起動する。process数はYAMLの値で最大4。

各rallyを可変長のNPZ+JSONで保存し、進捗manifestはatomicに更新する。
timestamp/cameraはfloat64、軌道/GMMはfloat32、maskはbool。
float32 export後もSPDと有限性を検査する。未完了/失敗は`complete`にしない。
入力設定・校正・生成codeのSHA、全イベント、実frame/bytes、simulation/triangulation時間、
process RSS、全量生成の線形予測をmanifestへ記録する。出力は上書きしない。
RGB生成、実Meiji評価、pipeline統合はこの入口の範囲外。

## 次のGPU実験

run 2はCPU smokeとforward/backward成功後の100-update memory smokeだけが承認済み。
本学習・較正・精度比較は別runで申請する。
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
