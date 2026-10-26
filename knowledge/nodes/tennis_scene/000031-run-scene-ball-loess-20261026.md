---
id: run-scene-ball-loess-20261026
type: run
task: tennis_scene
sequence: 31
recorded_at: '2026-10-26'
title: LOESSと外れ値除去による保存観測のCPU再三角測量と公開デモ更新
provider: codex
status: done
config:
  consensus_seconds: 0.7
  enabled: true
  loess_seconds: 0.35
  max_gap_seconds: 0.2
  minimum_residual_diagonal: 0.012
  residual_mad_multiplier: 4.0
  seed: 42
  trials: 80
  scene_schema_version: 2
  ball_points_version: 4
  source_fps: 59.94006
  source_frames: 1010
  frozen_cameras_and_people: true
  blender_ball_max_gap_frames: 0
metrics:
  valid_before: 821
  valid_after: 859
  lost_valid: 19
  gained_valid: 57
  median_reprojection_px: 2.8950562477111816
  p95_reprojection_px: 9.38376522064209
  before_median_reprojection_px: 3.519322395324707
  before_p95_reprojection_px: 10.61745548248291
  before_p95_adjacent_step_m: 0.6876861453056335
  after_p95_adjacent_step_m: 0.4123325049877167
  before_max_adjacent_step_m: 1.0741184949874878
  after_max_adjacent_step_m: 1.0766195058822632
  valid_3d_with_fewer_than_two_real_views: 0
  common_valid_frames: 802
  raw_reprojection_before_median_px: 3.554317890100016
  raw_reprojection_after_median_px: 4.810261841053709
  raw_reprojection_before_p95_px: 11.294080775597966
  raw_reprojection_after_p95_px: 12.697105462521348
  common_adjacent_step_before_p95_m: 0.6821850538253784
  common_adjacent_step_after_p95_m: 0.4049496054649353
  rejected_2d_observations: 108
  deterministic_replay: true
  pipeline_gallery_unit_tests_passed: 138
artifacts:
  run_dir: knowledge/runs/run-scene-ball-loess-20261026
  output_dir: outputs/scene-ball-loess
  replay: knowledge/runs/run-scene-ball-loess-20261026/replay.json
  comparison: knowledge/runs/run-scene-ball-loess-20261026/comparison.json
  comparison_plot: knowledge/runs/run-scene-ball-loess-20261026/comparison.png
  multiview_validation: knowledge/runs/run-scene-ball-loess-20261026/multiview-validation.json
  public_demo_audit: knowledge/runs/run-scene-ball-loess-20261026/public-demo-audit.json
  reproduction: knowledge/runs/run-scene-ball-loess-20261026/repro.sh
  comparison_video: outputs/scene-ball-loess/comparison/comparison.mp4
  extreme_frame_review: knowledge/runs/run-scene-ball-loess-20261026/extremes.json
  extreme_frame_plot: knowledge/runs/run-scene-ball-loess-20261026/extremes.png
  blender_validation: knowledge/runs/run-scene-ball-loess-20261026/blender-validation.json
  media_validation: knowledge/runs/run-scene-ball-loess-20261026/media-validation.json
parents:
- run-multiview-sample-standard-pipeline-20261021
- run-multiview-readme-gif-20261021
- run-blender-reveal-meiji-cam1-20261016
relations: []
papers: []
tags:
- temporal_smoothing
- loess
- cpu_replay
- public_demo
- integration_validation
date: '2026-10-26'
---

## 観測と比較条件

公開サンプルMeiji `video_000/clip_000` のREADME用sceneと、削除済みworktreeから保全されていたcomponent storeを発見し、保存済みのproduction検出観測に外れ値除去・LOESSを適用して球を再三角測量した。元storeの採用descriptor依存鎖、scene・sidecar・使用配列のchecksum、元動画3本の明示的なpath置換とchecksumを検査した。元のstore、export、動画は変更していない。

[v5プレゼンテーション](https://github.com/Motoki0705/tennis-lab/releases/tag/tennis-scene-presentation-v5)で用いた局所2次運動consensus・頑健な2次LOESSの方式とパラメータを共通utilityへ移した。プレゼンテーションのDPT checkpointとは異なる、README元runのproduction detector出力をそのまま再利用する。短い支持不足・長い欠損・解像度/FPSの違いを明示的に扱い、短い連続的な方向転換の再検査を保持した。手作業で除外frame区間を指定していない。

現行pipelineは`ball_points` v4をside推定・camera alignment・球三角測量へ渡す。本比較はこれら全段の再実行ではなく、**元のカメラ・side・人物復元を固定した球だけのreplay**である。通常runnerはsrcのcode fingerprint変更で全componentを無効化するため、旧artifactを新identityへ付け替えず、実験用entrypointと別出力を作った。`replay.json`とscene metadataへ元index・artifact参照・code指紋・再実行/固定範囲を記録した。人物・コートの20配列は元sceneと完全一致した。

## 定量的な結果と制約

| 指標 | 元の球三角測量 | 外れ値除去＋LOESS後 |
|---|---:|---:|
| 有効3D frame / 1010 | 821 | 859 |
| 各結果の採用inlier再投影中央値 (px) | 3.519 | 2.895 |
| 各結果の採用inlier再投影p95 (px) | 10.617 | 9.384 |
| 有効な隣接3D移動量p95 (m) | 0.688 | 0.412 |
| 有効な隣接3D移動量の最大 (m) | 1.074 | 1.077 |
| 共通802 frame・元2D観測への再投影中央値 (px) | 3.554 | 4.810 |
| 同・p95 (px) | 11.294 | 12.697 |

cam0/1/2で元の764/945/714実観測から31/35/42を除外し、733/910/672を採用した。診断用の短い補間は136/75/134 frameだが、下流へ渡す`observed`・confidence・uvへは混ぜていない。実観測2view未満で3Dが有効になったframeは0。元の有効frameのうち19が無効化され、57が新たに有効になった。同一入力・同一codeで独立に2回実行し、sceneの球配列・component配列/出所・replay manifestは完全一致した。

![元の結果とLOESS後の球の比較](../../runs/run-scene-ball-loess-20261026/comparison.png)

採用inlierの自己整合指標と隣接移動量の大部分は改善したが、最大の跳びは改善しない。元の未平滑観測への追従残差は増えた。比較対象のinlier集合も変わるため、自己整合誤差の減少を独立な位置精度の向上とは解釈しない。共通frameの隣接移動量p95でも0.682→0.405 mへ減ったが、滑らかさの指標である。人手3D基準やbounce/hit時刻GTはなく、カメラ校正やdetectorの精度は再評価していない。

最大移動は元sceneの570→571から、新sceneの969→970へ移った。後者は元sceneでは無効だった区間で、3viewから2viewへ変わる時点に1.077 mの跳びが残る。両viewの2D平滑化座標に大きな跳びは見えず、カメラ集合の変化と固定校正の不整合が関係する可能性があるが、原因は未確定である。lost_validの57〜59はバウンド付近で、2D実観測を保持しているにもかかわらず3D速度gateが3frameを欠損にする。過平滑化や方向転換付近の復元悪化がないとは主張しない。762/764はcam0の離れた検出候補を除外し、実観測が1viewだけになった。これら3区間を3カメラの実写cropと[局所比較図](../../runs/run-scene-ball-loess-20261026/extremes.png)で確認し、全19frameの理由を[extremes.json](../../runs/run-scene-ball-loess-20261026/extremes.json)へ残した。手動の区間除外や、このclipに合わせた3D速度閾値の変更は行っていない。

30/60/120fpsの速い動き・バウンドと単発外れ値の合成試験では、真の急な動きを保持し、外れ値だけを除外した。欠損の内側・長い欠損・両端・疎な観測、解像度倍率、noise低減と位相、storage round-tripとgeometry maskの回帰を確認した。補間/除外された点だけでは2つ目のviewを供給できないことを実際の三角測量まで検査した。pipeline・gallery・共通utilityの138テスト、変更Pythonのruff/mypyが通った。validator評価は指定されていないため0回。

## 公開デモと表示の確認

READMEの3言語が参照する`multiview_reconstruction.gif`を新sceneからCPUで再生成した。960×620、168frame、16.75秒。元frame 6〜1008を6frameごとに描画し、欠測を跨いだ軌跡線を作らない。入力パネルの全504比較でラベル部分を除くRGB最大差0、GIFと元PNGの最小PSNRは36.31 dB。代表6frameと2D比較動画から抽出した4frame、全区間の軌道比較図で、外れ値除外、球の欠損、バウンド付近の形状を目視確認した。この確認を全frameの真の3D位置の保証へ広げない。

同じ新sceneの3Dパネルから、直接公開されていた旧`rich_scene_rendering.gif`も置き換えた。Blender revealも同じsceneで再生成し、球の表示補間を0にして3D欠損を保つ。CPU llvmpipeを明示し、adapter probeでも確認した。Blenderの全1010frameの球位置・表示mask、9frameのmesh頂点で差0、カメラ投影の最大差0.000131 px。元frame 96と描画plateのPSNRは43.13 dBだった。169枚の960×540 PNGから16.917秒のMP4を作り、公開GIFは400×225・135frame・16.88秒。切り替え・自由視点を含む代表6frameを目視確認し、3つのGIFとMP4を全編デコードした。詳細は[Blender検証](../../runs/run-scene-ball-loess-20261026/blender-validation.json)と[media検証](../../runs/run-scene-ball-loess-20261026/media-validation.json)を参照。

2026-10-26時点の[公開先監査](../../runs/run-scene-ball-loess-20261026/public-demo-audit.json)では、GitHub Pagesはなく、公開HFモデル/Datasetにも描画済みGIF・動画demoはなかった。Datasetの3本のMP4は元入力。該当著者のHF Spacesは空だった。身体配置だけのGIFと元入力の`tennis_clip.gif`は球の再構成demoではない。過去の検出・追跡release、v5プレゼンrelease、knowledgeの過去runは固定した履歴であり、書き換えない。任意の第三者mirrorや過去commit URLの全網羅はしていない。

## 再現と次の確認

[run.json](../../runs/run-scene-ball-loess-20261026/run.json)に実行環境・path対応、[replay.json](../../runs/run-scene-ball-loess-20261026/replay.json)に元入力と使用artifact、`uncommitted.patch`に元commitからのcode差分を保存する。予測NPZ/NPY、元動画、全PNG、MP4、Blender bundleは`outputs/scene-ball-loess/`と保全元に残し、knowledgeへ大容量成果物を複製しない。学習はなくTensorBoard曲線はない。

```bash
bash knowledge/runs/run-scene-ball-loess-20261026/repro.sh \
  SOURCE_STORE VIDEOS_DIR CHECKPOINT_ROOT NEW_OUTPUT_DIR BLENDER_BIN
```

`VIDEOS_DIR`は`cam0.mp4`〜`cam2.mp4`を持つ明示的なrootで、元sourceのchecksum一致を要求する。checkpoint rootには既存のSMPL faces/regressor、BlenderにはCPUのllvmpipeを使う。出力先は新規で、元storeと重ならない場所を指定する。

本runは1clip・固定カメラ条件であり、既存のモデル/checkpoint採用判断を変更しない。次は独立したball 2D/3Dとbounce/hit時刻GTで、平滑化のbias、方向転換の保持、検出外れ値除去のprecision/recallを測る。別clipと欠損の多い条件も分けて評価する。
