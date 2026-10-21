---
id: run-multiview-readme-gif-20261021
type: run
task: tennis_scene
sequence: 29
recorded_at: '2026-10-21'
title: 公開3カメラ入力と標準3D再構成のREADME用GIF
provider: codex
status: done
issue: 1275
date: '2026-10-21'
config:
  clip: video_000/clip_000
  device: cpu
  start_frame: 6
  stop_frame: 1010
  frame_step: 6
  player_representation: smpl
  palette_max_colors: 192
metrics:
  frames: 168
  width: 960
  height: 620
repro:
  commit: 7c504fe6891ab85fb965dd09efb913ca960db983
  branch: codex/readme-gif
artifacts:
  run_dir: knowledge/runs/run-multiview-readme-gif-20261021
  execution: knowledge/runs/run-multiview-readme-gif-20261021/run.json
  input_provenance: knowledge/runs/run-multiview-readme-gif-20261021/input_provenance.json
  validation: knowledge/runs/run-multiview-readme-gif-20261021/validation.json
  contact_sheet: knowledge/runs/run-multiview-readme-gif-20261021/contact_sheet.png
  gif: assets/tennis_scene/multiview_reconstruction.gif
  output_dir: outputs/readme_gif/release
parents:
- run-multiview-sample-standard-pipeline-20261021
relations: []
papers: []
tags: [rendering, public_sample, cpu]
---

## 成果物と入力

[README用GIF](../../../assets/tennis_scene/multiview_reconstruction.gif)は、公開したMeiji
`video_000/clip_000` の同期済み3動画と、[標準pipelineの保存済みSceneResult v2](000028-run-multiview-sample-standard-pipeline-20261021.md)
からCPUだけで生成した。推論・学習・GPU描画は再実行していない。
左にcam0〜2（各320×180）、右にコート・人物・球の3D表示を並べ、同じ元frame番号を使った。
cam0の拡大と左右のletterboxは入力のまま残している。

先頭の描画sample（frame 0）は人物と球の有効maskが立っていないため、次のsampleであるframe 6から開始した。
元frame 6〜1008を6frameごとに描画した168frameで、表示区間は元動画の約0.10〜16.85秒。
入力の60000/1001fpsを6で割った10000/1001fpsを使い、最後のframeだけ3 centisecondsにした。
GIFの最終的なサイズ・時間・SHA-256・全frameの検査結果は`validation.json`を正本とする。

描画コードと生成scriptの版はcommit `7c504fe6891ab85fb965dd09efb913ca960db983`。
scene・metadata・3動画・SMPL faces資産・関節regressorの実パス、サイズ、SHA-256は
`input_provenance.json`に記録した。3動画のhashは先行runの公開サンプルの記録と一致する。
使用した既存資産は`TennisSceneRenderer`、`smpl_vertices_to_court`、共有camera/court/mesh renderer。
Blender revealの別入力・旧推論結果は混ぜていない。

## 再生成

checkoutのrootから、sceneと入力・正規取得済みのbody model資産を読み取り専用で指定する。
新しい出力先が必要で、生成PNGや大きな中間動画はGitへ保存しない。

```bash
bash knowledge/runs/run-multiview-readme-gif-20261021/repro.sh \
  /path/to/scene.npz /path/to/samples /path/to/ckpt "$PWD/outputs/readme_gif_reproduction"
```

`repro.sh`はCPU描画、FFmpegでのpalette生成とGIF化、完成GIFのdecode・入力照合・確認画像抽出を行う。
今回実行したコマンド、CPU thread制限、環境情報は`run.json`に記録した。
別の空の出力先で`repro.sh`全体を再実行し、再生成GIFのSHA-256が同梱GIFと一致することも確認した。
素材はMITの対象外で、利用条件と帰属表示は[LICENSES.md](../../../LICENSES.md#サンプルdatasetの利用条件)に集約した。

## 確認したことと限界

完成GIFを全frame decodeし、元動画の縮小画像と合成PNGのラベル外領域を、168frame×3視点の
504回比較した。RGB最大差は0。GIF量子化後のPNGとのPSNR・再生時間・ループ設定も検査した。
これは元frameとの対応の検査であり、元の撮影時の同期精度を再計測したものではない。

完成GIFから元frame 6・180・360・540・720・1008の画像を抜き出し、
[6枚のcontact sheet](../../runs/run-multiview-readme-gif-20261021/contact_sheet.png)と
原寸画像（`outputs/readme_gif/release-review/frame_*_source_*.png`）を目視確認した。
3カメラで同じラリーの動きが追え、3Dでは白いコートとネット、橙/青緑の2人、黄緑の球・軌跡を識別できた。
frame 180では近側選手の構え、frame 720では腕を上げる動きが入力側で読み取れる。
frame 720の近側選手は`player_smpl_valid=false`のため、既存rendererの仕様どおりCOCO17骨格表示になっている。
無効な球の区間やmesh欠測を補間せず、軌跡の揺れも保存sceneのまま残した。

ニューラル推論・カメラ校正・身体配置・球軌道の精度向上は評価していない。
GIFの技術的表示と生成経路を確認したもので、全frameの3D精度や身体姿勢の正しさを保証するものではない。
学習ではないためTensorBoard曲線はない。既存モデルの採用判断は変えず、
クリーン環境での導入は#1274、トップREADMEへの掲載とGitHub上での表示確認は#1276へ引き継ぐ。
