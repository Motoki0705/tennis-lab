# tennis_scene責務整理・実クリップ検証

責務整理と指定clipの実行・出力評価を完了した。モデル出力には検出欠損・3D軌道の異常が残る。生成処理の完了と再構成精度を分けて以下に記録する。

作業先: /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup。branch: codex/tennis-scene-responsibility-cleanup。base: fa8798a4c19f7efc6b123f4819130d57fe14b565。変更は専用worktree内にあり、メインrepoのコードは変更していない。

## 実装

- dataset_pipeline / reference_pipelineと旧動画解析・教師補正・専用品質レビュー経路を削除した。互換wrapperは残していない。
- tennis_scene/scriptsには指定5入口と__init__.pyだけを残した。ヘッドレス書き出し、旧layout移行、BLCS専用準備、専用設定・補助処理・テスト・CLI登録を整理し、GUI共通書き出しは保持した。
- scene形式・保存先・残すCLIの引数名を維持し、generate_datasetの動画rootをDATAへ修正。保存済み選手対応はcamera順とreference cameraを検証する。
- [5入口の入出力・必須ファイル・再実行規則の正本](/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/src/tennis_scene/README.md)へDATA/ARTIFACT/OUTPUT、ログ・中間結果、DATA内sceneの可視化と先頭camera描画の制約を集約した。
- [SLCS準備](/home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-responsibility-cleanup/src/tasks/slcs/generate_dataset/README.md)を追加。python -m src.tasks.slcs.scripts.generate_datasetで、完成済みの構造化datasetを検証し、同梱動画からDINO特徴とvideo単位splitを生成する。dataset指定必須、70/15/15・seed 0、既存配列・可視性・任意のlabel_qualityを使う。sceneは変更せず、新規教師補正・品質選別は追加しない。
- DINO既存出力はmanifest・camera・特徴仕様・保存配列まで検証し、splitも設定・収録単位・完全な割当を検証する。不整合や不完全sceneは明示エラーとし、再生成は既存overwrite設定で指定する。
- 実データからCourtの画像由来ROI探索、カメラ後方をまたぐCourt範囲の画像内clip、ViTPose visibilityの範囲監査、Ball checkpointの推論専用復元と保存RGB正規化、分数FPSの保持を修正した。

## 実行条件・承認された引き継ぎ

対象はvideo_000/clip_000、cam0,cam1,cam2、1920×1080、1010frame、59.94006fps（約16.850秒）。referenceはcam0、half-turnsは[false,false,true]。

- Courtは両入口で、このタスク内に全3030frameを独立推論した。結果SHAは同一だった。ユーザーの「コート推論は引き継ぎしてください。」に従い、その完了出力を修正後の実行へ引き継いだ。
- run_pipelineの人物推定は、このタスク内に再生成・映像照合済みのDINO/ViTPose/HMR2/GVHMRを、追加承認「今回生成した人物推定を引き継ぐ（推奨）」に基づいて引き継いだ。generate_datasetの人物推定は修正版で独立に実行した。
- 両入口のPLCS、Ball、BLCS、motion alignment、scene保存と全可視化は修正版で再実行した。元から存在した旧scene・旧自動予測を生成入力にはしていない。
- 既存の人手対応は、新旧bbox・2D姿勢の全1010frameと、通常時点・最小対応margin時点の元映像/cropを照合してから適用した。両入口とも[[0,0,1],[1,1,0]]、逆順が優位なframeは0。track番号だけで同一性を判断していない。
- 手動Courtと外注ballのobserved注釈は評価だけに使用した。Court領域の提案・採択は画像とモデル出力だけで行う。
- 全GPU処理は元repoの共有training queue、resource=all、FIFOで実行した。他タスクは中断・並び替えしていない。修正のため停止したのはこのタスクのジョブだけである。

[Court由来とSHA](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/court_reuse.json) / [人物結果の由来とSHA](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/human_reuse.json) / [元annotationの退避とSHA](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/annotation_backup_verified.json) / [確定追加指示を含む原要件](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/validation/requirements.md)

## Root・重み・実行記録

DATA=/home/kamimura/projects/tennis-lab/data、CHECKPOINT=/home/kamimura/projects/tennis-lab/ckpt、ARTIFACT=OUTPUT=/home/kamimura/projects/tennis-lab/outputs、EXTERNAL_ASSET=/home/kamimura/projects/tennis-lab/third_party。project_rootは上記worktree。

Courtは既存のmultiscale_depth3/b863df1f01f0.ckpt（SHA b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383）を使用し、他は現行checkpointを使用した。
確認済みDINO拡張: /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T170847Z/dino_extension/lib/MultiScaleDeformableAttention.so。正規build処理による別artifactで、CUDA forward/backwardと実32frameのGVHMR chainを確認した。旧buildはdeprecated Tensor.type()により実forwardで失敗したため使用していない。

[全35入力の識別情報](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/input_receipt.json) / [環境・アセット情報](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/environment.json) / [生成コマンド](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/commands.json) / [描画コマンド](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/render_commands.json) / [pipeline設定](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline.expanded_pipeline.yaml) / [dataset設定](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset.expanded_pipeline.yaml)

人手部分はchecked_entrypoint.pyで照合結果の確定を待ち、既存対応を渡した。実際の対象CLI moduleをrunpyで__main__として実行している。

## 4入口の結果

| 入口 | 結果 | 主な成果物 |
|---|---|---|
| run_pipeline | 終了0 | [scene.npz](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/responsibility_cleanup/20260923T020139Z-pipeline/scene.npz) / [metadata](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/generate/responsibility_cleanup/20260923T020139Z-pipeline/scene.metadata.json) |
| generate_dataset | 終了0、overwrite=true | [dataset scene](/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene/scene.npz) / [完成マーカー](/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene/annotation.json) |
| visualization | 両sceneとも終了0 | [pipeline](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualization/scene.mp4) / [dataset](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualization/scene.mp4) |
| visualize_tasks | 両sceneとも終了0 | [pipeline](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/gvhmr_alignment_viz.mp4) / [dataset](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/gvhmr_alignment_viz.mp4) |

visualizationは各sceneの3D動画、visualize_tasksはball_detection/court_kp/gvhmr/plcs/gvhmr_alignment/blcsの全6動画を生成した。計14本を全frameデコードし、すべて1010frame、約59.94fps、約16.850秒、blank判定0だった。FPS差の最大は0.000060fps。冒頭・中盤・終盤と位置/速度/高さの極端値frameを画像で確認した。

両sceneの22配列はすべて有限。camera・player・時間軸、metadata、dataset完成マーカーのclip manifest SHAと配列shape/dtypeが整合した。既存SLCS readerもdataset sceneを受理し、player有効frameは各1010、ballは978だった。この有効判定はラベル精度の保証ではない。

## 数値・視覚評価

Courtはcamera-local順のstage結果を同じ順の手動点へ比較した。ballは外注注釈のobserved frameだけを母数とし、距離はそのうち検出があるframeに限定する。欠損率を別に示す。両生成入口の値はほぼ同じため以下はpipeline値。

| camera | Court平均 / p95 (px) | ball検出 / observed | ball欠損率 | ball中央値 / p95 (px) |
|---|---:|---:|---:|---:|
| cam0 | 9.93 / 18.53 | 315 / 841 | 62.54% | 7.64 / 475.88 |
| cam1 | 13.13 / 28.20 | 651 / 985 | 33.91% | 4.30 / 311.59 |
| cam2 | 8.37 / 12.39 | 621 / 941 | 34.01% | 6.62 / 336.70 |

Courtは全cameraの全frameで14点が成立した。手動点は静的で、反復frameを独立標本の精度とは扱わない。ballは中央値が小さくても大きな誤検出と欠損が残る。

- 正規化欠落の診断baselineに対し、ball欠損率は84.1/35.1/44.8%から62.5/33.9/34.0%へ減った。GT基準と検出集合が異なるため、中央値やp95だけで全面的改善とは扱わない。
- 3D ballは高さ-0.750～4.452m、z<0が50frame、最大frame間速度412.18m/s。旧baselineのz<0は301frameだったが、最大速度は310.12m/sから増えた。3Dの不自然な跳びは残る。
- f807→808の3D ball jumpと同時にcam2の2D ballが約337px跳び、元画像でも対象位置から別の場所へ移る。両frameは同じ2つの128frame窓内で、blend weightは連続しており、窓結合・軸・FPS・二重denormalizeの実装不整合を示す証拠は見つからなかった。未blendの各窓予測は保存していないため寄与の分離は未確認。
- PLCS最大速度は36.56m/s（player1、f778→779）。cam0で可視関節が4/17から17/17へ変わり、他viewにも2D姿勢変動がある。遠方選手の姿勢欠損・不安定さが残る。
- GVHMR整列ソルバは両選手で成功したが、PLCSとの位置RMSEは約1.69/2.22m。図でも配置とheadingの差が見える。整列後root高さには終盤の変動があり、ソルバ成功を精度の保証とはしない。
- 3D実測GTはない。以上は2D参照との整合、予測値の不連続と視覚的妥当性の評価であり、3D精度の測定ではない。生成sceneを高品質な教師GTとみなすことはできない。

[全指標・配列比較・動画decode記録](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/evaluation.json) / [2D参照の推移](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/figures/observations.png) / [3D推移](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/figures/dataset-trajectories.png) / [速度](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/figures/dataset-speeds.png) / [Ball jumpの原映像](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/figures/pipeline-ball-max-step-source.jpg) / [姿勢mask変化のcrop](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/figures/pipeline-player-max-step-crops.jpg)

## 両入口の比較

Court配列、frame数・FPS・画素寸法、ball visibility等は完全一致した。独立の人物推定には2D/pose差があり、全配列のbit一致ではない。最大要素差は以下のとおり。

| 配列 | 最大差 |
|---|---:|
| player_position | 0.01241016 m |
| gvhmr_aligned_player_position | 0.02470875 m |
| smpl_vertices_local | 0.02784157 m |
| ball_3d | 0.00006098 m |
| player_yaw | 0.330936 deg（円周差） |
| gvhmr_aligned_player_yaw | 0.585866 deg（円周差） |

gvhmr_aligned_player_yawの単純減算では±πのwrapで約2πの差が出るため、円周差も記録した。SMPL body_poseの最大パラメータ差は約0.118rad、human_kp_2dの最大差は正規化座標で0.00940。設定差は出力/引継ぎパス、gvhmr source、保存共通設定のvideo_pathsで、実metadataのvideo_paths/camera_idsは同じだった。

generate_datasetのpipeline_config.yamlは共通設定を保存するため例示video_pathsが残るが、実入力はclip.json由来でmetadataへ記録する。この規則は正本READMEに明記した。

## 通常検証

| 検証 | 結果 |
|---|---|
| 統合後tennis_scene・SLCS結合・DINO cache・残す入口 | 353 passed |
| Ballの前処理・学習/eval・公開predictor/UI・pipeline | 241 passed |
| 変更箇所Ruff / mypy | passed / 49 files passed |
| 先行のGUI/export・archive等回帰 | 303 passed |
| 設定監査 | 6 passed |
| Court ROI/追跡とpipeline/GVHMR | 25 passed / 87 passed, 1 skipped |
| renderingと分数FPS実MP4の事前確認 | 7 passed、6frame×2経路 |

件数は重複しており合算しない。責務整理の初期関連suiteも562件が通過した。skipは旧alignment実験固有のローカル生成物を要求するテストで、今回の実scene確認とは別である。source diff検査は履歴保存用patchの必須context空白を除いて実施する。

[統合後353件ログ](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T231809Z/post_integration_tests.log) / [Ball241件ログ](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T231809Z/ball_preprocessing_tests_final.log) / [mypy49件ログ](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260922T231809Z/post_integration_mypy.log)

実TrackNet 8frameでは正しい保存前処理の復元により4px以内が5→4となった。旧経験値依存のテストは、既存datasetのNumPy前処理との全座標一致と従来のGT中央値上限を確認する形へ変更した。旧失敗ログ・before/afterを残している。モデル入力の最大差は0であり、閾値・GT・checkpointを調整して合わせていない。詳細はrun-ball-checkpoint-normalization-meiji-20260923。

## 動画一覧

| scene | 動画 | 確認画像 |
|---|---|---|
| pipeline | [scene.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualization/scene.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-scene-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-scene-extremes.jpg) |
| pipeline | [ball_detection_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/ball_detection_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-ball_detection_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-ball_detection_viz-extremes.jpg) |
| pipeline | [blcs_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/blcs_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-blcs_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-blcs_viz-extremes.jpg) |
| pipeline | [court_kp_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/court_kp_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-court_kp_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-court_kp_viz-extremes.jpg) |
| pipeline | [gvhmr_alignment_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/gvhmr_alignment_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-gvhmr_alignment_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-gvhmr_alignment_viz-extremes.jpg) |
| pipeline | [gvhmr_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/gvhmr_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-gvhmr_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-gvhmr_viz-extremes.jpg) |
| pipeline | [plcs_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-pipeline-visualize_tasks/plcs_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-plcs_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/pipeline-plcs_viz-extremes.jpg) |
| dataset | [scene.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualization/scene.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-scene-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-scene-extremes.jpg) |
| dataset | [ball_detection_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/ball_detection_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-ball_detection_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-ball_detection_viz-extremes.jpg) |
| dataset | [blcs_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/blcs_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-blcs_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-blcs_viz-extremes.jpg) |
| dataset | [court_kp_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/court_kp_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-court_kp_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-court_kp_viz-extremes.jpg) |
| dataset | [gvhmr_alignment_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/gvhmr_alignment_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-gvhmr_alignment_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-gvhmr_alignment_viz-extremes.jpg) |
| dataset | [gvhmr_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/gvhmr_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-gvhmr_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-gvhmr_viz-extremes.jpg) |
| dataset | [plcs_viz.mp4](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/responsibility_cleanup/20260923T020139Z-dataset-visualize_tasks/plcs_viz.mp4) | [冒頭/中盤/終盤](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-plcs_viz-contact.jpg) / [極端値frame](/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/responsibility_cleanup/20260923T020139Z/dataset-plcs_viz-extremes.jpg) |

## validator

指定2回、試行0回、完了0回。親と同じgpt-6-astra / openai / max、fork_turns=noneの新しいvalidatorを各回起動する。各回の評価中は対象を固定する。

評価結果と指摘の採否は完了後に追記する。
