# Court Detection

DINOv3 + Transformer + multiscale DPTの単一モデルで、コートの
`kp / seg / line / semantic_line / pose`を推定します。
出力先は[タスク出力規約](../OUTPUTS.md)、UIの起動方法は
[Web UI利用ガイド](visualization/README.md)を参照してください。

## 学習データ

学習入力は次の2種類です。画像と疎なラベルを読み、dense教師は学習解像度で
オンザフライ生成します。保存済みのdenseマスクへのfallbackはありません。

| 入力 | 保存先（data rootからの相対パス） | 教師 |
|---|---|---|
| TennisCourtDetector実画像 | `court_detection/tennis_court_detector-v1` | ordered KP14 |
| Synthetic Court V3 | `synthetic_data_generation/scenes/<scene>/datasets/court` | camera-view KP14、対象コート、camera poseと内部パラメータ |

TennisCourtDetectorは`dataset.json`、`index.npz`、JPEG shardを必須とします。
旧JSON＋個別画像の読み込み・移行機能は提供しません。`train / val`だけを持ち、
validationをtestとして代用しません。退化した注釈`QszoUKyCOHo_600`はsource設定で
明示的に除外し、除外IDがちょうど1件に一致しなければ停止します。

Synthetic consumerは`schema: v3`、`court_scope: target_court`専用です。
投影に複数コートがあっても、target bindingが指定する1面だけを全教師で共有します。
V1/V2、all-courtsは拒否します。splitは`train / validation / test`を
`train / val / test`へ対応付け、空splitとtrajectory group leakageを拒否します。
生成・保存形式・camera-view KP14の意味と座標契約の正本は
[Synthetic Court](../../synthetic_data_generation/dataset/court/README.md)です。

## モデルと教師

タスク固有モデル実装は`models/dinov3_dpt.py`の1ファイルです。
DINOv3の4段の特徴のうち最深段をTransformerで処理し、DPTが4段を融合します。
Transformerのpose queryをpose headへ、DPT特徴を4つのresidual dense headへ渡します。
DINOv3のロード・LoRA・共通Transformer部品は`src/utils/models`を利用します。
CNN encoder、FPN、U-Net、Transformerなし、linear headの選択肢はありません。

| 出力 | 内容 | 実画像 | 合成V3 |
|---|---|---|---|
| KP | 14 channel、各channel 1点のGaussian heatmap | 学習 | 学習 |
| SEG | backgroundを含む7クラスのコート領域 | 学習 | 学習 |
| LINE | binary line 1 channel | 学習 | 学習 |
| semantic LINE | backgroundを含む12種類の線 | 学習 | 学習 |
| pose | カメラ位置・回転・焦点距離（raw pose10d） | 教師なし | 学習 |

4つの画像教師は同じ幾何変換を共有します。SEG・LINE・semantic LINEはKP14から
推定したhomographyと物理線幅を使って生成し、元画像外やpaddingには外挿しません。
境界と細線は8倍の格子で描画し、画素内の被覆率を近似します。投影頂点を整数画素へ
丸めて細いポリゴンを捨てる処理や、KP位置への円による線の補修は新教師では使いません。
SEGは`float32 [7,H,W]`、semantic LINEは`float32 [12,H,W]`のクラス分布、
LINEは`float32 [1,H,W]`の0〜1の被覆率です。各画素のクラス分布は合計1で、
元画像外・paddingは背景1（LINEは0）です。クラスIDを補間せず、各クラスの被覆率を求めます。

categorical CE/Diceには分布をそのまま、LINEのBCE/Diceには被覆率を渡します。
IoU/Diceは既存のargmax・閾値による予測に対し、GT側を被覆率で重み付けします。境界の小数値をGT側で閾値化しません。
旧ハード教師とはschemaを区別し、そのスコアと新スコアを同一条件の結果として扱いません。
教師schemaの正本は`target_schemas.py`です。既定のKP sigmaは画像対角長の0.01、
LINE幅は通常線7.5 cm・baseline 15 cmです。各schemaの末尾の版番号は
データsourceのV3とは独立です。

モデル設定は`configs/model/dinov3_dpt.yaml`に集約しています。
既定はViT-B/16、8層のMHA + 2-D RoPE + SwiGLU、DPT large（512 channels）、
各dense headはhidden 256・residual depth 2です。数値設定を変更でき、
checkpointの読み込みでは保存された値を使用します。

DINOv3の外部sourceは `paths.external_asset_root`、学習済み重みは `paths.checkpoint_root` から読む。
相対パスの正本は `configs/model/dinov3_dpt.yaml`。旧source配下の重みへのfallbackは行わない。
checkpoint内に保存された旧 `dinov3/checkpoints/<filename>` は、推論境界で同名の
`dinov3/<filename>` へ明示変換し、警告と `backbone_asset_migration` に記録する。
呼び出し側のresolverを優先し、省略時の旧layoutはprojectの `ckpt/` を使う。
checkpoint本体・保存architectureは変更しない。新配置の資産が無ければ停止する。

学習でも `paths.checkpoint_root=ckpt` を使う。既存の学習出力から再開・初期化するときは
`run.resume={role:artifact,path:court_detection/.../last.ckpt}` または
`run.init_weights={role:artifact,path:court_detection/.../model.ckpt}` を明示する。
これらは `paths.artifact_root=outputs` を参照し、DINOv3の初期weightは引き続きckptから読む。
文字列だけの指定はcheckpoint root相対で、resumeとinit_weightsは同時に指定しない。

## 学習

入口は`src.tasks.court_detection.scripts.train`だけです。
既定は合成4枚＋実画像4枚の固定比率バッチ、4つのdense lossとpose lossです。
`data.source`が合成sourceを所有し、`mixed.sources.synthetic_court`がこれを参照します。
実画像sourceは`mixed.sources.tennis_court_detector`で指定します。
混合数の合計は`data.batch_size`と一致させます。

```bash
# このコマンドを共有training queueへ登録する。ローカルGPUへ直接起動しない。
.venv/bin/python -m src.tasks.court_detection.scripts.train \
  'data.source.scene_ids=[B00,B01,B02,B03]' \
  run.output_dir=court_detection/train/dinov3_dpt/s42-001
```

GPU実行手順は[training-queue skill](../../../.agents/skills/training-queue/SKILL.md)に従います。
worktreeも元repoのqueueを共有します。上記のscene集合・run出力先は実験ごとに明示します。
DINOv3の学習方法は`training=lora`や`model.encoder.train_mode`で設定できます。

`pose_supervision_mask`によりpose loss・pose metric・任意のKP–pose consistencyは
合成サンプルだけを対象にします。実画像にposeのゼロ教師を補いません。
pose教師が必要な合成サンプルで欠落していたら、モデル・worker構築前に失敗します。
pose学習では`data/augmentation=pose_safe`を使用します。
`run.test_after_fit=true`は合成の明示的test splitだけを評価します。

`data/processing`はpreviewや教師検査にも使うtarget選択です。学習の既定は`all`で、
`loss=default`はpose lossを無効にする明示的なdense-only実験に使用できます。

## 学習前の確認

`review_dataset.py`は両sourceのGTを表示し、checkpoint・GPUは不要です。
`preview_augmentation.py`は実際の学習tensor、可視KP、各クラスの画素数を確認します。
heatmap previewもこの入口へ統合しています。各drawをoverlayとtarget-onlyの2行で表示し、
LINEの濃度・categoricalの色とalphaに被覆率を反映します。JSONにはfractional pixel数と
クラス別pixel massも保存します。

```bash
.venv/bin/python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=synthetic_court data/processing=all data/augmentation=pose_safe \
  preview.require_pose=true preview.split=val preview.max_samples=4

.venv/bin/python -m src.tasks.court_detection.scripts.preview_augmentation \
  data/source=tennis_court_detector data/processing=all \
  preview.split=train preview.max_samples=4
```

YouTubeの取得・20点注釈・専用previewは、このタスクの提供範囲に含めません。

## 推論・checkpoint

既定checkpointは`ckpt/court_detection/multiscale_depth3/b863df1f01f0.ckpt`です。
4つのdense headとpose headを持ち、residual depthは3です。
`CourtPredictor.load_from_checkpoint`は保存モデル構成・target bundle・解像度を読み、
全モデル重みを`strict=True`でロードします。過去の別アーキテクチャへのfallbackはありません。
学習専用のsource/run設定は推論のために補完しません。
`outputs/`内のモデルは`visualization.checkpoint={role:artifact,path:court_detection/.../model.ckpt}`で指定します。
Python APIでは同じresolverと`checkpoint_role=PathRole.ARTIFACT`を渡し、
事前学習backboneは`paths.checkpoint_root`から読みます。

`CourtPredictor.predict(rgb, postprocess="hybrid")`はraw headとhomographyを分けて返します。
ordered KP14とLINEでHを推定し、失敗理由を返します。別Hへの代替はありません。
下流の採用点数`max_kp`は4〜8です。`downstream_keypoints()`は再投影14点と
画像内validityを返し、失敗時はゼロ座標・全不可視です。
raw KPは原画像pixel、raw LINEはnative gridで、結果は両サイズを保持します。

`CourtLinePredictor`などのhead別APIは同じforward経路へ委譲します。
LINEだけの利用は`predict(rgb, heads=("line",), postprocess="none")`と明示します。
固定カメラ向け領域探索は`inference/regions.py`に残し、`tennis_scene`が利用します。
合成データ生成のalignmentも`CourtLinePredictor`を利用します。

KPのcamera-view順序を複数cameraの物理point identityへ自動変換しません。
向きの設定は[tennis_scene](../../tennis_scene/README.md)を参照してください。

## Dataset review / inference UI

GTと予測を原画像へ重ね、KP・SEG・LINE・semantic LINEを比較します。
poseの画面表示は提供しません。dataset catalogは実画像と合成V3だけを受理し、
旧合成sceneや壊れたstoreは理由を表示して無効化します。
checkpointは本文の保存構成・target bundleを検証し、対応しない構成は理由付きで拒否します。
採点対象は教師schemaとchannel意味が一致するlayerだけです。直前のハード教師を持つ
既定b863などは予測を表示できますが、schemaの異なるdense headは警告付きで採点から除外します。
新教師を使う学習は新しいrunとして開始し、旧target bundleからの同一条件resumeとして扱いません。

検証入口は`.venv/bin/python -m pytest tests/unit/tasks/court_detection tests/integration/tasks/court_detection`です。
`local_data`テストは実データ・既定checkpointがある環境で明示的に実行します。
