# tennis_scene

同期済みの固定カメラ動画から、camera-local CourtV2観測、物体対応、三角測量、
GVHMRの身体復元を組み合わせてSceneResultを作ります。標準経路はコート点、人物対応、
view_half_turnsの手動入力を要求しません。根拠不足は欠測または理由付き失敗として保存します。

## 標準経路

1. Court hybridでCourtKP14を取得し、camera-localの初期校正から人物検出ROIを作る。
2. DINO＋BoT-SORT＋ViTPoseでcamera-local人物trackと2D poseを収集し、各camera/frameの単一球を検出。
3. camera-local観測とreferenceから、PLCSの独立した人物Re-ID・court sideモデルを推論。ボールは各camera/frameの単一検出を使用。
4. 共通sideを幾何検証し、近似カメラ校正をreference座標へ変換。
5. 人物の同一ID観測と、各カメラの単一球の実観測を三角測量。
6. GVHMRの関節姿勢を保ち、三角測量COCO17へ位置・yawを時系列で配置。
7. 元動画の時間軸でSceneResult、品質mask、診断、stage cacheを保存。

対応範囲は同期・同FPS・同解像度の3〜5 view、各camera累計4人物、球は各camera/frame高々1検出です。
各モデルは約30fpsでclip全体を各1回処理します。短い入力は512へpadし、
実入力の上限は1024です。時間圧縮や暗黙のwindow分割は行いません。
reference未指定時は校正可能camera IDの辞書順先頭を選びます。IDはclip内でのみ有効です。

設定の正本は[configs/pipeline.yaml](configs/pipeline.yaml)です。
Re-IDとsideは別checkpointです。新sideのアーキテクチャは暫定で、今回はRe-IDだけを学習します。既定の配布名は配置規約であり、
重みを自動取得・自動選定する処理はありません。旧3Dモデルへのfallbackもありません。
モデル規模とvalidationで選んだRe-ID閾値はcheckpointが所有します。
2D trackerの設定はpipelineのpeople_modelsが所有します。

```bash
# GPU実行は、このコマンドを共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[match/cam0.mp4,match/cam1.mp4,match/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]' \
  plcs_reid.checkpoint=plcs/player-reid-v2.ckpt \
  court_side.checkpoint=plcs/court-side-v1.ckpt
```

動画はDATA、checkpointはCHECKPOINT、外部モデルはEXTERNAL_ASSET、sceneはOUTPUT、
stage cacheはARTIFACTのrootから解決します。[タスク出力パス](../tasks/OUTPUTS.md)を参照。
dataset生成は実clipから入力を束縛し、設定中のサンプル動画名には依存しません。

## モジュール

| 場所 | 責任 |
|---|---|
| pipeline/orchestrator.py | 構築・同期検証・reference・実行receipt |
| pipeline/model_io/observations.py | pixel観測、confidence、実検出mask、raw検出対応 |
| pipeline/model_io/people.py / body.py | 2D観測と身体復元のtyped adapter |
| pipeline/components/person_observations.py | camera-local追跡ID・bbox・2D pose・実観測maskの収集 |
| pipeline/components/ball_detection.py | 各camera/frameで高々1点の球UV・score・visibility |
| pipeline/components/person_association.py | 独立したPLCS Re-ID/side predictorの遅延ロード・呼出し |
| pipeline/components/camera_geometry.py | Courtから初期校正・ROIを作り、対応人物/球でsideを検証して共通K/R/tを確定 |
| pipeline/components/player_reconstruction.py / ball_reconstruction.py | 人物ID別再構成、身体配置、単一球の三角測量 |
| motion_alignment/ | COCO17への時系列配置とhip/SMPL root差を補正したrenderer変換 |
| pipeline/assembly.py | maskを必須とするSceneResult v2構築 |
| pipeline/artifacts.py | 入力・設定・重み・実装hashを検証するcache |
| pipeline/utilts/ | Court reference・元frame対応などの補助 |

汎用三角測量は[src/utils/geometry/triangulation.py](../utils/geometry/triangulation.py)、
人物モデルの契約は[PLCS仕様](../tasks/plcs/ASSOCIATION.md)が正本です。

標準orchestratorの身体復元は`pipeline/model_io/body.py`のadapterを通ります。
`components/plcs.py`・`blcs.py`・`gvhmr.py`・`player_association.py`は、上記標準経路からは呼びません。

## 座標・対応

観測の正本はpixel座標、モデル入力とsceneの2D座標はpixel/(width,height)です。
Court componentの既存W-1/H-1形式は境界で明示変換します。
CourtKP14のcamera-local順は変えず、半回転は推論後の幾何だけへ適用します。

補間boxは実検出と区別し、observed_maskとjoint confidenceをvisibilityへ反映します。
人物観測0件ではRe-IDを省略します。人物・球の両方が0件ならsideを含む再構成を省略します。
Re-IDは有効な全人物trackをcosineでcamera間の人物groupへまとめ、元動画のID復元にはtracker IDを使います。
人物らしさの補助headや確率閾値によるtrack除外はありません。
補間やUV距離による別の人物trackingを挟みません。無観測の人物にIDは割り当てません。
ボールにはID推論・side推論・候補選択モデルを置きません。

PLCSが推定したsideには、最低evidence、referenceとの接続性、絶対的な幾何品質を
要求します。ボール観測も幾何検証に使いますがside/IDモデルは持ちません。Courtは成功したHのmedoidを選び、
点ごとのmedianで形を作り直しません。校正は単一平面pinhole・無歪みの近似です。

## SceneResult v2

[schema.py](schema.py)と[archive.py](archive.py)がスキーマ・I/Oを所有します。
metadata.scene_schema_version=2では次を必須とし、無効座標は0で保存します。
座標値0から有効性を推定しません。

| mask | shape | 意味 |
|---|---|---|
| player_observed | P,T | 確定IDの実2D観測 |
| player_valid | P,T | SMPL joint0のcourt配置 |
| player_heading_valid | P,T | yaw |
| player_kp_3d_vis | P,T,17 | 三角測量したCOCO17 |
| player_smpl_valid | P,T | 配置した身体mesh |
| ball_3d_valid | T | 球の三角測量 |

player_position/yawは三角測量とGVHMRによる配置です。smpl_vertices_localはroot中心の
canonical posed verticesに人物共通scaleを適用した値、smpl_global_orientは既存renderer式への配置用回転です。
元incamパラメータはbodies artifactに残し、v2ではgvhmr_aligned_*を使いません。

P=0、部分joint欠測、mesh欠測を表現できます。rendererはmaskに従い、mesh不足frameでは
有効COCO17を描画します。軌跡・速度・bounceは欠測を跨ぎません。
SLCS教師maskにはplayer_valid AND player_heading_valid、ball_3d_validを必ずANDします。
2D可視性で無効3Dのweightを復活させません。SLCSの人物数などの適格性制約は維持します。

versionのない旧archiveはv1です。v2でmaskが欠けた場合は拒否し、旧artifactは書換えません。

## 再開と検証

cache.source=executeは同一identityの完了cacheを再利用します。
cache.source=loadは必要なcacheがなければ失敗し、モデルを実行しません。
内容・入力・設定・重みが違うcacheは拒否し、更新にはcache.overwrite=trueを指定します。
cache.directoryの入力hash配下へstage artifactとrun.jsonを保存します。

statusは要求branchに有効結果のあるok、一部だけのpartial、両task無観測のempty、
校正・対応・3Dを成立させられないfailedです。okは全frameの有効性を保証しません。
有効frame数を別記し、emptyではsideや3D provenanceを捏造しません。

モデルと固定slotの検証は[PLCS仕様](../tasks/plcs/ASSOCIATION.md)を参照してください。
学習完了と実動画での校正・3D精度は別に評価します。sideの学習済み新checkpointは未作成です。

## 他の入口

- [generate_dataset](generate_dataset/README.md): 構造化clipへの増分疑似アノテーション。
- [clip_studio](clip_studio/README.md): 同期・ラリーclip切り出し。
- [motion_alignment](motion_alignment/README.md): COCO17への身体配置。
- [synthetic_data_generation](../synthetic_data_generation/README.md): 合成データ生成。
