# tennis_scene

同期済みの固定カメラ動画から、camera-local CourtV2観測、物体対応、三角測量、
GVHMRの身体復元を組み合わせてSceneResultを作ります。標準経路はコート点、人物対応、
view_half_turnsの手動入力を要求しません。根拠不足は欠測または理由付き失敗として保存します。

## 標準経路

1. 各cameraの最初の1frameをKP＋LINE共同推定し、固定コートの初期校正と人物検出ROIを作る。
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

## モジュールと成果物

componentのIO宣言、入力組立、runner、clip store、保存形式とexecute/loadの正本は
[pipeline/README.md](pipeline/README.md)です。人物検出・tracking・2D pose・視点選択・GVHMR・身体配置を
別componentとして扱い、`scene.json`が各成果物と完成した`scene.npz`の版を管理します。

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

実行・再開・外部成果物importは[pipeline仕様](pipeline/README.md#execute--load)を参照してください。
人物Re-IDモデルの契約と学習結果は[PLCS仕様](../tasks/plcs/ASSOCIATION.md)が正本です。
sideの学習済み新checkpointは未作成です。検証用の確認済みsideをloadする場合は、その出自を成果物に記録します。

## 他の入口

- [generate_dataset](generate_dataset/README.md): 構造化clipへの増分疑似アノテーション。
- [clip_studio](clip_studio/README.md): 同期・ラリーclip切り出し。
- [motion_alignment](motion_alignment/README.md): COCO17への身体配置。
- [synthetic_data_generation](../synthetic_data_generation/README.md): 合成データ生成。
