# tennis_scene

同期済みの固定カメラ動画から、camera-local CourtV2観測、人物・球の2D観測、三角測量、
GVHMRの身体復元を組み合わせてSceneResultを作ります。根拠不足は欠測または理由付き失敗として保存します。

## 標準経路

1. 各cameraのframe 0だけをKP＋LINE共同推定し、固定コートの初期校正と人物検出ROIを作る。
2. DINO＋BoT-SORT＋ViTPoseでcamera-local人物trackと2D poseを収集し、各camera/frameの単一球を検出。
3. camera間の人物対応（`player_association`）とcourt side（`court_side`）を読み込む。
4. 共通sideを幾何検証し、近似カメラ校正をreference座標へ変換。
5. 人物の同一ID観測と、各カメラの単一球の実観測を三角測量。
6. GVHMRの関節姿勢を保ち、三角測量COCO17へ位置・yawを時系列で配置。
7. 元動画の時間軸でSceneResult、品質mask、診断を保存。

手順3の2ノードにはまだモデル実装がありません（court side: #932、人物対応: #933）。
既定は`execution.<node>=load`で、同じschemaの確認済みartifactがclip storeに無ければ停止します。
他の手順はCourt・人物・球の手動入力を要求しません。

対応範囲は同期・同FPS・同解像度の3〜5 view、各camera累計4人物、球は各camera/frame高々1検出です。
camera alignmentと身体viewの選択は、clip全体を約30fpsの格子で1回処理し、格子が
`frame_sampling.max_frames`を超えるclipは切り詰めずに拒否します。
reference未指定時は校正可能camera IDの辞書順先頭を選びます。IDはclip内でのみ有効です。

設定の正本は[configs/pipeline.yaml](configs/pipeline.yaml)です。既定値は実clip（Meiji）で完走した構成です。
既定の配布名は配置規約であり、重みを自動取得・自動選定する処理はありません。
有効な機能が参照するcheckpointが欠けていれば、実行前のdefinition構築時に停止します。

```bash
# GPU実行は、このコマンドを共有training queueへ登録する。
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  'video_paths=[match/cam0.mp4,match/cam1.mp4,match/cam2.mp4]' \
  'camera_ids=[cam0,cam1,cam2]'
```

動画はDATA、checkpointはCHECKPOINT、外部モデルはEXTERNAL_ASSET、sceneはOUTPUT、
component storeはARTIFACTのrootから解決します。[タスク出力パス](../tasks/OUTPUTS.md)を参照。
dataset生成は実clipから入力を束縛し、設定中のサンプル動画名には依存しません。

## モジュールと成果物

componentのIO宣言、入力組立、runner、clip store、保存形式とexecute/loadの正本は
[pipeline/README.md](pipeline/README.md)です。

## 座標・対応

観測の正本はpixel座標、sceneの2D座標はpixel/(width,height)です。
Court・ball検出器のpixel格子正規化（W-1/H-1）は`src.utils.geometry.keypoints`で境界変換します。
CourtKP14のcamera-local順は変えず、半回転は推論後の幾何だけへ適用します。

補間boxは実検出と区別し、observed_maskとjoint confidenceをvisibilityへ反映します。
無観測の人物にIDは割り当てません。人物・球の両方が0件ならsideを含む再構成を省略します。
ボールにはID推論・side推論・候補選択モデルを置きません。

sideには、最低evidence、referenceとの接続性、絶対的な幾何品質を要求します。
ボール観測も幾何検証に使います。校正はframe 0で採用されたHomographyの投影点に対する
単一平面pinhole・無歪みの近似です。

## SceneResult v2

[schema.py](schema.py)と[archive.py](archive.py)がスキーマ・I/Oを所有します。

`metadata.scene_schema_version=2` のsceneは、再構成の有効性を次のmaskで明示します。
versionのない旧archiveはv1で、maskを持ちません（v1にmaskがあれば拒否します）。
v2でmaskや理由コードが欠けた・矛盾したarchiveは保存・読込とも拒否します。

| mask | shape | 意味 | 理由コード |
|---|---|---|---|
| player_observed | P,T | 確定IDの実2D観測 | — |
| player_valid | P,T | SMPL joint0のcourt配置 | player_rejection_code |
| player_heading_valid | P,T | yaw | — |
| player_kp_3d_vis | P,T,17 | 三角測量したCOCO17 | player_kp_3d_rejection_code |
| player_smpl_valid | P,T | 配置した身体mesh | — |
| ball_3d_valid | T | 球の三角測量 | ball_rejection_code |

無効な座標は0で保存し、座標値0から有効性を推定しません。理由コード0は有効を意味します。
heading・meshは有効なrootを、3Dの人物は実2D観測を、球の3Dは2 view以上の観測を必要とします。
有効な3Dを持つsceneは `metadata.court_reference` を必須とします。検証の正本は
`schema.validate_scene_result_arrays` です。v2では `gvhmr_aligned_*` を使いません。
rendererはmaskに従い、mesh不足frameでは有効COCO17を描画し、軌跡・速度・bounceは欠測を跨ぎません。
SLCSの教師maskは `player_valid AND player_heading_valid` と `ball_3d_valid` を必ずANDし、
2D可視性で無効な3Dのweightを復活させません。

## 他の入口

- [generate_dataset](generate_dataset/README.md): 構造化clipへの増分疑似アノテーション。
- [clip_studio](clip_studio/README.md): 同期・ラリーclip切り出し。
- [motion_alignment](motion_alignment/README.md): COCO17への身体配置。
- [synthetic_data_generation](../synthetic_data_generation/README.md): 合成データ生成。
