# `src/utils` — shared utilities

`src/utils` は、複数タスクから再利用する実装の置き場です。AI が実装を始めるときは、まずここに既存の共通化先がないかを見ると探索が速くなります。

判断基準は単純で、特定タスクの設定や型に強く依存しない処理はここ、依存する処理は各タスク側です。

## Modules

### Top-level
- **`paths.py`**: `PROJECT_ROOT` と `resolve_project_path()`。repo ルート基準のパス解決が必要なときに見る。
- **`device.py`**: strictな `resolve_device()` と `select_accelerator()`。availabilityに応じた選択は`"auto"`だけが行い、明示CUDA/GPU要求を満たせない場合は`DeviceSelectionError`でmodel/Trainer構築前に失敗する。
- **`seeding.py`**: `seed_everything()` と `make_sample_rng()`。軽量な RNG 初期化や dataloader worker-aware なサンプル単位 RNG を扱う。
- **`io.py`**: ディレクトリ作成、JSON/JSONL の読み書き、atomic write、相対パス化、UTC timestamp 生成、拡張子フォールバック付きファイル探索 `find_existing_file()`。スクリプトやメタデータ保存まわりで最初に見る。
- **`commands.py`**: `subprocess.run(..., check=True)` の薄い共通ラッパー `run_command()`。
- **`hydra.py`**: 型付き `hydra_main()`。CLI エントリポイントで `hydra.main` の型回避を再実装しないための共通化先。
- **`tensor_utils.py`**: `clone_tensor_dict()`、`to_numpy()`、`masked_mean()`、`normalize_padding_mask()`、`flatten_time_to_batch()`/`restore_time_from_batch()`。テンソル辞書の複製、NumPy 変換、mask 付き集約、(B,C,T,H,W)↔(B·T,C,H,W) の変形。

### `configuration/`
- **`paths.py` / `schema.py` / `contracts.py`**: `PathRole`・`PathResolver`・`RuntimePathRoots`、strict schema、typed adapter inspection の正本。設定値や path role の暗黙補完は行わない。
- **`discovery.py`**: runtime boundary を source-only に列挙する下位 discovery の唯一の実装。`catalog.py` と `audit.py` はこの module を一方向に参照する。
- **`catalog.py`**: source declaration と runtime boundary を結び付けた inspectable contract catalog の唯一の import path。
- **`audit.py` / `inventory.py`**: 現在の repository-owned source から configuration/path の禁止パターンを直接検査し、明示的な runtime boundary 契約と照合する library API。行番号依存の migration/exemption snapshot は保持しない。運用 entrypoint は root の `scripts/audit_configuration.py` のみ。

### `data/`
- **`heatmaps.py`**: Gaussian heatmap 生成と、argmax / soft-argmax / peaks / pixel coordinates への復号、`resize_heatmap_sequence()` による (B,T,H,W) の bilinear リサイズ。
- **`augmentation.py`**: keypoint 系 augmentation、visibility dropout、false positive 注入、ImageNet 正規化/逆正規化、`tensor_images_to_uint8_rgb()`、設定値レンジの parse を実装。
- **`scene_io.py`**: scene ディレクトリの `*.npy`、`scalars.json`、`meta.json` をまとめて読む `load_scene_payload()`。
- **`splits.py`**: `GroupSplitConfig` と `make_group_split_map()`。group 単位の deterministic な split を作る。

### `geometry/`
- **`affine.py`**: 画像・点の augmentation で共有する source→destination affine 行列生成、OpenCV / PIL 用変換、点座標変換。
- **`angles.py`**: torch ベースの角度差、角度誤差、軸周り signed angle、ベクトル正規化。
- **`skeleton.py`**: COCO-17 に対する joint angle、torsion、torso twist、bone length 計算。
- **`court_pose.py`**: 正規化コート座標と world pose の相互変換、コート上位置からの平行移動生成。
- **`matrices.py`**: NumPy ベースの回転行列と `apply_plcs_transform[_batch]()`。
- **`rotation_conversions.py`**: torch ベースの axis-angle / quaternion / matrix / 6D 相互変換（PyTorch3D 互換 API、pytorch3d 依存の置き換え先）。
- **`keypoints.py`**: pixel 座標と正規化座標の相互変換、画素座標 clamp。
- **`bbox.py`**: bbox の最大辺比率 `bbox_max_side_ratio()`。bbox の縦横スケール比較が必要なときに見る。
- **`image_size.py`**: `resize_short_side_aligned()`。short side 指定 + 8 の倍数 align の画像サイズ計算。

### `projection/`
- **`camera_projector.py`**: `Camera`、`CameraConfig`、`CameraView`、`CameraProjector`、`make_look_at_camera()`、`project_points()`。ピンホール投影の共通実装。
- **`differentiable_projection.py`**: `DifferentiablePinholeProjection`。任意のworld point shape `(B,...,3)`を固定camera群へ微分可能に投影し、normalized UVと正depth maskを返す。court座標のdenormalizeやtask固有maskは呼び出し側が担当する。

### `rendering/`

タスクの Scene 型・Hydra 設定・ラベル・モデル出力に依存しない描画プリミティブの置き場。tennis_scene / BLCS / PLCS の 3D 可視化はすべてここを直接 import する（task 固有の変換・イベント解釈・HUD 行の選択は各 task 側に残す）。

- **`court_renderer.py`**: コート線・スタイル定義と `CourtRenderer`。3D はランオフ（apron）二色サーフェス・網目ネット・ポスト・センターストラップ対応。ネットサグは `schema.court.net_height_at_x()` に一本化（`net_top_curve()`）。
- **`skeleton_renderer.py`**: COCO / SMPL 系 skeleton 描画用の `SkeletonRenderer` と関連 enum / style。
- **`ball_renderer.py`**: ボール軌跡・イベント描画用の `BallRenderer`、`BallEvent`、`BallStyle`。
- **`mesh_renderer.py`**: 三角メッシュ（SMPL 等）の `MeshRenderer`。カメラ視点オーバーレイ（cv2 painter's algorithm、K 行列投影）と matplotlib 3D 描画。
- **`effects.py`**: 3D 描画エフェクト。フェード付きポリライン（軌跡）、地面の擬似影、バウンスリング（経年で拡大・フェードする `render_impact_ring()` 含む）。
- **`trajectory_analysis.py`**: 軌道からの物理量抽出（純 NumPy）。`compute_speeds()`（フレーム毎速度）と `detect_bounces()`（接地バウンス検出）。
- **`camera_view.py`**: 3D 視点の単一共有 API。`CameraView3D`・視点プリセット・`CameraController`（static / orbit / keyframes、Hydra mapping からの `from_config()`）と、`ax.clear()` 後に毎フレーム呼ぶ `apply_scene_camera()`（view_init + コート固定 framing + zoom）。#630 の `look_at` / `scene_camera` モードはこのモジュールを拡張して実装する（並行実装を作らない）。
- **`theme.py`**: `SceneTheme`（light / dark）。figure / axes 背景、テキスト色、axes chrome 非表示、full-bleed レイアウト、テーマに合わせた `CourtStyle`。`resolve_theme()`、`apply_figure_theme()` / `apply_axes_layout_3d()`（figure レベル、作成時のみ）、`apply_axes_theme_3d()`（`ax.clear()` 後に毎フレーム）。
- **`layers.py`**: 共有 z-order 規約 `SceneLayer`（surface < ground < net < structure < player < ring < trail < marker < ball < overlay）と、mplot3d の深度ソートを無効化する `enable_explicit_layering()`。
- **`hud.py`**: 汎用 HUD。呼び出し側が組んだテキスト行を `text2D` で描く `render_hud_text()` と、`format_frame_clock()` / `format_speed_kmh()`。Scene 型や「球速」「バウンス」の意味は知らない。
- **`minimap.py`**: plain NumPy 配列（現在位置 dots・トレイル・イベント位置）を受け取るトップダウンミニマップ `MinimapRenderer` / `MinimapStyle`。配列の切り出し・色の意味付けは呼び出し側。

### `schema/`
- **`court.py`**: 物理コート寸法、唯一の固定 `COURT_COORD_SCALE_*`、`CourtConfig`、20 点 court keypoints、court skeleton、`net_height_at_x()`。
- **`court_normalization.py`**: 固定scaleを使うshape-safeなposition / velocity変換とcontract identity。
- **`player.py`**: COCO-17 / SMPL / SMPL-H の keypoint 名、index、skeleton、角度計算用 group、SMPLH-to-COCO 対応。

#### Court coordinate normalization contract

`court.py` の `COURT_COORD_SCALE_X/Y/Z/XYZ` が唯一の数値正本です。
`S = HALF_LENGTH = 11.885 m`、`scale_xyz = (S, S, S)` とし、
`court_normalization.py` のhelperがこの値だけを使用します。position は
`position_norm = position_m / scale_xyz`、velocity は
`velocity_norm = velocity_mps / scale_xyz` で表現します。
逆変換は同じ `scale_xyz` を掛けます。物理コート寸法、軸、原点、metre単位の
world座標、およびPLCSのroot-relative `canonical_pose_3d`は変更しません。

normalized BLCS/PLCS sceneとcheckpointは
`court_coordinate_normalization` metadataを必須とします。この変更より前のartifact、
metadata欠落、不明なidentity、scale/単位不一致はload/resume/evaluation/inference前に
明示的なerrorとなります。自動推測・自動変換は行わないため、normalized datasetは
再生成し、checkpointは新契約で再学習してください。

### `video/`
- **`reader.py`**: OpenCV ベースの `probe_video_info()`、単フレーム読み出し、`OpenCVVideoFrameReader`、一括 RGB 読み出し `read_video_rgb()`、smart-seek 付きランダムアクセス `RandomAccessVideoReader`。
- **`audio.py`**: PyAV ベースの音声トラック読み出し `read_audio_mono()` と RMS `audio_envelope()`。音声ベースのカメラ同期が主な用途。
- **`writer.py`**: PyAV ベースの H.264 `VideoWriter`（fps / CRF 指定）と `save_video_rgb()`。
- **`windows.py` / `batching.py`**: 時系列 window / batch iterator、`build_window_starts()`（末尾 anchor 付き sliding-window 開始位置）、`chunked()`。
- **`prefetch.py`**: イテレータの先読み `PrefetchIterator`。
- **`video/transforms.py`**: `BgrToTensorTransform` と tensor の ImageNet 正規化。
- **`encoding.py`**: JPEG エンコードと、選択フレームの JPEG 列挙。
- **`sampling.py`**: 秒指定の parse、time range ベース sampling、uniform sampling。
- **`types.py`**: `VideoInfo`、`FramePacket`、`TemporalWindow`、`TemporalBatch`。
- **`youtube.py`**: YouTube ダウンロード済み動画の探索、ダウンロード、H.264 向け encoder args、transcode。

### `models/`
- **`__init__.py`**: `__all__`に列挙した高頻度model primitiveのcanonical public API。列挙外の専門APIは責務別sub-packageが公開元となる。
- **`components/`**: Transformer の基本部品。attention、RoPE、FFN、norm、`TransformerBlock`、`CrossAttnBlock` に加え、fixed-query multi-view modelで共有するmHC object-temporal → global spatial → query-temporalの`FixedQueryTrackStage`がここにある。ablation専用の`FixedQueryTrackAblationStage`は既存stageを変更せず、`ffn_mode=per_attention|shared`と`mhc_writeback=after_object_temporal|layer_end`を必須にする。前者はAttentionごとの3個のSwiGLUまたはstage末尾の共有SwiGLU 1個、後者はspatial幅`Q+V×P`または`Q+V`を選ぶ。BLCS Eだけが`query_ffn_after_spatial=true`を指定し、shared/layer-end構成のspatial後・query-temporal前へquery-only SwiGLUを追加する。attention forwardはboundaryで検証済みのsame-device `(B,Q,K)` maskと、`RotaryFrequencyComputer`がconstructor時に固定した`(...,T,1,D/2)` complex RoPEだけを受け取り、rank/dtype/device補完や実装選択を行わない。
- **`components/ops/`**: compressed time-local attention の CUDA / reference 実装、autograd bridge、extension loader/build。backendはCSWAの構築時に固定する。
- **`embeddings/`**: court / player / ball の埋め込みとgroup token系の構成要素。`CourtBallGroupEmbedding` / `CourtPlayerGroupEmbedding`はcourtとobject観測を1 element = 1 tokenへ写像し、呼び出し側が与えたleading-axis順を変えずにtoken軸へ保持する。入力はadapterで検証済みの構造化shape（court/playerは末尾`(K,2)`、ballは末尾`2`、visibilityはleading shape）に統一し、flattened旧variantやrank補完は受け付けない。
- **`loading/`**: DINOv3 backbone 読み込み、LoRA 適用、trainability 切り替え。dynamic `forward_features` responseはmodel forward内で検査せず、外部model-I/O boundaryの`require_dinov3_patch_tokens()`が型・key・shapeを検証する。
- **`multiview_padding.py`**: `padding_mask[B,V,T]`（`True=padding`、viewごとに異なる非矩形padding可）からfixed-Q model用のcontext/frame/state validityとdense attention keep-mask（`True=keep`）を一意に構築する。delayed mHC writeback用の`build_compressed_spatial_attention_keep_mask()`も同じpadding-only validityとtoken-0 repairで`Q+V` maskを構築する。
- **`architectures/`**: 現状は `TransformerSequenceDiscriminator` を配置。公開forwardは`sequence[B,T,F]`と`padding_mask[B,T]`（`True=padding`）だけを受け、invalid-token置換、CLS validity、dense attention keep-maskを内部で構築する。
- **`blocks.py`**: `DepthwiseSeparableConv2d` と `Conv2dWiseWiseBlock`。CNN 系の共通ブロック。
- **`heads.py`**: `MLPHead`。
- **`lora.py`**: 汎用 LoRA 実装と trainable parameter 制御。
- **`kimi_delta_attention.py`**: Kimi Linear の中核である Kimi Delta Attention (KDA) の pure PyTorch CPU reference。通常の softmax attention とは別の、stateful な linear attention recurrence。
- **`transformer_utils.py`**: self-attention mask 構築、RoPE base 解決、RoPE 次元検証。
- **`attention_extraction.py`**: attention module 検索と map 抽出の補助。
- **`axial_multiview_mixin.py`**: multi-view 系モデル向けの mixin。

#### Kimi Delta Attention の利用例

公開 API は `from src.utils.models import kimi_delta_attention`。shape は
`query,key,log_decay=[B,T,H,K]`、`value=[B,T,H,V]`、`beta=[B,T,H]`、
state は `[B,H,K,V]`。`valid_mask=[B,T]` は boolean かつ `True=有効` で、
無効 token は state を変更せず zero を出力する。recurrence と返却 state は
float32、sequence output だけが `value.dtype` に戻る。

```python
import torch

from src.utils.models import kimi_delta_attention

B, T, H, K, V = 2, 8, 4, 16, 32
query = torch.randn(B, T, H, K)
key = torch.randn_like(query)
value = torch.randn(B, T, H, V)
log_decay = torch.full_like(query, -0.1)
beta = torch.full((B, T, H), 0.5)
valid_mask = torch.ones(B, T, dtype=torch.bool)

output, final_state = kimi_delta_attention(
    query,
    key,
    value,
    log_decay,
    beta,
    valid_mask=valid_mask,
)

# returned state を次 chunk に渡す呼び出しは、上の full sequence 実行と同値
split = 4
first_output, first_state = kimi_delta_attention(
    query[:, :split],
    key[:, :split],
    value[:, :split],
    log_decay[:, :split],
    beta[:, :split],
    valid_mask=valid_mask[:, :split],
)
second_output, split_final_state = kimi_delta_attention(
    query[:, split:],
    key[:, split:],
    value[:, split:],
    log_decay[:, split:],
    beta[:, split:],
    valid_mask=valid_mask[:, split:],
    initial_state=first_state,
)
torch.testing.assert_close(
    torch.cat((first_output, second_output), dim=1),
    output,
)
torch.testing.assert_close(split_final_state, final_state)
```

current token の state update 後にその token の output を読む inclusive causal
semantics で、future token は参照しない。`log_decay` と `beta` は活性化済みの値を
直接渡し、この関数内で query scaling、sigmoid、softmax、SDPA、別 algorithm への
fallback は行わない。式と dtype/device/state の完全な契約は
`kimi_delta_attention()` の docstring を正本とする。

## Adding a utility

1. まず近い既存モジュールを探す。新規ファイル追加より、既存モジュールへの寄せ先があることが多い。
2. task 非依存の核だけを `src/utils` に置く。task 依存部分は呼び出し側に薄く残す。
3. 公開して再利用させたいものは `__all__` と必要な `__init__.py` に反映する。
4. 共通化したらローカル実装を残して分岐させず、 import に置き換える。
