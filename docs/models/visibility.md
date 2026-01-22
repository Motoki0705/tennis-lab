# Visibility の用途まとめ

このドキュメントは visibility（可視フラグ/観測マスク）の使い方をまとめたものです。
今回の統一方針では、visibility は Attention Mask に使わず、
不可視トークンへの差し替えで表現します。

- Attention Mask は seq_len / num_views / padding の整列にのみ使用
- visibility は「不可視トークン」に差し替え
- 不可視トークンは court/player/ball で共通の 1 つ
- 損失/メトリクスは visibility ではなく padding mask を使用
- データセットは visibility を生成し、collate が padding mask を生成


## 共通埋め込み

- 不可視トークン: `src/common/models/embeddings/shared.py`
- Court: `src/common/models/embeddings/court.py`
- Player: `src/common/models/embeddings/player.py`
- Ball (2D/3D): `src/common/models/embeddings/ball.py`


## データフローの役割分担

- Dataset: `*_vis` を生成（不可視フレーム/トークン判定）
- Collate: `*_mask` を生成（padding/バッチ整列用）
- Model: `*_vis` で不可視トークンに差し替え、`*_mask` を Attention Mask にのみ使用
- Loss/Metric: `*_mask` を使用（visibility 由来のマスクは使わない）


## スコープ内モデル（更新済み）

- BLCSModel
  - visibility は不可視トークンに差し替え
  - Attention Mask は padding/seq_len にのみ使用
  - 参照: `src/blcs/models/blcs_model.py`

- UVEventModel
  - visibility は不可視トークンに差し替え
  - Attention Mask は padding/seq_len にのみ使用
  - 参照: `src/evnet_detection/models/uv_event_model.py`

- Traj3DEventModel
  - visibility 入力なし（不可視トークンは未使用）
  - Attention Mask は seq_len のみ
  - 参照: `src/evnet_detection/models/traj3d_event_model.py`

- PLCSModel
  - visibility は不可視トークンに差し替え
  - Attention Mask は padding/seq_len にのみ使用
  - 参照: `src/plcs/models/plcs_model.py`

- UVTrajectoryCompletionModel
  - visibility/観測マスクは不可視トークンに差し替え
  - Attention Mask は padding/seq_len にのみ使用
  - 参照: `src/trajectory_completion/models/uv_completion_model.py`


## Predictor API の目安

- visibility は `*_vis` で渡す（不可視トークン用）
- padding mask は `*_mask` で渡す（Attention Mask 用）


## スコープ外（未更新）

以下は今回のスコープ外のため、visibility のマスク利用が残っています。

- PLCSSequenceModel: `src/plcs/models/plcs_sequence_model.py`
- BLCSMultiViewModel: `src/blcs/models/blcs_multiview_model.py`
- PLCSMultiViewModel: `src/plcs/models/plcs_multiview_model.py`
