# テニス 3D 位置・ポーズ復元 — 実装設計

本書は「docs/design/training/tennis_3d_player_pose.md」で定義した仕様を、現行リポジトリに最小改修で統合するための実装設計である。既存の SceneModel 系を温存し、テニス用スタック（tennis_pose）を並列追加する。

## 1. 目的とスコープ
- 目的: テニスコート座標系でのプレーヤー 3D 位置・ポーズ（ViTPose 17）とラケット 3 点の復元。
- 入力: 多視点（B-1）および単眼+時系列（B-2）の 2D キーポイント系列（人物/ラケット/コート、可視性含む）。
- 出力: コート座標系 3D の 20 点（人物 17 + ラケット 3）。
- スコープ: 幾何/シミュレーション/データ I/O/モデル/学習/評価/テスト/可視化/運用。

## 2. 既存リポジトリとの統合方針
- スタック分離: `tennis_pose` を新設（SceneModel と独立）。
- 再利用: Lightning/コールバック/ロギング/設定分割のパターンは SceneModel に準拠。
- 段階導入: まず B-1（Multi-View → 3D）を実装し、B-2（Mono + 時系列）は B-1 擬似ラベルで学習。
- ConfigLoader は薄い分岐追加（将来的な動的 import へ拡張可能）。

## 3. ディレクトリ/モジュール構成（新規）
- 幾何・仕様
  - `src/tennis/geometry/court.py` — コート定数/キー点/フェンス/座標系ヘルパ。
  - `src/tennis/geometry/skeleton.py` — ViTPose 17 + ラケット 3 のインデックス/ボーン定義。
- シミュレーション/データ生成
  - `src/tennis/sim/schema.py` — シーン JSON/NPZ スキーマ、検証。
  - `src/tennis/sim/generator.py` — カメラ配置、3DTennisDS リターゲット、投影、ノイズ/欠損。
  - `src/cli/gen_tennis_pose_scenes.py` — 合成データの一括生成 CLI。
- データセット/コラテ
  - `src/datasets/tennis_pose.py` — 合成/実データの統一ローダ。
  - `src/datasets/collate_tennis_pose.py` — 可変カメラ数/可視性/NaN 取り扱いのコラテ。
  - `src/training/tennis_pose/datamodule.py` — LightningDataModule（SceneModel 準拠）。
- モデル/学習
  - `src/models/tennis_pose/court_mvposenet.py` — B-1: Multi-View → 3D 回帰。
  - `src/models/tennis_pose/court_mono_temporal.py` — B-2: Mono + 時系列 → 3D 回帰。
  - `src/training/tennis_pose/losses.py` — 3D L2、ボーン長、地面拘束、手首-ラケット拘束、再投影（任意）。
  - `src/training/tennis_pose/metrics.py` — MPJPE、PCK、物理一貫性指標。
  - `src/training/tennis_pose/lightning.py` — LightningModule（最適化/可視化を内包）。
- 可視化
  - `src/visualize/tennis_pose.py` — 2D/3D オーバーレイ、時系列表示。
- CLI/Config
  - `src/cli/train_tennis_pose.py` — 学習 CLI（SceneModel の UX 踏襲）。
  - `configs/datasets/tennis_pose_sim.yaml` — 合成データ読み込み設定。
  - `configs/models/tennis_mvpose.yaml` — B-1 モデル設定。
  - `configs/models/tennis_mono_temporal.yaml` — B-2 モデル設定。
  - `configs/training/tennis_mvpose.yaml` — Trainer/最適化/ロギング。
  - `configs/tennis_pose.yaml` — includes による束ね設定。

## 4. Config/Loader 設計
- `cfg.task: "tennis_pose" | "scene_model"` を導入。デフォルトは後方互換で scene_model。
- `src/training/utils/config.py` のファクトリを分岐:
  - `build_datamodule()`/`build_lit_module()` が `cfg.task` を見て `tennis_pose` 実装を返す。
- 将来: `cfg.entrypoints.{datamodule,lit_module} = "pkg.mod:Class"` に一般化可。

## 5. シミュレーション/データ生成
- 入力: 3DTennisDS の人体+ラケット MoCap → ViTPose 17 + ラケット 3 にリターゲット。
- カメラ: フェンス上サンプリング、z∈[2.5,3.5]m、中心 LookAt+±5° 摂動。
- 射影: 内部で intrinsics/extrinsics を保持し 3D→2D 射影（学習入力には渡さない）。
- ノイズ/欠損: 仕様に基づくガウスノイズ（人物: 高さ比率、ラケット: 4%、コート: 0.3%H_img）と可視性乱数。
- カメラドロップアウト: p≈0.05。
- 出力: 1 シーン = 1 ファイル（`.json` or `.npz`）。

### 5.1 スキーマ（概略）
- ルート: `scene_id`, `fps`, `num_cameras`, `cameras[]`, `frames[]`。
- `cameras[i]`: `id`, `image_size`（内部: intrinsics/extrinsics は保持可）。
- `frames[t]`:
  - `court_keypoints_2d[i][20][2]` + `visibility[20]`
  - `player_keypoints_2d[i][17][2]` + `visibility[17]`
  - `racket_keypoints_2d[i][3][2]` + `visibility[3]`
  - `player_joints_3d[17][3]`, `racket_points_3d[3][3]`（合成のみ GT）。

## 6. データセット/コラテ
- Dataset: シーンファイルを読み込み、フレーム単位でサンプル化。
  - 可変カメラ数対応（マスク/パディング）。
  - 2D キーポイントは NaN を欠損として保持、別途 visibility マスク。
- Collate: バッチ整形（カメラ/フレーム次元をパディング、`padding_mask` を提供）。
- DataModule: `train/val` DataLoader 構築、乱数 Generator（seed）管理、SceneModel と同等の引数構成。

## 7. モデル設計
### 7.1 CourtMVPoseNet（B-1）
- 入力（時刻 t）: 各カメラ i の 2D（人物17/ラケット3/コート20 + 可視性）とカメラ ID 埋め込み。
- 前処理: 画像サイズで正規化、欠損は visibility でマスク。
- 視点埋め込み: 小型 MLP/Transformer Encoder で `f_i` を抽出。
- マルチビュー統合: Self-Attention または GNN で `{f_i}` を集約（視点数可変/ドロップアウト耐性）。
- 出力: 3D（20×3）を直接回帰（コート座標系）。
- 損失:
  - L2（3D GT）, ボーン長一貫性, 足の z<0 罰則, 手首-ラケット距離制約。
  - 任意で再投影損失（合成のみカメラ利用）。

### 7.2 CourtMonoTemporal（B-2）
- 入力: 単一カメラの時間窓 `[t−L, …, t+L]` の 2D 系列 + 可視性。
- 時系列モデリング: TCN / Transformer / Bi-LSTM。
- 出力: 各フレームの 3D（20×3）。
- 教師: B-1 による擬似 3D（実データ）+ 合成の真値。

## 8. 学習/評価フロー
1) 合成データで B-1 を事前学習 → 幾何・多視点統合を学習。
2) 実多視点データで B-1 を微調整（可能なら三角測量/SfM 併用）。
3) B-1 を実データへ適用し擬似 3D ラベル生成。
4) B-2 を擬似 3D で学習（単眼適用性を獲得）。

- 指標: MPJPE（人物/ラケット別）, PCK@τ, ボーン長分散, 足の地面貫通率, 手首-ラケット距離誤差。

## 9. 可視化/ロギング
- TensorBoard を利用（SceneModel 準拠）。
- 2D: 入力キーポイント分布/欠損率の時系列ヒストグラム。
- 3D: 簡易 3D スケルトン/ラケットのワイヤ描画、または 2D への再投影オーバーレイ（合成時）。

## 10. テスト計画
- 単体
  - court/skeleton 定数の数値一致テスト。
  - 投影/逆投影の往復誤差（ランダム姿勢）。
  - ノイズ/欠損の統計（σ/欠損率の許容帯）。
  - Collate の可変カメラ/フレームのパディングとマスク検証。
- 結合
  - DataModule の `setup()/train_dataloader()/val_dataloader()` スモーク。
  - LightningModule の 1 step fwd/bwd（最小バッチ）。
  - 合成→学習→評価の極小 E2E（エポック1/バッチ1）でロス下降を確認。
- 回帰
  - 乱数 seed 再現性、スモークの安定性。

## 11. 開発フェーズ（チェックリスト）
- [x] P0: ConfigLoader 分岐/CLI/Config 雛形（0.5d）
  - [x] `cfg.task` 導入と後方互換確認（既存 SceneModel 影響なし）
  - [x] `src/cli/train_tennis_pose.py` ひな形追加
  - [x] `configs/tennis_pose.yaml`（includes 骨子）
- [x] P1: 幾何/シミュレーション/スキーマ/生成 CLI（1.5d）
  - [x] `src/tennis/geometry/court.py` 実装＋定数単体テスト
  - [x] `src/tennis/geometry/skeleton.py` 実装＋インデックス検証テスト
  - [x] `src/tennis/sim/generator.py`（カメラ配置/投影/ノイズ/欠損）
  - [x] `src/tennis/sim/schema.py`（スキーマ定義＋バリデーション）
  - [x] `src/cli/gen_tennis_pose_scenes.py`（合成データ生成 CLI）
- [ ] P2: Dataset/Collate/DataModule/単体テスト（1d）
  - [ ] `src/datasets/tennis_pose.py`（合成/実データ I/O 統一）
  - [ ] `src/datasets/collate_tennis_pose.py`（可変カメラ数/可視性マスク）
  - [ ] `src/training/tennis_pose/datamodule.py`（seed/loader 設定）
  - [ ] DataLoader スモーク（1 バッチ取得）
- [ ] P3: CourtMVPoseNet/損失/学習スモーク（2d）
  - [ ] `src/models/tennis_pose/court_mvposenet.py`（視点埋め込み/統合/回帰）
  - [ ] `src/training/tennis_pose/losses.py`（L2/ボーン長/地面/手首-ラケット）
  - [ ] `src/training/tennis_pose/lightning.py`（最適化/ロギング/最小可視化）
  - [ ] 合成データで 1 エポック学習スモーク（ロス下降確認）
- [ ] P4: 実データ I/O/適用（1d）
  - [ ] ViTPose/コート検出 2D 読込スキーマ統一
  - [ ] 実多視点への推論・微調整パス整備
- [ ] P5: CourtMonoTemporal/擬似ラベル学習（2d）
  - [ ] `src/models/tennis_pose/court_mono_temporal.py` 実装
  - [ ] 擬似 3D ラベル生成パイプライン（B-1 適用）
  - [ ] 単眼学習スモーク（短ウィンドウで収束確認）
- [ ] P6: 可視化/評価/Docs 仕上げ（1d）
  - [ ] 指標（MPJPE/PCK/物理一貫性）集計/レポート
  - [ ] 3D/2D 可視化整備（TensorBoard 出力）
  - [ ] ドキュメント更新（仕様/運用/FAQ 追記）

## 12. リスク/対策
- 多視点数の変動/欠損: マスク駆動の統合と Camera-ID 埋め込みで安定化。
- ラケット点の高ノイズ: 正則化（手首-ラケット距離・剛体近似）と時系列平滑（B-2）で緩和。
- 実/合成ドメインギャップ: 2D ノイズ分布のチューニングと実微調整を前提化。
- カメラ未知の単眼推定: コート 2D 配置から暗黙的に視点を推定する設計（B-2）。

## 13. 今後の拡張
- WholeBody（顔/手/足指）への拡張、ラケット点増加。
- 物理シミュレータ連携（足接地拘束、剛体/関節制約の強化）。
- 3D スケルトンの SMPL-X 等へのマッピング。
- SfM/SLAM ベースの弱教師（実データの相対幾何活用）。

---
- 仕様参照: `docs/design/training/tennis_3d_player_pose.md`（座標系/スケルトン/ノイズ設計）。
- 実行導線（想定）:
  - 合成生成: `python -m src.cli.gen_tennis_pose_scenes --out data/tennis_synth --num_scenes 1000`
  - 学習（B-1）: `python -m src.cli.train_tennis_pose --config configs/tennis_pose.yaml task=tennis_pose stage=B1`
  - シーン可視化: `python -m src.cli.render_tennis_pose_scene --scene data/tennis_synth/scene_00000.json --out outputs/scene_00000_cam0.mp4`
  - 擬似ラベル→B-2 学習: パイプラインスクリプトで自動化（別途作成）。
