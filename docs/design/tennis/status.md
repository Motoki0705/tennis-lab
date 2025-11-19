# テニススタック実装状況・課題・方針

## 1. 現状整理
- **仕様と設計の基盤**: 詳細な要件は `docs/design/training/tennis_3d_player_pose.md` と実装設計 `docs/design/tennis/tennis_pose_implementation.md` にまとまっており、機能ブレークダウン（P1〜P6）や I/O スキーマが確定している。
- **幾何レイヤ (`src/tennis/geometry/*.py`)**: コート定数・キー点・カメラ生成 (`court.py`) と ViTPose 17 + ラケット 3 のインデックス/ボーン情報 (`skeleton.py`) を提供。`tests/unit/tennis/test_court_geometry.py` / `test_skeleton.py` が形状と定数をカバーしており、P1 で要求されていた最低限の幾何ヘルパは揃った。
- **シミュレーション (`src/tennis/sim/*.py`)**: `schema.py` が JSON/NPZ 風辞書のバリデーションを提供し、`generator.py` は静的ポーズ + ラケットをフェンス上のカメラから投影する最小シーン生成器を実装。ノイズ付与・可視性ドロップ・カメラドロップを備えており、`src/cli/gen_tennis_pose_scenes.py` からバッチ生成、`src/cli/render_tennis_pose_scene.py` + `src/visualize/tennis_pose.py` で可視化まで一通り動く。
- **Config/CLI スケルトン**: `configs/tennis_pose.yaml` は `task: tennis_pose` を設定するトップレベルのみで、dataset/model/training include は未接続。`src/cli/train_tennis_pose.py` は設定読み込みとタスク検証・seed だけを行い、学習処理は未接続のまま安全に終了する。
- **トレーニング統合**: `src/training/utils/config.py` では `cfg.task == "tennis_pose"` の場合 `NotImplementedError` を投げるようにしており、SceneModel 側を壊さずにテニススタックの鉤を用意している。
- **テストと品質**: 幾何テストに加えて `src/tennis/sim/schema.py`/ジェネレータは CLI の `validate_scene_dict` 実行を通じて最小限の検証が行われるが、自動テストは未整備。

## 2. 現在の課題
1. **データ I/O パイプライン未実装**: `src/datasets/tennis_pose.py`、`src/datasets/collate_tennis_pose.py`、`src/training/tennis_pose/datamodule.py` が存在せず、P2 項目（可変カメラ/フレームの読み出し、LightningDataModule 連携）が開始できない。ConfigLoader もここが未達なため学習全体がブロックされている。
2. **学習スタック欠落**: P3 以降で想定していた CourtMVPoseNet (`src/models/tennis_pose/...`)、損失/メトリクス/LightningModule がまだ空で、`configs/models/*.yaml` や `configs/training/*.yaml` も未作成。結果として `python -m src.cli.train_tennis_pose` は常にスケルトン終了となる。
3. **データ品質・多様性**: 生成器は「静的 1 ポーズ + 簡易ラケット」のみで、3DTennisDS リターゲットや時系列動き、現実的なカメラばらつきがない。P1 仕様との差分として、カメラ height/pan 摂動や複数ポーズ、複数プレーヤーの扱いが未達。
4. **評価・可視化ワークフロー不足**: `src/visualize/tennis_pose.py` はフレーム描画のみで、TensorBoard ロギングや定量評価 (MPJPE/PCK/地面拘束) の導線が未整備。`docs/design/...` で定義されたメートル系評価との乖離がある。
5. **ドキュメントと CLI の乖離**: 設計書では includes ベースの設定や実データハンドリングを前提にしているが、現状の `configs/tennis_pose.yaml` / CLI には説明やサンプルコマンド以上の導線がなく、利用者が「どこまでできるか」を把握しにくい。

## 3. 今後の方針
1. **P2（データ I/O）を最優先で実装**
   - Dataset/Collate/DataModule とその設定 (`configs/datasets/tennis_pose_sim.yaml`) を実装し、一度 `ConfigLoader.build_datamodule()` が動くところまで到達させる。
   - 合成シーン JSON 読み込み→ミニバッチを返すまでのスモークテストを `tests/unit/tennis/test_dataset.py`（新設）などで担保。
2. **P3（モデル/学習）を段階導入**
   - CourtMVPoseNet + 損失/LightningModule/Trainer 設定を P3 checklist に沿って追加し、`src/cli/train_tennis_pose.py` から SceneModel と同等の学習ループを走らせる。
   - 合成データのみで 1 エポック学習し損失が減少することを自動テストまたは CI ジョブで確認。
3. **シミュレーション強化と実データ導線**
   - 静的ポーズのみという制限を解消し、3DTennisDS リターゲットと複数フレーム動作を導入。これに合わせ、`configs/datasets/tennis_pose_sim.yaml` のノイズ/欠損パラメータを調整できるようにする。
   - 実データ I/O（P4）のスキーマ整理と CLI（例: 実データ→推論→可視化）の整備を進め、合成→実の橋渡しを早期に確認。
4. **評価・可視化とドキュメント更新**
   - MPJPE/PCK/拘束違反率などのメトリクス計測コード（`src/training/tennis_pose/metrics.py`）と TensorBoard ログ出力を追加し、`src/visualize/tennis_pose.py` を trainer コールバックから呼び出せるようにする。
   - 本ドキュメントと `docs/index.md` / `docs/design/tennis/*.md` を同期し、CLI の使い方・制限を明示。
5. **リスク低減タスク**
   - ConfigLoader の `NotImplementedError` を削除できるタイミングで段階的に `cfg.task == "tennis_pose"` の CI を追加し、SceneModel との回帰を守りながら切り替える。
   - 生成データの統計を可視化し（2D ノイズ分布、visibility 率など）、実データとのギャップを定量的に追跡。

更新日: 2025-11-19
