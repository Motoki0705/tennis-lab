# AGENTS.md

このrepoで働くすべてのAIエージェント（Claude・Codexなど）の入口。AI運用の正本は [.agents/README.md](.agents/README.md)。

## 最初に読むもの

1. このファイル（全作業に効くルール）
2. [.agents/README.md](.agents/README.md): AI運用の原則、情報の置き場所、GitHub運用ルール
3. [.agents/memory/INDEX.md](.agents/memory/INDEX.md): 共有memoryの索引。関係するエントリだけを開く。memoryは規則ではなく検証待ちの観察で、コードと食い違ったらコードが正しい
4. 作業対象ディレクトリの `README.md`

## プロジェクト概要

テニスシーンの3次元再構成をAIで解く。入力はマルチカメラの動画である。各カメラでのボール位置・プレーヤーposeの2次元検出から始め、2D → 3Dの再構築モデルで3D空間へ写像する。

| モジュール | 役割 |
|---|---|
| `src/tasks/ball_detection` | 2Dボール検出 |
| `src/tasks/court_detection` | 2Dコート検出 |
| `src/submodules` | 2Dプレーヤーpose検出、3Dプレーヤーpose推定（GVHMR 移植版。重み・body model は `ckpt/`） |
| `src/tasks/blcs` | 2D ball + 2D court から3Dボール軌道を推論 |
| `src/tasks/plcs` | 2D pose + 2D court から3Dプレーヤーの位置・回転を推論 |

最終的に、GVHMRの3D poseとplcsの3D位置・回転を統合し、コート座標系でプレーヤーの軌跡を3D上に再構築する。

## 開発環境

- Pythonは `.venv/bin/python` で実行する。
- テストは `pytest` で実行する（`-n auto` で並列実行される）。
- コミット時に pre-commit が ruff と mypy を実行する。設定の正本は `.pre-commit-config.yaml` と `pyproject.toml`。

## 開発ルール

- **worktreeで作業する**: コードを変更するタスクは、メインworktree配下の `.claude/worktrees/<task-name>` に専用のgit worktreeを作り、その中で行う。メインworktreeのファイルは変更しない。他の場所にworktreeを作らない。
- **PR本文は日本語で書く。**
- **意味のあるテストを書く**: モジュールを実装・改善したら、テストを作成・改善する。`src/utils` や `src/tasks/base` など、下流への影響が大きいモジュールでは必須。
- **静かなフォールバックを禁止する**: 実験的な試みが多いrepoなので、意図しない動作がまかり通る状況を作らない。データの流れとモデルアーキテクチャには特に注意し、必要なら実データで十分に検証する。コードの挙動（分岐・データの流れ）が一意に定まるようにする。これはモデル出力の決定性を求める規則ではない（拡散モデルなど確率的な出力はよい）。
- **不明点は積極的に質問する**: 方針を人間に明確にしてもらう必要があれば質問する。あいまいなまま動くだけのコードは技術負債になる。時間をかけて品質を追求する。
- **モジュラーに作る**: 再利用が見込めるものは `src/utils` へ切り出す。タスクドメインに閉じたモジュールは、該当タスクのディレクトリ配下に新しいフォルダを作って置いてよい。
- **ドキュメントを二重管理しない**: 同じ事柄を2か所に書かない。置き場所は [.agents/README.md](.agents/README.md#ファイル地図情報の住み分け) で決める。
- **READMEから探索する**: 大きなディレクトリには実装を簡潔にまとめたREADMEがある。まずそれを読む。
- **リファクタリングを優先する**: 単純な追加を続けるとコードが肥大化する。有効なリファクタリング案があれば、機能実装より先に自律的に行ってよい。

## 学習の実行

- **ローカルGPU（既定）**: 必ず [.agents/skills/training-queue/SKILL.md](.agents/skills/training-queue/SKILL.md) を読み、training queue 経由で実行する。GPUに対して複数の学習プロセスを直接・同時に起動しない。worktreeで作業していても、queue state はworktree内に作らず、元のrepo rootの `.training_queue/` を共有する（必要なら `TRAINING_QUEUE_DIR` をrepo rootの `.training_queue/` に明示する）。
- **Colab（ユーザーが指定した場合のみ）**: `scripts/colab` にシェルスクリプトを実装し、Colabではそのシェルを実行するだけにする（ドライブのマウントは別途行う）。
