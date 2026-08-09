# AGENTS.md

## プロジェクト概要

このプロジェクトは、テニスシーンの3次元再構成をAIによって解くことを目的とする。入力はマルチカメラの動画であり、各カメラにおけるボール位置・プレーヤーpose検出という2次元検出から始まり、それらを2D → 3Dへ再構築するモデルが最終的に3D空間へ写像する。

### パイプライン構成

| モジュール | 役割 |
|---|---|
| `src/tasks/ball_detection` | 2Dボール検出 |
| `src/tasks/court_detection` | 2Dコート検出 |
| `src/submodules` | 2Dプレーヤーpose検出 および 3Dプレーヤーpose推定（GVHMR 移植版。重みは `ckpt/` → `third_party/GVHMR/inputs/checkpoints` の symlink） |
| `src/tasks/blcs` | 2D ball + 2D court から3Dボール軌道を推論 |
| `src/tasks/plcs` | 2D pose + 2D court から3Dプレーヤーの位置・回転を推論 |

最終的に、GVHMRの3D poseとplcsの3D位置・回転を統合し、コート座標系におけるプレーヤーの軌跡を3D上で再構築する。

## 開発環境

- Pythonの実行には `.venv/bin/python` を使用する。
- テストは `pytest` で実行する。`-n auto` が設定されているため並列実行される。
- コミット時には `.pre-commit-config.yaml` により以下が実行される。
  - **ruff**: `select = ["E", "F", "UP", "B", "SIM", "I"]`, `ignore = ["F405", "F403", "E501"]`
  - **mypy**: `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `warn_unused_ignores`, `warn_unreachable`, `strict_equality`
  - **task-script-reviewer**: `**/scripts/**` にスクリプトを作成する際の規約（モジュールdocstringの強制）

## 開発スタイル

### リポジトリアーキテクチャ契約

`src/**`、`scripts/**`、設定、path、import、module責務、model I/O、実行境界を変更する前に、[Repository Architecture Contract](docs/architecture/repository-contract.md)を読み、遵守すること。この文書を恒久的なarchitecture ruleの単一の正本とする。

- 計画とPRでは、変更に適用されるrule IDとcanonical ownerを特定する。
- 既存の違反を理由にalias、wrapper、silent fallbackを追加しない。repository-owned consumerを同時に移行するか、契約に従って例外を登録する。
- package READMEは現在の実装案内であり、契約を上書きしない。

### テスト

新しくモジュールを実装・改善した際は、**意味のあるテスト**を作成・改善すること。特に `src/utils` や `src/tasks/base` など、下流に強く影響するモジュールでは必須。

### 静かなフォールバックの禁止

静かなフォールバックは禁止する。必須設定、dependency、data、checkpoint、model I/O contractが不足または不正な場合は、副作用やmodel実行より前のboundaryで明示的に失敗させる。`auto`や代替backendを提供する場合も、選択条件と失敗条件をtyped public contractとして定義する。詳細はRepository Architecture Contractの`CFG-*`、`PATH-*`、`FAIL-*`に従う。

### 不明点は積極的に質問する

上記の制約において、人間による方針の明確化が必要だと感じたら積極的に質問を投げること。あいまいなまま**動くだけのコード**を作ることは、将来的に技術負債となりうる。時間をかけて品質を追求すべき。

### モジュラーな構成

実装は常にモジュラーな構成を目指す。canonical placementは次を原則とし、詳細はRepository Architecture Contractの`OWN-*`に従う。

- task非依存の汎用責務は `src/utils` に置く。
- 複数taskで共有するdata、training、inference、visualization等のlifecycle責務は `src/tasks/base` に置く。
- task domain固有の責務は該当task packageに置く。

### ドキュメントの二重管理禁止

同じ事柄を2つの場所に記述しない。恒久的なarchitecture ruleは`docs/architecture/repository-contract.md`、各packageの現在の構造・実行方法・public APIは該当READMEを正本とする。

### READMEを起点とした探索

大きなディレクトリにはREADMEを設置し、具体的な実装を簡潔にまとめている。AIが開発に取り組む際は、まずREADMEを読み込むことで効率的な探索ができる。

### リファクタリング優先

ある実装を行う際、必要ならばリファクタリングを先に行うこと。機能実装は既存コードへの単純な追加で足りる場合が多いが、それを続けるとコードが肥大化する。有効なリファクタリング案があるなら、自律的に先行して実施してかまわない。

### colabでの学習
ユーザーの指定がある場合、学習はローカルのGPUを用いずに、colabで実行します。その時、`scripts/colab`でシェルスクリプトを実装して、colabではそのシェルを実行するだけにします（ドライブのマウントは別途行う）。指定がない場合はローカルGPUを用います。必ず`.training_queue/`経由での実行にします。