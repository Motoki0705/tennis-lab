# AGENTS.md

このrepoで働くすべてのAIエージェント（Claude・Codexなど）の入口。AI運用の正本は [.agents/README.md](.agents/README.md)。

## 最初に読むもの

**作業を始める前に、必ず共有memoryの [.agents/memory/SUMMARY.md](.agents/memory/SUMMARY.md) を読む。** memoryの読み方・書き方・圧縮・削除は [memory skill](.agents/skills/memory/SKILL.md) に従う。memoryは規則ではなく検証待ちの観察で、コードと食い違ったらコードが正しい。

その後に読むもの:

1. このファイル（全作業に効くルール）
2. [.agents/README.md](.agents/README.md): AI運用の原則、情報の置き場所、GitHub運用ルール
3. 作業対象ディレクトリの `README.md`

## プロジェクト概要

テニス動画から、コート座標系でプレーヤーの3D動作とボール軌道を再構成する研究プロジェクト。同期済みの複数固定カメラ動画を使うシーン再構成と、検出・追跡・3D推定モデルの開発、学習データの生成・注釈・評価を行う。

標準のシーン再構成では、各カメラのコート・人物・pose・ボールの2D観測から、カメラ校正とカメラ間の人物対応を求め、人物の関節とボールを三角測量する。GVHMRで復元した身体を3D関節に合わせて配置し、プレーヤーとボールを共通のコート座標系へ統合する。出力には欠測・有効性maskと品質診断を保持する。

| モジュール | 役割 |
|---|---|
| `src/tennis_scene` | シーン再構成pipeline、実動画からの学習データ生成、結果の可視化 |
| `src/tasks/court_detection` | コートkeypoint・ラインの2D検出 |
| `src/tasks/ball_detection` | ボールの2D検出 |
| `src/tasks/player_detection` | テニス選手検出器の学習・推論 |
| `src/tasks/person_tracking` | カメラ内の人物追跡・選手候補の選別 |
| `src/tasks/court_side` / `player_association` | カメラ間のコート向きの解決・同一人物の対応付け |
| `src/submodules` | 人物検出・2D pose・画像特徴・GVHMRの推論モデル |
| `src/tasks/blcs` / `plcs` | 複数カメラの2D観測とコートkeypointから、ボール軌道／プレーヤー位置・向きなどを推定する学習モデル |
| `src/tasks/slcs` | 単眼の2D観測と画像特徴から、プレーヤーとボールの3D時系列を同時推定する学習モデル |
| `src/tasks/ball_refiner_3d` | 三角測量したボール軌道の補正・欠損補完とイベント推定 |
| `src/synthetic_data_generation` | 実映像のシーン復元・描画を利用した合成学習データ生成 |
| `src/tasks/base` / `src/utils` | 学習・注釈・可視化の共通基盤、幾何・物理・モデルなどの共通部品 |

標準pipelineの構成・入出力は [tennis_scene README](src/tennis_scene/README.md)、各モデルの仕様と実行方法は各モジュールのREADMEを参照する。

## 開発環境

- Pythonは `.venv/bin/python` で実行する。
- テストは `pytest` で実行する（`-n auto` で並列実行される）。
- コミット時に pre-commit が ruff と mypy を実行する。設定の正本は `.pre-commit-config.yaml` と `pyproject.toml`。

## 開発ルール

- **worktreeで作業する**: コードを変更するタスクは、メインworktree配下の `.claude/worktrees/<task-name>` に専用のgit worktreeを作り、その中で行う。メインworktreeのファイルは変更しない。他の場所にworktreeを作らない。
- **PR本文は日本語で書く。**
- **意味のあるテストを書く**: モジュールを実装・改善したら、テストを作成・改善する。テストは設計してから実装する（詳細は [tests/README.md](tests/README.md#テスト作成方針)）。`src/utils` や `src/tasks/base` など、下流への影響が大きいモジュールでは必須。
- **静かなフォールバックを禁止する**: 実験的な試みが多いrepoなので、意図しない動作がまかり通る状況を作らない。データの流れとモデルアーキテクチャには特に注意し、必要なら実データで十分に検証する。コードの挙動（分岐・データの流れ）が一意に定まるようにする。これはモデル出力の決定性を求める規則ではない（拡散モデルなど確率的な出力はよい）。
- **不明点は積極的に質問する**: 方針を人間に明確にしてもらう必要があれば質問する。あいまいなまま動くだけのコードは技術負債になる。時間をかけて品質を追求する。
- **モジュラーに作る**: 再利用が見込めるものは `src/utils` へ切り出す。タスクドメインに閉じたモジュールは、該当タスクのディレクトリ配下に新しいフォルダを作って置いてよい。
- **ドキュメントを二重管理しない**: 同じ事柄を2か所に書かない。置き場所は [.agents/README.md](.agents/README.md#ファイル地図情報の住み分け) で決める。
- **サブエージェントはCodexを既定にする**: 調査・実装などをサブエージェントへ任せるときは、共通ランナー [agent_exec.sh](.agents/ops/lib/agent_exec.sh) 経由のCodexを使う。モデルとeffortは [agents.conf](.agents/ops/lib/agents.conf) の既定（現行はgpt-6.1-sol / max）のまま、`--model`・`--effort` で上書きしない。Claude Codeの組込みサブエージェント（Agent tool）とCodex CLIの直接起動は、ユーザーが明示した場合だけ使う。dshはユーザーがその案件で明示指定した場合だけ、[deepseek-harness skill](.agents/skills/deepseek-harness/SKILL.md) に従って使う。
- **親同士の連絡**: 名前での連絡と受信への応答は [parent-board skill](.agents/skills/parent-board/SKILL.md) に従う。
- **READMEから探索する**: 大きなディレクトリには実装を簡潔にまとめたREADMEがある。まずそれを読む。
- **リファクタリングを優先する**: 単純な追加を続けるとコードが肥大化する。有効なリファクタリング案があれば、機能実装より先に自律的に行ってよい。

## 学習の実行

- **ローカルGPU（既定）**: 必ず [.agents/skills/training-queue/SKILL.md](.agents/skills/training-queue/SKILL.md) を読み、training queue 経由で実行する。GPUに対して複数の学習プロセスを直接・同時に起動しない。worktreeで作業していても、queue state はworktree内に作らず、元のrepo rootの `.training_queue/` を共有する（必要なら `TRAINING_QUEUE_DIR` をrepo rootの `.training_queue/` に明示する）。
- **Colab（ユーザーが指定した場合のみ）**: [.agents/skills/tennis-colab/SKILL.md](.agents/skills/tennis-colab/SKILL.md) を読み、ローカルでの開発とGit経由の反映、Colabでの実行・調査・修正・差分回収を行う。Driveの操作と管理方針は [.agents/skills/tennis-drive/SKILL.md](.agents/skills/tennis-drive/SKILL.md) とそのreferencesを正本とする。
