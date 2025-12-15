# AGENTS.md: tennis-lab リポジトリ作業ルール (AI Agent / 人間共通)

## 0. TL;DR（必読）

*   **Hydra必須**（`scripts/` で `argparse` 禁止）
*   **GPU/大規模DL禁止**（`ML_SANDBOX_DISABLE_GPU` 想定、テストは `tests/fixtures/` を使う）
*   **main直作業禁止**（必ずブランチを切る）
*   **pre-commit/testのノイズはサブエージェントへ**：`agents_workspace/sub_agents/{pre_commit_subagent.sh,test_subagent.sh}` を優先利用する
*   **メインに貼るのはJSONのみ**：サブのstdout（1行JSON）だけを共有し、詳細ログはパスで参照する

---

## 1. コンテキストと環境 (Context & Env)

### ディレクトリ構造
ソースコードは `src/` 配下にあり、タスクごとに分かれています。

*   `src/base`: 複数タスクで共有する抽象化・共通I/F（例: `BasePredictor` など）
*   `src/blcs`: BLCS（Ball Localization in Court System）
*   `src/plcs`: PLCS（Player Localization in Court System）
*   `src/wasb`: WASB/HRCNet 等の半自動アノテーション／データセット拡張
*   `src/utils`: 共有ユーティリティ（`geometry/`, `rendering/` など）

### 共通実装
複数タスクで共通するロジックは、必ず `src/` 直下または `src/base/`, `src/utils/` に実装し、docstringでその旨を明文化すること。**タスク内での再実装は禁止。**

### 依存・環境
*   **依存管理**: `pyproject.toml`（ロックファイルは `uv.lock`）
*   **Python実行**: `uv run`
*   **Git操作**: `git`
*   **PR作成**: `gh`

---

## 2. サンドボックスと禁止事項 (Critical Constraints)

*   **Hydraの強制**
    *   `scripts/` 配下のスクリプトでは構成管理に必ず **hydra** を使用すること。`argparse` の使用は厳禁。

*   **学習の制限**
    *   `ML_SANDBOX_DISABLE_GPU` 環境下を想定。GPUを使用した重い学習や推論は禁止。
    *   大規模なデータセットのダウンロードや生成は禁止。テストは `tests/fixtures/` のダミーデータを使用する。

*   **既存テストの保護**
    *   `tests/integration/` はパイプラインの健全性を保証するため、原則として破壊・削除しない（ロジック変更に伴う最小修正のみ許容）。

*   **サブエージェントの sandbox 注意**
    *   `agents_workspace/sub_agents/` のシェルは `codex exec --sandbox danger-full-access` を使用する。
    *   用途は **pre-commit/test の実行・修正に限定**し、stdoutは **JSONのみ** をメインに共有する（詳細ログはファイル参照）。

---

## 3. ワークフロー (Workflow)

tennis-lab の開発は、以下のサイクルを標準とします。

### 3.0 サブエージェント（pre-commit/test）運用

pre-commit / pytest はログが長くなりやすく、メイン会話のコンテキストを汚しやすい。原則として以下のシェルで委譲し、メインは結果（JSON）だけを受け取って意思決定する。

*   **pre-commit サブエージェント**: `agents_workspace/sub_agents/pre_commit_subagent.sh`
    *   責務:
        *   変更ファイルに対する pre-commit（ruff/mypy 等）実行を担当する
        *   失敗時にログを保存し、必要に応じて codex サブエージェントへ修正を委譲する
        *   メイン会話には結果サマリ（stdout の JSON 1行）だけを返す
    *   出力: stdoutに **1行JSON**（例: `{"status":"pass", ...}`）
    *   詳細ログ: `agents_workspace/sub_agents/logs/pre_commit_*.log`
    *   引数:
        *   なし（ヘルプ: `-h` / `--help`）
*   **test サブエージェント**: `agents_workspace/sub_agents/test_subagent.sh`
    *   責務:
        *   pytest の実行（デフォルトまたは `--test-cmd` 指定コマンド）を担当する
        *   失敗時にログを保存し、ログから関連ファイルを推定して codex サブエージェントへ修正を委譲する
        *   メイン会話には結果サマリ（stdout の JSON 1行）だけを返す
    *   出力: stdoutに **1行JSON**
    *   詳細ログ: `agents_workspace/sub_agents/logs/pytest_*.log`
    *   引数:
        *   `--test-cmd '...'`: 実行するテストコマンドを上書きする（`bash -lc` で実行される）
            *   デフォルト: `uv run --no-sync pytest -q`
            *   例: `./agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q tests/unit'`
            *   例: `./agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q -m integration'`
            *   例: `./agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q -k "some_keyword"'`
        *   ヘルプ: `-h` / `--help`

メインに貼るのは **サブのstdout（JSON 1行）だけ**。必要ならログファイルパスを共有し、ログ全文の貼り付けは避ける。

### 3.1 ブランチ作成
1.  `main` に移動し最新化する。
2.  新規ブランチを作成する（`feature/`, `fix/`, `refactor/`, `exp/` 等）。

> **Note**: `main` で直接作業してはならない。

### 3.2 設計（設計書 + 実装計画 + テスト計画）
実装前に以下を明確化してください：
*   **目的**（What / Why）
*   **変更・追加するモジュール**（`src/base` / `src/{task}` 等）
*   **共通化判断**（再実装禁止の観点で、どこに置くべきか）
*   **影響範囲**（unit / integration）
*   **制約チェック**（Hydra / GPU禁止 / integration保護）

### 3.3 実装
1.  実装（コード変更）
2.  テスト設計の明確化（責務/想定機能/ケース）
3.  テスト作成（原則 unit、必要なら integration）
4.  テスト実行
    *   pytestの失敗が出た場合は、原則 `agents_workspace/sub_agents/test_subagent.sh` を実行し、stdoutの **JSON 1行**のみをメインに共有する。
5.  `git commit`（ここで pre-commit が走る）
    *   pre-commit（ruff/mypy等）のエラーが出た場合は、原則 `agents_workspace/sub_agents/pre_commit_subagent.sh` を実行し、stdoutの **JSON 1行**のみをメインに共有する。
6.  pre-commit が失敗した場合は修正し、再度 commit する（通るまで繰り返す）

### 3.4 PR下書き → PR作成
1.  `gh` コマンドでPRを作成する。

PRには以下を明確に記載してください：
*   変更の目的
*   設計上の判断
*   テスト範囲（unit / integration）
*   実行コマンドと結果の要点

---

## 5. スクリプトとドキュメント規約（scripts/）

*   **エントリポイント**: 実行可能なスクリプトは `src/{task}/scripts/` に配置する。
*   **Docstring必須**: `scripts/` 配下のファイルには、必ずモジュールレベル docstring を記述する。含める内容：
    *   スクリプトの目的
    *   実行コマンド例（`uv run ...`）
    *   Hydraパラメータの説明
*   **Hydra必須**: `scripts/` で `argparse` を使用しない。

---

## 6. テスト方針（運用ルール）

*   変更ロジックには原則 **unit test** を追加する（`tests/unit/{task}/`）。
*   データやモデルは `tests/fixtures/` の再利用を優先する。
*   パイプラインに影響する場合のみ **integration test** を実行する（`tests/integration/`）。
*   `tests/integration/` の既存テストは原則として破壊・削除しない。

テストに関しては PR本文に以下を必ず残してください：
*   対象モジュールの責務・想定機能
*   テスト設計（ケース、境界、例外）
*   実行コマンド
*   実行結果（PASS/FAIL と要点）
*   integration を実行した/しない判断（理由）

---

## 7. pre-commit（品質ゲート）

1.  `git commit` により pre-commit が実行され、lint/format/typeチェックが走ります。
2.  pre-commit が失敗した場合、Agent は **修正して再コミット**し、成功するまで先へ進みません。
