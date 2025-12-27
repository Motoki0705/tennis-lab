# AGENTS.md: tennis-lab リポジトリ作業ルール (AI Agent / 人間共通)

---
## 1. コンテキストと環境 (Context & Env)

### ディレクトリ構造
ソースコードは `src/` 配下にあり、タスクごとに分かれています。

*   `src/base`: 複数タスクで共有する抽象化・共通I/F（例: `BasePredictor` など）
*   `src/blcs`: 画面上のシーケンシャルなボールの位置とコートのキーポイントを用いて3d上のボールの軌道を推定するモデルの学習環境やそのデータセットをシミュレーションによって生成する。
*   `src/plcs`: 画面上のプレーヤーのキーポイントとコートのキーポイントを用いて3d上のプレーヤーの位置と回転を推定するモデルの学習環境やそのデータセットをシミュレーションによって生成する。
*   `src/wasb`: ボールの位置を画面上の点として特定するモデルの学習環境などを提供する
*   `src/utils`: 共有ユーティリティ（`geometry/`, `rendering/` など）

### 共通実装
複数タスクで共通するロジックは、必ず `src/` 直下または `src/base/`, `src/utils/` に実装し、docstringでその旨を明文化すること。**タスク内での再実装は禁止。**

### 依存・環境
*   **依存管理**: `pyproject.toml`（ロックファイルは `uv.lock`）
*   **Git操作**: `git`

## 2. スクリプトとドキュメント規約（scripts/）

*   **Hydraの強制**: `scripts/` 配下のスクリプトでは構成管理に必ず **hydra** を使用すること。`argparse` の使用は厳禁。
*   **エントリポイント**: 実行可能なスクリプトは `src/{task}/scripts/` に配置する。
*   **Docstring必須**: `scripts/` 配下のファイルには、必ずモジュールレベル docstring を記述する。例えば以下のようなフォーマットで書くこと。
```python
"""Generate PLCS scenes for training using Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.generate_dataset`
    `uv run python -m src.plcs.scripts.generate_dataset run.output_dir=data/plcs simulation.num_scenes=10`

Config entry point: `src/plcs/configs/generate_dataset.yaml`
"""
```

## 3. 使用可能なツール

### `agents_workspace/sub_agents/pre_commit_subagent.sh`
変更ファイル（`git diff --name-only HEAD`）に対して `pre-commit` を実行し、失敗した場合は Codex Sub-agent に修正を委譲します（通常は `uv run --no-sync pre-commit ...` を実行）。ログは `agents_workspace/sub_agents/logs/`（または `CODEX_SUBAGENT_LOG_DIR`）に保存され、標準出力には 1 行の JSON を返します。

**実行例**
```bash
bash agents_workspace/sub_agents/pre_commit_subagent.sh
```
（ヘルプ：`bash agents_workspace/sub_agents/pre_commit_subagent.sh -h`）

**想定出力（例）**
```json
{"status":"pass","fixed":false,"files_touched":[],"remaining_errors":[],"summary":"pre-commit passed","needs_main":false,"message_for_main":""}
```

### `agents_workspace/sub_agents/test_subagent.sh`（推奨：テスト用）
指定したテストコマンド（デフォルトは `uv run --no-sync pytest -q`）を実行し、失敗した場合はログから対象ファイルを抽出して Codex Sub-agent に修正を委譲します。ログは `agents_workspace/sub_agents/logs/pytest_*.log`（または `CODEX_SUBAGENT_LOG_DIR`）に保存され、標準出力には 1 行の JSON を返します。

**基本（デフォルトの pytest）**
```bash
bash agents_workspace/sub_agents/test_subagent.sh
```
（ヘルプ：`bash agents_workspace/sub_agents/test_subagent.sh -h`）

**テスト対象を絞る（例：単体テスト/ケース指定）**
```bash
bash agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q tests/test_example.py::test_case'
```

**想定出力（例）**
```json
{"status":"pass","fixed":false,"files_touched":[],"remaining_failures":[],"summary":"tests passed","needs_main":false,"message_for_main":""}
```
```json
{"status":"fail","fixed":false,"files_touched":[],"remaining_failures":["..."],"summary":"...","needs_main":false,"message_for_main":""}
```

## 4. 推奨ワークフロー（ブランチ作成→作業→検査→テスト）

### AI Agent向け（必須）チェックリスト
このリポジトリでは、AI Agent はユーザーから明示されていなくても **必ず** 以下を満たしてから作業してください（逸脱する場合は、作業前にユーザーへ確認し、最終報告にも理由を明記すること）。

1) **ブランチ作成（必須）**
- **`main` / `master` / `develop` での直接編集は禁止**
- 変更を 1 行でも加える前に、現在ブランチを確認し、`main` 等であれば新規ブランチへ移動する

2) **検査とテスト（必須）**
- 変更後は必ず `pre_commit_subagent.sh` → `test_subagent.sh` の順で実行する（ユーザーが「実行しないで」と言った場合を除く）

3) **例外の扱い（必須）**
- 環境都合（例: `uv run` の権限エラー等）で推奨コマンドが失敗した場合も、回避策を適用して **同等の検査/テストを実行する**
- どうしても実行できない場合は、その理由と、代替で何を確認したかを最終報告に明記する

1) **ブランチを切る（main から）**
（`main` 直作業禁止）
```bash
git checkout main
git pull
git checkout -b feature/<task>-<short-desc>
```
命名規則（推奨）: `<type>/<task>-<short-desc>`（例：`feature/wasb-add-foo`）

2) **作業する**
- 変更・追加を行う（規約は本ドキュメントに従う）

3) **pre-commit を実行（ツール推奨）**
```bash
bash agents_workspace/sub_agents/pre_commit_subagent.sh
```

4) **テストを実行（ツール推奨）**
```bash
bash agents_workspace/sub_agents/test_subagent.sh
```
テストを絞る場合は `--test-cmd` を使用する（例）:
```bash
bash agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q tests/test_example.py::test_case'
```

---
## 5. `uv run` の Permission denied 回避（重要）

Codex 実行環境では、`uv` のデフォルトキャッシュが `/root/.cache/uv` を指し、権限の都合で `Permission denied` になることがあります。

### 推奨: `--cache-dir` を workspace 配下に固定
```bash
uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pytest -q
uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pre-commit run -a
```

### 代替: 一時キャッシュ（遅いが確実）
```bash
uv --no-cache run --no-sync pytest -q
```

### subagent 経由で回避する例
```bash
bash agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync pytest -q tests/...'
bash agents_workspace/sub_agents/pre_commit_subagent.sh
```
