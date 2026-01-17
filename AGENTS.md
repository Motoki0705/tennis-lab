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

### リポジトリ全体の構成
*   `assets/`: 可視化や資料で利用する静的ファイル
*   `data/`: 生成データや入力データ（大容量になりやすい）
*   `docs/`: 仕様・設計・運用のドキュメント
*   `experiments/`: 実験ログや検証メモ
*   `outputs/`: 実行結果・推論結果・生成物
*   `tests/`: ユニット/統合テスト
*   `third_party/`: 外部コードのミラーやサブモジュール
*   `docker/`: 開発・検証用のコンテナ定義

### 設定ファイル
各タスクの設定は原則 `src/{task}/configs/` に配置し、実行スクリプトはその YAML をエントリポイントとして参照する。

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
*   **関数/クラスのDocstring**: 関数やクラスのDocstringは必ず Google スタイルで記述すること。
    例:
```python
def sample_points(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Sample points uniformly from a point cloud.

    Args:
        points: Input point cloud with shape (N, 3).
        num_samples: Number of points to sample.

    Returns:
        Sampled points with shape (num_samples, 3).
    """

class ExampleModel(nn.Module):
    """Small example model for smoke testing.

    Args:
        in_channels: Number of input channels.
        hidden_dim: Hidden feature dimension.
    """
```

## 3. モジュール単体のスモークテスト（必須の自己確認）

AI Agent が作成・変更したモジュールは、簡単な実行確認ができるように `if __name__ == "__main__":` で小さなスモークテストを提供すること。特に `src/base/` や `src/utils/`、各タスクのモデル/データ処理コンポーネントは、最小限の入力で「実行できる」「期待する形状や型を返す」を確認できるようにする。

例（あくまで最小限・短時間で終わる内容にすること）:
```python
if __name__ == "__main__":
    # quick smoke test for shape/type sanity
    dummy = torch.zeros(1, 3, 224, 224)
    model = ExampleModel()
    out = model(dummy)
    assert out.shape[0] == 1
```

## 4. Git操作ルール（固定フォーマット）

### ブランチ命名
`<type>/<task>-<short-desc>`（例: `feature/wasb-add-foo`）

### コミットメッセージ
`type(scope): summary` を基本とする。`type` は `feat`, `fix`, `refactor`, `docs`, `test`, `chore` を使用する。
例: `feat(plcs): add camera pose sampler`

### PR タイトル/本文テンプレート
タイトル: `type(scope): summary`

本文（簡潔に）:
```
## Summary
- ...

## Testing
- ...
```

## 5. 推奨ワークフロー（ブランチ作成→作業→スモークテスト→ドキュメント）

### AI Agent向け（必須）チェックリスト
このリポジトリでは、AI Agent はユーザーから明示されていなくても **必ず** 以下を満たしてから作業してください（逸脱する場合は、作業前にユーザーへ確認し、最終報告にも理由を明記すること）。

1) **ブランチ作成（必須）**
- **`main` / `master` / `develop` での直接編集は禁止**
- 変更を 1 行でも加える前に、現在ブランチを確認し、`main` 等であれば新規ブランチへ移動する

2) **スモークテストの用意と実行（必須）**
- 変更したモジュールに `if __name__ == "__main__":` の実行確認コードを用意し、短時間で実行できることを確認する
- 可能な限り小さな入力で、形状/型の期待値を assert する

3) **ドキュメントの整合性確認（必須）**
- 変更内容が各タスクの `README.md` やドキュメントに影響する場合は、必ず更新する
- 特に `src/{task}/README.md` は実装と矛盾がないよう常に最新に保つ
- 新機能追加・API変更・設定変更があった場合は、関連ドキュメントの更新を確認する

4) **例外の扱い（必須）**
- 環境都合（例: `uv run` の権限エラー等）で推奨コマンドが失敗した場合も、回避策を適用して **同等の自己確認を実行する**
- どうしても実行できない場合は、その理由と、代替で何を確認したかを最終報告に明記する

### ステップバイステップの手順

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

3) **スモークテストを実行**
- `if __name__ == "__main__":` の確認コードを実行して、短時間で動作確認する

4) **ドキュメントの更新確認**
- 変更がREADMEやドキュメントに影響する場合は更新する
- 特に `src/{task}/README.md` は実装と矛盾がないよう最新に保つ

---
## 6. `uv run` の Permission denied 回避（重要）

Codex 実行環境では、`uv` のデフォルトキャッシュが `/root/.cache/uv` を指し、権限の都合で `Permission denied` になることがあります。

### 推奨: `--cache-dir` を workspace 配下に固定
```bash
uv --cache-dir agents_workspace/tmp_cache/uv_cache run --no-sync python -m src.plcs.scripts.generate_dataset
```

### 代替: 一時キャッシュ（遅いが確実）
```bash
uv --no-cache run --no-sync python -m src.plcs.scripts.generate_dataset
```
