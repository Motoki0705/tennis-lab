# tennis-lab

`tennis-lab` リポジトリにおける作業ルール：

## 1. コンテキストと環境 (Context & Env)

- **ディレクトリ構造**: ソースコードは `src/` 配下にあり、タスクごとに分かれています。
  - `src/base`: 複数タスクで共有する抽象化・共通I/F（例: `BasePredictor` など）を置くレイヤ。
  - `src/blcs`: BLCS（Ball Localization in Court System）。2Dのボール観測とコート情報から、コート座標系での3Dボール軌道推定を行うタスク一式（`models/`, `data/`, `training/`, `inference/`, `scripts/`, `configs/`）。
  - `src/plcs`: PLCS（Player Localization in Court System）。2D姿勢観測から、コート座標系でのプレイヤー位置・向き（および系列モデル）を推定するタスク一式（`models/`, `data/`, `training/`, `inference/`, `scripts/`, `configs/`）。
  - `src/wasb`: WASB/HRCNet 等を用いたテニスボール半自動アノテーション／データセット拡張のためのパイプライン一式（`pipeline/`, `models/`, `data/`, `training/`, `inference/`, `scripts/`, `configs/`）。
  - `src/utils`: 共有ユーティリティ（主に `geometry/` と `rendering/`）。コート定数・幾何・可視化（コート/スケルトン/ボール描画）を提供。
- **共通実装**: 複数のタスク（blcs, plcs, wasb）で共通するロジックは、必ず `src/` 直下または `src/base/` に実装し、docstringでその旨を明文化すること。各タスクフォルダ内での再実装は避けること。
- **依存**: 依存関係は `pyproject.toml` で管理（ロックは `uv.lock`）。
  - 実行時主要依存（抜粋）: `torch`, `torchvision`, `pytorch-lightning`, `hydra-core`, `omegaconf`, `opencv-python`, `numpy/scipy`, `pandas`, `matplotlib`, `tensorboard`, `tqdm`
  - モデル/推論周辺（必要に応じて）: `transformers`, `accelerate`, `peft`, `ultralytics`, `torchreid`, `deep-sort-realtime`, `bitsandbytes`
  - データ取得/補助: `yt-dlp`, `gdown`, `pillow`, `PyYAML`, `ezc3d`, `smpl`, `smplx`, `termcolor`, `huggingface-hub`, `sam3`
  - 開発用（`[dependency-groups].dev`）: `ruff`, `pytest`, `pytest-cov`, `mypy`, `coverage`, `black`, `types-*`, `pydoclint`

## 2. サンドボックスと禁止事項 (Critical Constraints)

- **Hydraの強制**: `scripts/` 配下のスクリプトでは構成管理に必ず `hydra` を使用すること。**`argparse` の使用は厳禁とする。**
- **学習の制限**:
  - `ML_SANDBOX_DISABLE_GPU` 環境下であると想定せよ。GPUを使用した重い学習や推論を行ってはいけない。
  - 大規模なデータセットのダウンロードや生成を行ってはいけない。テストには必ず `tests/fixtures/` のダミーデータを使用すること。
- **既存テストの保護**: `tests/integration/` 配下のテストは、パイプラインの健全性（`train.py` がエラーなく完走するか等）を保証するものであるため、原則としてロジック変更に伴う修正以外で破壊・変更してはならない。



# ## 3. ワークフロー (Workflow)

`tennis-lab` における開発は、以下の標準ワークフローを必ず遵守すること。
**AI Agent は各ステップで、対応する内容を `agents_workspace/` に記録しながら進めること。**

---

## 3.1 ブランチ作成と準備

1. 作業開始時は、必ず最新の `main` に移動する。
2. `main` を最新状態に更新した上で、新規ブランチを作成する。

* 直接 `main` で作業してはならない。
* ブランチ名は `feature/`, `fix/`, `refactor/`, `exp/` など、目的が分かる接頭辞を付けること。


---

## 3.2 設計 (Design)

実装前に、以下を明確にすること：

* 変更・追加するモジュール（`src/base`, `src/{task}` など）
* 共通化すべきロジックの有無（タスク内再実装は禁止）
* 影響を受けるテスト範囲（unit / integration）
* 制約の確認（Hydra必須 / GPU禁止 / integration test 保護）

### agents_workspace への記録（必須）

* `agents_workspace/design/<branch_name>.md` を作成し、以下を記載すること：

  * 目的（What / Why）
  * 変更対象モジュール
  * 共通化判断とその理由
  * 制約チェック結果
  * 影響範囲（unit / integration）

※ この設計メモは、PRレビュー時の一次資料となる。

---

## 3.3 実装 (Implement)

* 設計に基づきコードを実装する。
* 設定値は原則として Hydra の `.yaml` 経由で注入する。
* `scripts/` 配下のファイルには **必ずモジュールレベル docstring** を記述すること。

### agents_workspace への記録（必須）

* `agents_workspace/implementation/<branch_name>.md` に以下を逐次記録すること：

  * 実装済み内容
  * 未実装 / TODO
  * 設計からの変更点と理由
  * 既存実装を再利用・共通化した箇所

※ 実装ログは「進捗管理」と「設計逸脱の可視化」を目的とする。

---

## 3.4 テスト (Testing)

### 1. ユニットテスト

* 新規・変更ロジックに対応するテストを `tests/unit/` に追加する。
* 既存の `tests/fixtures/` を優先的に再利用すること。

### 2. 統合テスト（影響がある場合）

* パイプライン変更時は `tests/integration/` を用いて疎通確認を行う。
* 既存の統合テストは原則として破壊・削除してはならない。

### 3. テスト実行

* まず変更に関連するテストのみを実行する。
* まとめて確認する場合は全体テストを実行する。

---

## 3.5 ステージング・コミット

1. 変更内容を確認し、ステージングする。
2. コミット時に pre-commit が走り、lint / format が保証される。
3. pre-commit が失敗した場合は、修正後に再度コミットを行う。

---

## 3.6 反復・PR作成

* 必要に応じて「設計 → 実装 → テスト → commit」を反復する。
* 作業完了後、PR を作成する。

### agents_workspace への記録（必須）

* `agents_workspace/prs/<branch_name>.md` を作成し、以下を記載すること：

  * 変更の目的
  * 設計上の判断
  * テスト範囲（unit / integration）
  * レビュー時に注意すべき点

※ この内容は `gh pr create` にそのまま使用できることを想定する。

### コマンドについて
- python系のコマンドは`uv run` を用いること。
- ステージング、コミットなどは`git`コマンド、PRは`gh`を用いること。

## 4. ドメイン固有の規約 (Domain Specifics)

### スクリプトとドキュメント
- **エントリポイント**: 実行可能なスクリプトは `src/{task}/scripts/` に配置すること。
- **Docstring**: `scripts/` 配下のファイルには、必ずモジュールレベルの docstring を記述すること。そこには以下を含めること：
  - スクリプトの目的
  - 具体的な実行コマンド例（CLI引数を含む）
  - Hydraパラメータの説明

### ML/Config (Hydra)
- コンフィグファイルは各タスクの `configs/` ディレクトリで管理する。
- コード内でハードコードされた設定値は避け、`.yaml` ファイル経由で注入すること。

## 5. テスト戦略 (Testing Strategy)

テストディレクトリ構成 (`tests/`)：

```text
tests/
  unit/           # 個別モジュールの機能テスト
  integration/    # scripts/train.py 等の疎通確認
  fixtures/       # 共通のテスト用データ・モデル定義
    toy_data.py   # 小規模かつ決定論的なデータ生成
    tiny_model.py # 本番モデルのI/Fを模倣した極小モデル
    dummy_configs/# テスト用の最小構成yaml
  conftest.py     # seed固定, device切替, 共通fixture
```

- **マーカーの使用**: テスト関数には必ずデコレータをつけること。
  - `@pytest.mark.unit`: 単体テスト
  - `@pytest.mark.integrate`: 統合テスト
- **再利用の原則**: テストデータやモデルが必要な場合は、その場で実装せず `tests/fixtures/` 内の定義を使用すること。存在しない場合のみ、`fixtures/` に新規追加することを検討せよ。
- **統合テストの役割**: `integration` テストは、「学習が開始され、1エポック（または数ステップ）回って終了するか」を確認するためにある。精度や収束を確認するものではない。

# ## 6. Agent Workspace ルール

本リポジトリでは、**AI Agent は思考・設計・判断・進捗を `agents_workspace/` に記録しながら開発を行うことを必須とする。**

---

## 6.1 基本原則

* Agent は **設計・判断理由をコード外に明示的に残すこと**
* `agents_workspace/` は、人間と Agent が共有する開発メモである
* 設計判断・制約解釈・却下した案も記録対象とする

---

## 6.2 記録の単位

* **1ブランチ = 1作業単位**
* `design / implementation / tests / prs` は同一ブランチ名で対応付けること

---

## 6.3 必須ファイル

作業内容に応じて、以下の作成を必須とする：

* 設計変更を伴う場合
  → `agents_workspace/design/<branch_name>.md`
* 実装を行った場合
  → `agents_workspace/implementation/<branch_name>.md`
* PR を作成する場合
  → `agents_workspace/prs/<branch_name>.md`

---

## 6.4 禁止事項

* `agents_workspace/` を更新せずに設計・実装を進めること
* 設計判断をコード内コメントのみに閉じること
* テスト追加の意図を記録しないこと

---

## 6.5 目的

このルールの目的は以下である：

* Agent の判断過程を人間が追跡可能にすること
* 長期的な研究・実装の文脈を失わないこと
* PRレビュー・引き継ぎ・再開を容易にすること