# コンフィグアクセス実態調査レポート

## 調査目的
以下の 2 パターンについて、`src/**/scripts/**/*.py`（および `experiments/**/scripts/**/*.py`）を対象にコードベースを調査した。

1. **OmegaConf / Hydra (`DictConfig`) のインスタンスをスクリプト内で保持し、`cfg.xxx` で設定値にアクセスするパターン**
2. **`dataclass` のインスタンスを設定保持用途で持ち、スクリプト内でそのフィールドにアクセスするパターン**

## 調査範囲
- 対象: `src/**/scripts/**/*.py`
- `experiments/**/scripts/**/*.py` は該当ファイルなし
- `.agents/**` は対象外（`src/`/`experiments/` 外）
- 調査時点: 2026-03-27（ブランチ: `copilot/investigate-config-usage`）

## 結果サマリ
- 調査対象スクリプト（`__init__.py` 除く）: **24 ファイル**
- **パターン 1（OmegaConf/DictConfig 直接アクセス）**: 複数ファイルで確認
- **パターン 2（dataclass を設定保持として利用）**: **該当なし**

結論として、現状のスクリプトでは設定アクセスの主流は `DictConfig` (`cfg`) 経由であり、`dataclass` は主にメタデータや統計値など**非設定用途**で使われている。

---

## 1. OmegaConf/DictConfig インスタンス経由の設定アクセス

### 1-1. 典型例（`cfg.xxx` で直接アクセス）

- `src/tasks/blcs/scripts/generate_dataset.py`
  - 例: `cfg.run.seed`, `cfg.run.train_ratio`, `cfg.generator.num_scenes`
- `src/tasks/plcs/scripts/generate_dataset.py`
  - 例: `cfg.run.seed`, `cfg.paths.smplh_model_path`, `cfg.motion_sources.*`
- `src/tasks/plcs/scripts/analysis/analyze_dataset_distribution.py`
  - 例: `cfg.run.seed`, `cfg.data.scene_dir`, `cfg.analysis.mode`
- `src/tasks/ball_detection/scripts/visualize_augmentation.py`
  - 例: `cfg.preview.*`, `cfg.data.*`
- `src/tennis_scene/scripts/run_pipeline.py`
  - 例: `cfg.video_path`, `cfg.device.*`
- `src/tennis_scene/scripts/visualization.py`
  - 例: `cfg.style.*`, `cfg.get(...)`

### 1-2. `DictConfig` から型付き構造へ変換する例

- `src/developing/mae/scripts/produce_epoch_cache.py`
  - `OmegaConf.to_container(cfg.producer.preprocess, resolve=True)` を使って、
    `PreprocessConfig(...)` / `CacheProducerConfig(...)` を構築。
  - ただし起点は `cfg` であり、スクリプト内で `cfg.task.split` 等も参照しているため、
    基本パターンは `DictConfig` 主体。

---

## 2. dataclass インスタンス経由の設定アクセス

### 2-1. 調査結果

`@dataclass` 定義は以下で確認できたが、いずれも**設定保持用途ではない**。

- `src/tasks/ball_detection/scripts/download_videos.py`
  - `SummaryEntry`, `DownloadResult`（ダウンロード結果やサマリ情報）
- `src/tasks/ball_detection/scripts/package.py`
  - `ArchiveSummary`（成果物メタ情報）
- `src/tasks/ball_detection/scripts/train.py`
  - 学習状態・可視化情報の保持用 dataclass
- `src/tasks/plcs/scripts/analysis/analyze_dataset_distribution.py`
  - `RunningStats`（統計量計算用）

### 2-2. 判定

- **「dataclass インスタンスをコンフィグとして保持し、その値をスクリプト内で参照する」実装は見当たらない。**
- `dataclass` はあくまで中間データ／計算補助／出力メタデータの構造化用途。

---

## 3. 補足（実装スタイル）

- 一部スクリプトは `cfg` を runner や builder に渡す薄いエントリポイントになっており、
  スクリプト自身での設定解釈を最小化している。
- 一方でデータ生成系・分析系スクリプトでは、エントリポイントで `cfg` を直接読み込んで分岐・初期化する実装が多い。

## 結論

- 本リポジトリのスクリプトにおける設定アクセスは、**Hydra/OmegaConf (`DictConfig`) ベースが中心**。
- **dataclass ベースの「設定インスタンス」アクセスは確認できなかった**（非設定用途の dataclass 利用は存在）。
