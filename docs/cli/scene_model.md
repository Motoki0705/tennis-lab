# CLI Spec: scene_model

`scene_model` タスク向けの CLI は、主に以下のスクリプトから構成される。

- `src/cli/scene_model/train.py`
- （将来的に追加される補助 CLI があれば、ここに追記する）

## 1. train.py

- **実装**: `src/cli/scene_model/train.py`
- **想定するタスク**: `cfg.task == "scene_model"`
- **代表的な設定ファイル**: `configs/scene_model.yaml`

### 1.1 引数と設定

- `--config PATH`
  - トップレベル YAML のパスを指定する。
  - 例: `configs/scene_model.yaml`
- `--set key=value` 形式
  - Hydra 互換の `key=value` オーバーライド。
  - 例: `--set training.trainer.max_epochs=5`

その他の引数は `argparse` 互換のスタイルで追加される想定だが、基本的な使い方は上記 2 つで十分である。

### 1.2 直接実行例

```bash
uv run python src/cli/scene_model/train.py \
  --config configs/scene_model.yaml \
  --set training.trainer.max_epochs=5
```

### 1.3 scripts/ ラッパとの対応

`scene_model` 学習は、通常は `scripts/train/run_train_scene_model.sh` から起動する。

```bash
./scripts/train/run_train_scene_model.sh
```

- 内部では、次のように `uv run` を呼び出す:
  - `uv run python src/cli/scene_model/train.py --config ${CONFIG}`
- `CONFIG` 環境変数を上書きすることで、別の YAML を指定できる:

```bash
CONFIG=configs/scene_model_debug.yaml ./scripts/train/run_train_scene_model.sh
```

追加の `--set` オプションは、そのままシェルスクリプトに渡せば CLI へ伝播する。
