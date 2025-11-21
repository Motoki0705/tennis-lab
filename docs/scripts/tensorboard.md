# scripts: tensorboard

TensorBoard 関連のスクリプトは `scripts/tensorboard/` にまとめられている。

- `scripts/tensorboard/run_tensorboard.sh`
- `scripts/tensorboard/collect_tensorboard_summaries.sh`
- `scripts/tensorboard/collect_and_summarize.py`

## 1. run_tensorboard.sh

- **役割**: 指定した runs ディレクトリに対して TensorBoard を起動する。

### 1.1 使い方

```bash
./scripts/tensorboard/run_tensorboard.sh
```

- 既定のログディレクトリ: `runs/`
- `RUNS_DIR` 環境変数で変更可能:

```bash
RUNS_DIR=runs/scene_model ./scripts/tensorboard/run_tensorboard.sh --port 7007
```

## 2. collect_tensorboard_summaries.sh

- **役割**: `runs/` 以下の `events.out.tfevents.*` を走査し、CSV と Markdown サマリを生成する。
- **内部で呼び出す CLI**: `scripts/tensorboard/collect_and_summarize.py`

### 2.1 使い方

```bash
./scripts/tensorboard/collect_tensorboard_summaries.sh
```

- 既定の対象ディレクトリ: `runs/`
- 第 1 引数で変更可能:

```bash
./scripts/tensorboard/collect_tensorboard_summaries.sh runs/tennis_multi_cam_3d_pose
```

## 3. collect_and_summarize.py

- **実装**: `scripts/tensorboard/collect_and_summarize.py`
- TensorBoard の event ファイルごとに:
  - scalar 情報を CSV (`*.scalars.csv`) に書き出し
  - サマリ Markdown (`*.summary.md`) を生成する

直接実行する場合は次のように呼び出す:

```bash
uv run python scripts/tensorboard/collect_and_summarize.py \
  --runs-dir runs/tennis_multi_cam_3d_pose
```
