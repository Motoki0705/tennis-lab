# Config / YAML システム Overview

本ドキュメントは、`configs/*.yaml` と `src/training/utils/config.py` による **設定の仕組み** をまとめる。

- `load_cfg()` が YAML をどのように読み込むか
- `includes:` による階層構成
- `ConfigLoader` がどのように DataModule / LightningModule / Trainer を構築するか
- `--set key=value` による上書き方法

SceneModel / テニスいずれのタスクでも共通の仕組みとして設計されている。

---

## 1. YAML 読み込み: `load_cfg()`

実装: `src/training/utils/config.py:load_cfg`

```python
def load_cfg(path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    cfg_path = Path(path)
    base = OmegaConf.load(cfg_path)
    include_map = _container(base.get("includes"))
    merged: DictConfig = OmegaConf.create({})
    for key, rel in include_map.items():
        include_path = _resolve_path(cfg_path.parent / rel)
        part = OmegaConf.load(include_path)
        merged = OmegaConf.merge(merged, OmegaConf.create({key: part}))
    cleaned = {k: v for k, v in base.items() if k != "includes"}
    merged = OmegaConf.merge(merged, OmegaConf.create(cleaned))
    # 省略: CFG_DEBUG_MINIMAL, overrides の反映
    return merged
```

### 1.1 `includes` による階層構成

トップレベル YAML では、次のように `includes` キーで部分 YAML を参照する:

```yaml
# 例: configs/tennis_multi_cam_3d_pose_v2.yaml
task: tennis_multi_cam_3d_pose
experiment_name: tennis_mvpose_dev_v2

includes:
  dataset: datasets/tennis_multi_cam_3d_pose_sim.yaml
  model: models/tennis_mvpose_v2.yaml
  training: training/tennis_mvpose_v2.yaml
  logging: logging/tennis_mvpose.yaml
```

これにより、最終的な `cfg` は次のような階層を持つ:

```yaml
cfg:
  task: tennis_multi_cam_3d_pose
  experiment_name: ...
  dataset: # ← datasets/..yaml の内容
  model:   # ← models/..yaml の内容
  training:# ← training/..yaml の内容
  logging: # ← logging/..yaml の内容
```

SceneModel でも同様に `includes.dataset`, `includes.model`, `includes.training`, `includes.logging` を持つ構成になっている。

### 1.2 CFG_DEBUG_MINIMAL

環境変数 `CFG_DEBUG_MINIMAL=1` を立てると、`{"debug": {"minimal": true}}` がマージされる。

- デバッグ用の軽量設定を YAML 側で `cfg.debug.minimal` を見て切り替える、といった使い方を想定。

---

## 2. コマンドラインからの上書き: `--set key=value`

SceneModel / テニスの CLI では、共通して `--set` 引数を受け取る。

```bash
uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config configs/tennis_multi_cam_3d_pose_v2.yaml \
  --set training.trainer.max_epochs=2 \
  --set dataset.loader.train.batch_size=8
```

CLI 側では `overrides: list[str]` として受け取り、`load_cfg()` に渡す:

```python
cfg = load_cfg(args.config, args.overrides)
```

`overrides` は OmegaConf の dotlist 形式で解釈され、既存の YAML 階層に上書きされる。

- 例: `training.trainer.max_epochs=2`
- 例: `dataset.name=sim_fps60_dur3p0_C4_P1-20_T10`

これにより、**YAML ファイルを直接編集せずに** 実験パラメータを変更できる。

---

## 3. ConfigLoader: cfg からオブジェクトを構築する

実装: `src/training/utils/config.py:ConfigLoader`

主な役割:

- `build_datamodule()`
- `build_lit_module()`
- `build_logger()`
- `build_callbacks()`
- `build_trainer()`

### 3.1 タスク判定: `_task()`

```python
def _task(self) -> str:
    task = self.cfg.get("task")
    return str(task) if task else "scene_model"
```

- `cfg.task` が未設定の場合、後方互換のために `"scene_model"` を既定値とする。

### 3.2 DataModule の構築（レジストリ方式）

`ConfigLoader` は内部に **DataModule レジストリ** を持つ:

```python
_DATAMODULE_REGISTRY: dict[str, Callable[[Any, Any], Any]] = {}

def build_datamodule(self) -> DancetrackDataModule | TennisPoseDataModule:
    task = self._task()
    dataset_cfg = self.cfg.get("dataset")
    debug_cfg = self.cfg.get("debug")
    dataset_keys = list(dataset_cfg.keys()) if dataset_cfg else []

    builder = _DATAMODULE_REGISTRY.get(task)
    if builder is None:
        experiment_name = self._experiment_name()
        msg = (
            f"Unsupported task={task} (experiment_name={experiment_name}) for datamodule"
        )
        self._logger.error(msg)
        raise NotImplementedError(msg)
    datamodule = builder(dataset_cfg, debug_cfg)
    self._logger.info(
        "DataModule built for task=%s (dataset_keys=%s)", task, dataset_keys
    )
    return datamodule
```

デフォルトのレジストリ登録（ファイル末尾の `_register_default_builders()`）では、次のように紐づけられている。

```python
_DATAMODULE_REGISTRY["scene_model"] = _build_scene_datamodule
_DATAMODULE_REGISTRY["tennis_multi_cam_3d_pose"] = _build_tennis_datamodule
```

これにより、タスクごとの DataModule を if/else ではなくレジストリで切り替える設計になっている。

### 3.3 LightningModule の構築（タスク × バージョン）

LightningModule も同様に、**(task, variant) → builder 関数** のレジストリで管理される。

```python
_LIGHTNING_REGISTRY: dict[tuple[str, str], Callable[[DictConfig], Any]] = {}

def _lightning_key(self) -> tuple[str, str]:
    task = self._task()
    if task == "tennis_multi_cam_3d_pose":
        experiment_name = self._experiment_name()
        if "v3" in experiment_name:
            variant = "v3"
        elif "v2_5" in experiment_name:
            variant = "v2_5"
        elif "v2" in experiment_name:
            variant = "v2"
        else:
            variant = "v1"
    else:
        variant = "default"
    return task, variant

def build_lit_module(self) -> ...:
    task = self._task()
    experiment_name = self._experiment_name()
    training_cfg = self.cfg.get("training", {})
    target = str(training_cfg.get("_target_", ""))
    key = self._lightning_key()
    builder = _LIGHTNING_REGISTRY.get(key)
    if builder is None:
        msg = (
            "Unsupported LightningModule selection for task="
            f"{task} (experiment_name={experiment_name}, target={target})"
        )
        self._logger.error(msg)
        raise NotImplementedError(msg)
    self._logger.info(
        "Building LightningModule for task=%s (variant=%s)", task, key[1]
    )
    module = builder(self.cfg)
    self._logger.info("LightningModule built for task=%s", task)
    return module
```

デフォルト登録では、たとえば次のようにバインドされている。

```python
_LIGHTNING_REGISTRY[("scene_model", "default")] = _build_scene_module
_LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v1")] = _build_tennis_v1_module
_LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v2")] = _build_tennis_v2_module
_LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v2_5")] = _build_tennis_v25_module
_LIGHTNING_REGISTRY[("tennis_multi_cam_3d_pose", "v3")] = _build_tennis_v3_module
```

**判定ロジック（テニスタスク）**:

- `experiment_name` に `v3` を含む → v3
- `experiment_name` に `v2_5` を含む → v2.5
- `experiment_name` に `v2` を含む → v2
- それ以外 → v1

これにより、YAML 側の `experiment_name` だけで v1/v2/v2.5/v3 を切り替えられるようになっている。

### 3.4 Logger / Callback / Trainer

```python
def build_callbacks(self) -> list[Callback]:
    from src.training.scene_model.callbacks import build_callbacks
    return build_callbacks(self.cfg.get("logging"))


def build_logger(self) -> Logger:
    from src.training.scene_model.callbacks import build_logger
    experiment_name = self.cfg.get("experiment_name")
    return build_logger(self.cfg.get("logging"), experiment_name)


def build_trainer(self, logger: Logger | bool | None = None,
                  callbacks: Iterable[Callback] | None = None) -> Trainer:
    trainer_cfg = _container(self.cfg.get("training")).get("trainer", {})
    pl_logger = logger if logger is not None else self.build_logger()
    callback_list = list(callbacks) if callbacks is not None else self.build_callbacks()
    return Trainer(logger=pl_logger, callbacks=callback_list, **trainer_cfg)
```

- `cfg.logging` に基づいてロガーとコールバックを構築
- `cfg.training.trainer` を `Trainer(**trainer_cfg)` の引数として解釈

---

## 4. 典型的な構成パターン

### 4.1 SceneModel

- トップレベル: `configs/scene_model.yaml`
- 構成:
  - `includes.dataset`: `configs/datasets/dancetrack.yaml`
  - `includes.model`: `configs/models/scene_model.yaml`
  - `includes.training`: `configs/training/scene_model.yaml`
  - `includes.logging`: `configs/logging/scene_model.yaml`

CLI からの利用例:

```bash
uv run python src/cli/scene_model/train.py \
  --config configs/scene_model.yaml \
  --set training.trainer.max_epochs=2
```

### 4.2 テニス multi-cam 3D pose v1/v2/v2.5/v3

- トップレベル:
  - v1: `configs/tennis_multi_cam_3d_pose.yaml`
  - v2: `configs/tennis_multi_cam_3d_pose_v2.yaml`
  - v2.5: `configs/tennis_multi_cam_3d_pose_v2_5.yaml`
  - v3: `configs/tennis_multi_cam_3d_pose_v3.yaml`

どちらも `includes.dataset` と `includes.logging` は共有し、`includes.model` と `includes.training` だけが異なる。

- v1: `models/tennis_mvpose.yaml`, `training/tennis_mvpose.yaml`
- v2: `models/tennis_mvpose_v2.yaml`, `training/tennis_mvpose_v2.yaml`
- v2.5: `models/tennis_mvpose_v2_5.yaml`, `training/tennis_mvpose_v2_5.yaml`
- v3: `models/tennis_mvpose_v3.yaml`, `training/tennis_mvpose_v3.yaml`

CLI からの利用例:

```bash
# v1
uv run python src/cli/tennis_multi_cam_3d_pose/train.py \
  --config configs/tennis_multi_cam_3d_pose.yaml

# v2 (v2.5/v3 も同じ train CLI を使用)
uv run python src/cli/tennis_multi_cam_3d_pose/train.py \
  --config configs/tennis_multi_cam_3d_pose_v2.yaml
```

---

## 5. 実験を増やす / 設定を分岐させるときの指針

- **新しいモデルバージョン (v3) を追加したい**:
  - `configs/models/tennis_mvpose_v3.yaml` を追加
  - `configs/training/tennis_mvpose_v3.yaml` を追加
  - トップレベル `configs/tennis_multi_cam_3d_pose_v3.yaml` を作成し、`includes.model` / `includes.training` を v3 に向ける
  - `training._target_` を新しい LightningModule に設定
  - 必要に応じて `ConfigLoader.build_lit_module` に v3 分岐を追加

- **データセットだけ変えた実験をしたい**:
  - `configs/datasets/tennis_multi_cam_3d_pose_sim_variant.yaml` を作成
  - トップレベル YAML で `includes.dataset` を差し替える、あるいは `--set dataset.name=...` で切り替える

- **Trainer の設定だけ変えたい**:
  - `--set training.trainer.max_epochs=...` や `--set training.trainer.devices=...` を CLI から与える

---

## 6. 参考ドキュメント

- SceneModel トレーニングパイプライン: `docs/training/scene_model.md`
- テニス Training パイプライン: `docs/training/tennis_multi_cam_3d_pose.md`
- テニス CLI: `docs/cli/tennis_multi_cam_3d_pose.md`
- scripts ラッパ: `docs/scripts/*.md`
