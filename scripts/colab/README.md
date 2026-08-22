# Colab scripts

Colab では Google Drive を `/content/drive` にマウントした後、対象の train スクリプトを実行する。train の配置規則は [`train/README.md`](train/README.md) を参照する。

`setup/` は train から `source` される内部モジュールであり、Colab から直接実行しない。

- `install_deps.sh`: 共通のシステム・Python 依存関係を導入する。
- `path_contract.sh`: Hydra の role root と、その配下の相対パスを実行前に検証する。
- `prepare_archive_dataset.sh`: Drive 上のアーカイブや DINOv3 assets などの特定のモジュールを配置する。
- `prepare_generated_dataset.sh`: BLCS / PLCS の生成データセットを必要時に作成する。

## パス設定

Colab train スクリプトでは、Hydra の role-based path contract に合わせて root とその配下のパスを分ける。

- `DATA_ROOT`, `ARTIFACT_ROOT`, `OUTPUT_ROOT`, `CHECKPOINT_ROOT`: 絶対パス。
- `DATASET_DIR`, `CHUNKS_DIR`, `OUTPUT_DIR`: 対応する root 配下の相対パス。
- `run.resume`, `run.init_weights`: `CHECKPOINT_ROOT` 配下の相対パス。

絶対保存先を変更するときは leaf 側へ絶対パスを渡さず、対応する root を変更する。契約違反は依存導入・データ生成・学習より前にエラーにする。
