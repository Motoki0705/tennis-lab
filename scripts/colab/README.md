# Colab scripts

Colab では Google Drive を `/content/drive` にマウントした後、対象の train スクリプトを実行する。train の配置規則は [`train/README.md`](train/README.md) を参照する。

`setup/` は train から `source` される内部モジュールであり、Colab から直接実行しない。

- `install_deps.sh`: 共通のシステム・Python 依存関係を導入する。
- `prepare_archive_dataset.sh`: Drive 上のアーカイブや DINOv3 assets などの特定のモジュールを配置する。
- `prepare_generated_dataset.sh`: BLCS / PLCS の生成データセットを必要時に作成する。
