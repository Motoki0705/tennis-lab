# Colab setup scripts

Colab固有の学習起動スクリプトは廃止されました。現在このディレクトリで管理するのは、必要なセットアップ処理を明示的にsourceするための`setup/`モジュールだけです。Google Driveを `/content/drive` にマウントした後、各処理の実行入口から必要なモジュールをsourceします。

`setup/` のシェルは単体の実行入口ではなく、呼び出し元から `source` して利用します。

- `install_deps.sh`: 共通のシステム・Python 依存関係を導入する。
- `path_contract.sh`: Hydra の role root と、その配下の相対パスを実行前に検証する。
- `prepare_archive_dataset.sh`: Drive 上のアーカイブや DINOv3 assets などの特定のモジュールを配置する。
- `prepare_generated_dataset.sh`: BLCS / PLCS の生成データセットを必要時に作成する。

## パス設定

`path_contract.sh` は Hydra の role-based path contract を検証します。root は絶対パス、root配下のdataset・artifact・output・checkpointは相対パスとして呼び出し元から渡してください。絶対保存先を変更するときは leaf 側へ絶対パスを渡さず、対応する root を変更します。契約違反は依存導入・データ生成より前にエラーになります。
