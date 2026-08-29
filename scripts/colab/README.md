# Colab scripts

## B01〜B03の3DGS学習・コートアライメント

対象ブランチをcheckoutしたColabのGPUランタイムで、次の1コマンドを実行する。

```bash
!bash scripts/colab/run_b01_b03_alignment.sh
```

`run_b01_b03_alignment.sh`はGoogle Driveのmount、locked依存関係、NHT、DINOv3、入力配置をすべて処理する。GPUでB01→B02→B03の`ingest → reconstruction`を逐次実行し、その裏で直前シーンのラインモデルも同じGPUを共有して推論する。ライン投影後の幾何学的`alignment`はCPU実行する。各シーンはalignmentで終了し、datasetとreportは生成しない。終了時にはraw・視点別weighted・weighted射影のラインヒートマップを本体validatorで再検証する。

入力動画、ライン検出checkpoint、DINOv3 checkpointは`/content/drive/MyDrive/tennis_lab/`配下から読み、固定SHA-256と一致しない入力は実行前に拒否する。学習とatomic publishはColab VMのローカルfilesystemで行う。各GPU学習の完了直後に`reconstruction/`をシーンごとのDrive出力へ確定保存し、検証を通過した`alignment/`を同じ出力へ追記する。

```text
/content/drive/MyDrive/tennis_lab/outputs/synthetic_data_generation/
  alignment-runs/<UTC時刻>-<commit>/<scene>/
```

Drive/GPUを使わずコマンド契約だけ確認する場合は、次を実行する。

```bash
!bash scripts/colab/run_b01_b03_alignment.sh --dry-run
```

## Setup modules

`setup/` のシェルは単体の実行入口ではなく、呼び出し元から `source` して利用します。

- `install_deps.sh`: 共通のシステム・Python 依存関係を導入する。
- `path_contract.sh`: Hydra の role root と、その配下の相対パスを実行前に検証する。
- `prepare_archive_dataset.sh`: Drive 上のアーカイブや DINOv3 assets などの特定のモジュールを配置する。
- `prepare_generated_dataset.sh`: BLCS / PLCS の生成データセットを必要時に作成する。

## パス設定

`path_contract.sh` は Hydra の role-based path contract を検証します。root は絶対パス、root配下のdataset・artifact・output・checkpointは相対パスとして呼び出し元から渡してください。絶対保存先を変更するときは leaf 側へ絶対パスを渡さず、対応する root を変更します。契約違反は依存導入・データ生成より前にエラーになります。
