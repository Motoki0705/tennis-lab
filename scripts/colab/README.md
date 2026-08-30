# Colab scripts

## B01〜B03の3DGS学習・コートアライメント

対象ブランチをcheckoutしたColabのGPUランタイムで、次の1コマンドを実行する。

```bash
!bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh
```

`train/20260829T150257Z/run_b01_b03_alignment.sh`は今回の検証実行を再現するtimestamp固定の入口で、Google Driveのmount、locked依存関係、NHT、DINOv3、入力配置をすべて処理する。GPUでB01→B02→B03の`ingest → reconstruction`を逐次実行し、その裏で直前シーンのラインモデルも同じGPUを共有して推論する。ライン投影後の幾何学的`alignment`はCPU実行する。各シーンはalignmentで終了し、datasetとreportは生成しない。終了時にはraw・視点別weighted・weighted射影のラインヒートマップを本体validatorで再検証する。

失敗経緯、B01が1/2面で止まった原因、3面対応アルゴリズム、GPU/CPU並行実行、推論キャッシュ、Drive成果物、定量結果、PR競合解消の詳細は[`train/20260829T150257Z/REPORT.md`](train/20260829T150257Z/REPORT.md)を参照する。

入力動画、ライン検出checkpoint、DINOv3 checkpointは`/content/drive/MyDrive/tennis_lab/`配下から読み、固定SHA-256と一致しない入力は実行前に拒否する。学習とatomic publishはColab VMのローカルfilesystemで行う。各GPU学習の完了直後に`reconstruction/`をシーンごとのDrive出力へ確定保存し、検証を通過した`alignment/`を同じ出力へ追記する。

court detectionの生確率マップは`<scene>/court-line-inference/<fingerprint>/`へ視点単位で原子的にキャッシュする。fingerprintはモデル・backbone・前処理・実行deviceだけに依存し、後段の投影、コート数探索、受理閾値を変更しても再推論しない。各NPY、閲覧用PNG、進捗manifestは生成直後にDriveの同名ディレクトリへミラーし、各alignment試行ログも`alignment-attempts/`へ保存する。

Colab実行では固定カメラprefixを72視点へ拡張する。設定済み48視点をfit 32・holdout 16へ固定分割し、追加24視点は再選択や再fitに使わずholdout tailへ割り当てるため、最終分割はfit 32・holdout 40となる。既存prefixと同じnested-uniform順序を保つため、48視点キャッシュがある場合は先頭48視点を再利用し、追加24視点だけを推論する。

```text
/content/drive/MyDrive/tennis_lab/outputs/synthetic_data_generation/
  alignment-runs/<UTC時刻>-<commit>/<scene>/
```

Drive/GPUを使わずコマンド契約だけ確認する場合は、次を実行する。

```bash
!bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh --dry-run
```

## Setup modules

`setup/` のシェルは単体の実行入口ではなく、呼び出し元から `source` して利用します。

- `install_deps.sh`: 共通のシステム・Python 依存関係を導入する。
- `path_contract.sh`: Hydra の role root と、その配下の相対パスを実行前に検証する。
- `prepare_archive_dataset.sh`: Drive 上のアーカイブや DINOv3 assets などの特定のモジュールを配置する。
- `prepare_generated_dataset.sh`: BLCS / PLCS の生成データセットを必要時に作成する。

## パス設定

`path_contract.sh` は Hydra の role-based path contract を検証します。root は絶対パス、root配下のdataset・artifact・output・checkpointは相対パスとして呼び出し元から渡してください。絶対保存先を変更するときは leaf 側へ絶対パスを渡さず、対応する root を変更します。契約違反は依存導入・データ生成より前にエラーになります。
