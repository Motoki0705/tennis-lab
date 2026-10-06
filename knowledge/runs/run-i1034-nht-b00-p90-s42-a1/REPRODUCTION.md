# 再実行の前提と保存範囲

`repro.sh`、`run.json`、`uncommitted.patch`はqueueの歴史的captureを保持する。元commandのoutputを上書きするために直接実行しない。新しいattempt ID・出力先を用意して共有training-queueから起動する。

実際に使ったconfigは`nht-config.yaml`、入力hashは`input-manifest.json`、native versionは`runtime-requirements.txt`に保存した。元のB00動画と原画像はデータ入力であり、この小さいbundleに複製していない。画像の復元後はmanifestのSHA-256を照合する。

NHT source `9493001f1f5650f62e8f9809b023618edcde3f03`とPyCOLMAP 4.1.1をisolated runtimeに復元する。関連環境の記録は[ENVIRONMENT.md](../../../experiments/sfm_comparison/ENVIRONMENT.md)を参照する。freeze内のlocal editableパスはそのまま他machineへ適用できない。

現行`kg_repro_paths.py`はraw command中の`outputs/.../configs/nht-sift.yaml`を解決できずERRORとなる。config自体は上記bundleへ保存済みだが、checkerは保存先との対応付けや外部native環境の復元を扱わない。この検査を成功扱いにせず、structural knowledge validationと区別する。
