# 公開用記録の整備

初回commit前の点検で、計測値・内容照合値・戻り値・実行オプション・相対pathを保持し、
公開に不要なマシン固有のpathと識別子を置換した。これは転送の再実行記録ではない。

- `${REPO_ROOT}`: 計測元のrepo root。receiptのargv内の値は表記上の置換で、shell展開済みの実行履歴ではない。
- `${RCLONE_CONFIG}` / `${COLAB_RCLONE_CONFIG}`: ローカル／Colabのrclone設定ファイルの場所。
- `${COLAB_STATE}/recovered/${RECOVERY_ID}`: 回収先のローカルstate。
- `<redacted>`: Colab session識別子とDrive inventoryのentry ID。相対path・サイズ・内容hashで照合できる。

JSONの数値・真偽値・内容hashが置換前から変わっていないことを構造比較した。
Markdownには内容hashの値を載せず、JSONの照合情報を参照する。
bundleは約85 KBで、各ファイルは256 KiB未満。rcloneのログは6本の短い転送記録で、
コピー工程の所要時間の根拠として残す。重み・画像・動画・予測配列は含まない。
既存の遠隔実行スクリプトは履歴として保持し、この子タスクでは実行しない。
