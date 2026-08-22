# Colab train scripts

train スクリプトは作成日ごとの `YYYY-MM-DD/` に保存する。同日に作成した複数のスクリプトは同じディレクトリに置く。

各スクリプトは必要な setup を内部で実行するため、Colab 側では Drive のマウント後に train スクリプトだけを実行する。

- `2026-07-02/`: 既存の7学習スクリプト。
- `2026-08-22/`: BLCS / PLCS の base-size track-query、および synthetic Court v2 KP の学習スクリプト。
