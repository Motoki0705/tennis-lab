# SfM comparison / issue #1034

ARISキャンペーンの実測を既存のtraining-queueとNHTのCLIで保存する。
productionのsceneやconfigは変更しない。作業状態・凍結入力・レポートは
`outputs/aris/i1034-sfm-night-20261006/`、確定した結果はknowledgeへ登録する。

`run_command.py`はqueueからだけ起動し、明示したtimeoutで単一コマンドを実行する。
終了コードだけでなく必要artifactと測定JSONも確認する。失敗は別attempt IDで記録し、
上書きや解像度の暗黙変更をしない。GPUメモリは装置全体のサンプルで、process専有量ではない。

比較元のNHTはcommit `9493001f1f5650f62e8f9809b023618edcde3f03`、PyCOLMAPは4.1.1。
VidMap候補はcommit `1a48f2a1c9b59ba1e28bf40eeba1777f5d34ebb1`。
別々の`.cache/runtimes/`へ依存を導入する。source・environment・入力manifestは
campaignの`environment/`と`inputs/`に記録する。
native環境の構築で判明した制約と復元手順は[ENVIRONMENT.md](ENVIRONMENT.md)を参照する。

初回入力はB00先頭90秒、NHT既定1 fpsと既定quality filterを適用した画像集合。
VidMapの小さいsanityは別条件であり、full inputのSIFTと順位付けしない。
独立したcamera/ground GTがないため、登録率と再投影残差だけで絶対精度を主張しない。
