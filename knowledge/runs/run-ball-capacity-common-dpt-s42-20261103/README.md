# 共通X3D/CSN＋DPTの起動保存確認

500更新のcheckpoint・optimizer・RNG・抽出位置と、7 payloadのDrive snapshotを照合した。
起動担当の終了後も学習が継続している。これは10,000更新の完了やvalidation精度の報告ではない。

[固定レシピ・実行・再開手順](../run-ball-capacity-scaling-20261103/README.md)を使う。
実行コードの版はcampaignのplan.jsonに保存されている。
source bundleとprepared manifestsはinputs snapshot、以降の保存は500更新ごとの世代別snapshotに残る。

[起動証拠](startup.json)に500更新時点のloss・勾配・実時間・VRAMとDrive検証を記録した。
定期ポーリングは行わず、学習終了・異常終了イベントで続きを確認する。
