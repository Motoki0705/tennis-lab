球像の測定・目視監査・注釈の来歴を[実験記録](https://github.com/Motoki0705/tennis-lab/blob/0864dce25c4f3d2322994408ff5eacf1b1a80421/knowledge/nodes/ball_detection/000097-run-i2637-ball-image-difficulty.md)にまとめました。時系列比較は共有queueに登録済みです。

- **球像の小ささと縮小による分離の難しさを支持。** 3解像度で計測できた同じ339フレームの径中央値は **5.96 → 4.09 → 1.76 px**。512×288で計測できたのは全2,208件中360件でした。
- **遮蔽と画面外の実例を確認。** 64件の層化目視で遮蔽2件、画面内の識別不能11件、画面外1件、判定保留22件でした。全184,912件のうち画面外の座標中心は4,136件（2.24%）で、うち4,125件は推定値です。全収録の真の不可視率とカメラ画質の因果効果は未確定です。
- **形状ベース基準は20 px以内が7/88件（7.95%）。** mean/P95は458.73/987.17元画像px、存在FPR/FNRは62.50%[local path]。X3Dとi935も同じ144フレームで比較します。
- **注釈の来歴は部分的に支持。** `gpt-6.2-sol / max` は修正版3 clipの実行ログと出力で確認しました。初版69 clipはChatGPTのダウンロードZIPと一致し、生成モデルは未確認です。

HEAD `0864dce25c4f3d2322994408ff5eacf1b1a80421`。knowledge検査は754ノードでエラー・警告ともに0件でした。PRはGPU結果を反映して作成します。 [#2637 の実測と投入前見積もり](https://github.com/Motoki0705/tennis-lab/issues/2637#issuecomment-7191452156)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204305540952854139_2026-11-26T12-00-59-047145860Z_363728_ball-paper-temporal-x3d-i935.job"]}}