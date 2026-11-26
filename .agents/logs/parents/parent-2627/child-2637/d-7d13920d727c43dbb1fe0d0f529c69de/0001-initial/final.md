[#2637 の評価設計](https://github.com/Motoki0705/tennis-lab/issues/2637#issuecomment-7190734705)を記録し、全69 clipの球像計測と形状ベース検出器のCPU評価を起動しました。

- **注釈の来歴は部分的に支持。** 初版69 JSONはChatGPTからダウンロードされたZIPと全件バイト一致しました。初版の生成モデルは未確認です。`clip_012`・`clip_015`・`clip_026` の再注釈は、Codex CLIの実行ログと公開JSONのSHA-256を照合し、`gpt-6.2-sol / max` と確認しました。
- **画面外座標は4,136 / 184,912件（2.24%）。** うち4,125件は推定座標です。実際の画面外・遮蔽・識別不能の割合は、視覚確認用サンプルを集計して報告します。
- **短時間検証は通過。** 既存pytest 16件、i935のstrict復元とCPU推論、Meijiの実映像1 clipでの計測を確認しました。

ブランチ `codex/parent-2627-issue-2637-primary`、HEAD `365d15e92cc848479d3417a33c295a81fb31e781`。PRは未作成です。CPU結果を受けたら球像の目視確認を行い、固定した144評価フレームでGPUの時系列比較を共有queueに投入します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T113758Z_1100cbe90df9_ball-paper-shape","20261126T113758Z_b215433fb167_ball-paper-footprint"]}}