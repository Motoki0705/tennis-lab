#2636 の停止14 clipを対象に、同期4視点の元画と DINO box・追跡 ID・選別 group の重畳画像を生成する CPU ジョブを起動しました。試行した clip_012・038 では21同期フレーム（84カメラフレーム）を抽出し、関連する既存テスト3件が通過しました。 [検証設計](https://github.com/Motoki0705/tennis-lab/issues/2636#issuecomment-7190567683)

成果物は `codex/parent-2627-issue-2636-primary` の作業ツリーにあります。ジョブの結果を受けて全件を目視し、確認した誤りの CPU 再生と、論文の主張に対する判定を記録PRにまとめます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T112440Z_a93481dc77bb_meiji-stopped-clip-evidence"]}}