ユーザー判断に沿ってSapiens2-1B/5Bの全6条件をGPU時間の制約で対象外とし、精度表を主比較15条件・補助3条件に更新しました。`codex/parent-2492-issue-2490-primary` の `bcd10180deaf7011684b9b4754ae998d312c2d14` をpush済みです。 [ドラフトPR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500)

1B PT32は担当親が約8時間で取り消し、終了値143と空の予測出力先を確認しました。担当親が観測した進捗は約34%、約3.2秒/人物で、全コホートの所要時間は約23時間の見積もりです。5B PT32のCUDAロード記録と、両サイズの初回TensorRT変換が `ReduceMean` の軸入力で停止したログも保存しました。 [実行記録](https://github.com/Motoki0705/tennis-lab/blob/bcd10180deaf7011684b9b4754ae998d312c2d14/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2_attempts.json)

既存pytest 69件とknowledge検査が通過しました。新HEADのCIを待ちます。残る速度・VRAM・Pareto比較は、#2497の修正版公開と親の再投入指示が揃ってから進めます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"bcd10180deaf7011684b9b4754ae998d312c2d14","resume_on_success":true,"next_step":"#2497の修正版と親の再投入指示を照合し、残る候補の共通速度計測・VRAM・parityを進める。計測結果からPareto比較と採用候補をまとめる。"}}