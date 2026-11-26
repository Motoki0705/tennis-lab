meiji_4cam の段階1比較ページを作成しました。旧経路が停止した14本と、成功55本から seed 42 で抽出した `clip_009`・`clip_052` を収録しています。4視点とコート上面を同期再生し、全16本で #2458 の3D球を両側に表示します。 [比較ページ](https://github.com/Motoki0705/tennis-lab/blob/a9f3f50f66a2d085a4f6695a0f580d3c39f2b78e/outputs/pose-comparison/index.html) · [入力の版とclip別指標](https://github.com/Motoki0705/tennis-lab/blob/a9f3f50f66a2d085a4f6695a0f580d3c39f2b78e/outputs/pose-comparison/manifest.json)

新経路は全12,855フレームで骨格候補を出しました。`clip_012` は旧経路がコート向きの判定で停止し、新経路では f339 の手前と奥の人物に骨格が重なります。一方、`clip_020` は189/651フレームでコア内が3体、`clip_049` は506/1,120フレームで5体以上でした。16本それぞれの所見はPRに記録しました。 [PR #2461](https://github.com/Motoki0705/tennis-lab/pull/2461)

全16本のH.264デコード、Chromeでの再生・切替・ライト／ダーク表示を確認しました。全体31.96 MB、最大4.19 MBです。関連テスト41件、設定監査396件、pre-commit、knowledge検査が通りました。

未使用視点への再投影誤差は未計測です。Artifact公開と、#2460 の採用シーン更新後の再生成は親へ引き継ぎます。

ブランチ `codex/parent-1739-issue-2459-primary`  
HEAD `a9f3f50f66a2d085a4f6695a0f580d3c39f2b78e`  
[CI実行中](https://github.com/Motoki0705/tennis-lab/actions/runs/45326506743)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2461,"head":"a9f3f50f66a2d085a4f6695a0f580d3c39f2b78e","resume_on_success":false}}