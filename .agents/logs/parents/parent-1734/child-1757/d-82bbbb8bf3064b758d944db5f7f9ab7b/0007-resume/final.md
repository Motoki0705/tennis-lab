条件3と同一HEADの逐次対照を記録し、Draft PRを更新しました。計測HEADは `fb6255a2`、記録HEADは `27c26f536` です。 [PR #2273](https://github.com/Motoki0705/tennis-lab/pull/2273)

旧逐次とのボール3D有効性の9フレーム差は、commitを変えた新しい逐次でも再現しました。新しい逐次と条件2の初回は、全141 nodeとscene配列が完全一致しました。ボールの数値差はheatmapと候補から現れ、差を生じさせた個別の変更は未特定です。 [逐次対照の記録](https://github.com/Motoki0705/tennis-lab/blob/27c26f5364f664f38dc2a4f90795986718c447de/knowledge/runs/run-i1757-serial-fb6255a-first-1/result.md)

| #1757の受入条件 | 判定 | 検証結果 |
| --- | --- | --- |
| 3条件の比較と、品質を維持した時間短縮 | 未達・実測完了 | 条件3は同一HEADの逐次より初回6.76%短縮しましたが、warm中央値は1874.99秒で、条件2より72.23秒遅くなりました。DINOのforwardは約38秒短縮し、dispatch・モデル読込・帰属先未計測の区間が増えました。充填率は99.90%以上でした。 |
| 固定許容差と離散値の確認 | 未達 | 同一HEADの条件3でボール3Dの有効性が6フレーム、人物3Dの有効性が6関節で変わりました。ボールを外し、保存済みの実GPU出力からCPUで再組立てしても、共通有効箇所の人物root最大2.35 cm・関節最大42.72 cmの差が残りました。 |
| scene再読込、失敗例、差分動画 | 一部達成 | 追加の12 sceneを再読込し、court_sideは全件 `[false,false,true,true]`。空検出と3D棄却を含む差分動画6本を確認しました。clip全体の再構成失敗は未検証です。 |
| READMEとknowledgeへの記録 | 達成 | 両READMEから、計測結果、再現手順、固定許容差を参照できます。 |
| 実GPUの中断・再開とcache混在 | 未検証 | 今回の追加GPU計測は逐次対照の初回1回です。 |

逐次job `0-8204898262595246886_2026-11-19T15-22-17-404753113Z_3013697_i1757-serial-fb6255a-first-1` は01:03:55 JSTに終了コード0で完走し、queueの `done` と `measurement/terminal.json` の `complete` を照合しました。 [queueログ]([local path]) [差分動画・clip_027](https://github.com/Motoki0705/tennis-lab/blob/27c26f5364f664f38dc2a4f90795986718c447de/outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/serial-fb6255a-first-1/review/to-batched-clip_027.mp4) [固定許容差](https://github.com/Motoki0705/tennis-lab/blob/27c26f5364f664f38dc2a4f90795986718c447de/knowledge/runs/run-i1757-video-batch-meiji/tolerances.json)

既存CPUテスト159件（98.15秒）、Ruff、knowledge検査、記録HEADのPRチェック17件が成功しました。pre-commitは対象ファイルなしでした。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44563330047) [Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/44563330051)