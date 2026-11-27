#2633の判定は、事前基準で**部分的支持**です。seed 43では広分布でKP誤差が下がりました。2 seed平均では、広分布でKP誤差が増え、ホモグラフィ成立率が下がりました。

| `real-video-v1` | 枚数 | KP平均誤差（長辺512換算px）狭→広 | H成立率 狭→広 |
|---|---:|---:|---:|
| val 全体 | 4,468 | 102.61 → 111.65 | 32.30% → 23.03% |
| test 全体 | 4,372 | 103.50 → 111.07 | 32.41% → 22.19% |
| test 低角・斜め向き・見切れの和集合 | 439 | 106.02 → 114.70 | 30.98% → 20.05% |

GTへの平均再投影誤差が画像対角長の1%以内だったHは、test全体で7/8,744件から1/8,744件に減りました。横向き45°以上に該当する画像は0枚でした。TCDの全498動画をval/testに割り当てたため、TCDのみの学習対照は省きました。 [視点別の表と図](https://github.com/Motoki0705/tennis-lab/blob/c66d8b89ac4a80d6614865a4d4cac59869846806/knowledge/nodes/court_detection/000084-group-i2633-court-viewpoint-diversity.md)

両条件とも各512枚で、2条件×2 seedの各6,000更新と採点を約5時間49分で完了しました。テスト14件が通過し、再集計したCSV・JSON・図はコミット済みの成果物とバイト単位で一致しました。

記録専用PR [#2671](https://github.com/Motoki0705/tennis-lab/pull/2671) のHEADは `c66d8b89ac4a80d6614865a4d4cac59869846806` です。Knowledge CIが実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2671,"head":"c66d8b89ac4a80d6614865a4d4cac59869846806","resume_on_success":false}}