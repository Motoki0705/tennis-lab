**第6章の主張は不支持です。** 停止14 clipに共通する原因を2D検出に帰属できませんでした。基本42時点で判定可能だった665選手・視点すべてにDINO boxと追跡IDがあり、581件が選別後に残りました。004・028では、人物が重なって独立boxが途切れ、その前後で追跡IDが入れ替わりました。 [clip別の所見](https://github.com/Motoki0705/tennis-lab/blob/2f206770ea2ac42aa1e6089e2c504dcec1976bea/knowledge/runs/run-meiji-stopped-clip-observations/README.md)

CPUで元の停止理由を14/14件再現しました。確認した検出行17件の除去で復帰したのは0/8件、同一選手groupの統合では0/2件でした。次点groupを外した感度比較では7/9件が通過し、確認時点で4人のIDが一貫して各人2視点以上に残ったのは5件でした。先行比較では、同じ検出・追跡で選別幅を変えると012・020・058が三角測量まで復帰しています。新しいboxに対応するpose・外観を補った場合の復帰は未計測です。 [CPU再生結果](https://github.com/Motoki0705/tennis-lab/blob/2f206770ea2ac42aa1e6089e2c504dcec1976bea/knowledge/runs/run-meiji-stopped-clip-observations/replay.json)

記録専用 [PR #2656](https://github.com/Motoki0705/tennis-lab/pull/2656) に観測記録と論文用の4視点PDF14図を収録しました。既存テスト3件とknowledge検査が通過し、全14図の再生成一致を確認済みです。HEADは `2f206770ea2ac42aa1e6089e2c504dcec1976bea`、CIは実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2656,"head":"2f206770ea2ac42aa1e6089e2c504dcec1976bea","resume_on_success":false}}