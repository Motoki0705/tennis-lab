#2633 の実験設計を記録し、B00〜B03 の描画プールの検証と学習用データの固定をローカルジョブで開始しました。 [実験設計](https://github.com/Motoki0705/tennis-lab/issues/2633#issuecomment-7190709855)

事前集計では両条件とも各シーン128枚、計512枚を確保できました。分布の広がりを示す実効区画数は、狭分布から広分布で方位 **7.33→7.83**、高さ **3.39→4.73**、水平画角 **1.67→2.88** でした。テストは7件通過しました。

準備コードは `codex/parent-2627-issue-2633-primary` の `4031c6cb97bee606e9e435d5b490e7c30ab3aa34` にコミット済みです。準備結果を照合した後、4本の学習・評価を共有GPUキューに投入します。主張の判定と記録PRは実測後です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T113214Z_18ccbb4113fa_i2633-viewpoint-inputs"]}}