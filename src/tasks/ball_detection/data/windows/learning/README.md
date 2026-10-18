# 増分公開からの座標学習入力

`training_partition(layout, snapshot)`は公開snapshotのtrain所属をcanonical source/clip IDで返す。
ball storeのsplitとfile/semantic hashを照合し、val/testの再割当ては行わない。
共通学習管理は、この所属と固定holdoutの所属を比較してsource単位の混入を拒否する。

`prepare_training_windows(layout, publication, policy, directory)`は、公開receiptと累積snapshot、
requestのplanが固定したbase snapshotを照合し、baseからの追加・再利用clipの差分を検証する。
旧clipの注釈・画像identityを保持したままMDD-only manifestを作る。
`WindowPolicy`にプレイ区間・教師数・FPSの条件を明示する。poseの生成・読込みは不要。
manifestはtrainだけを含み、画像は元のimmutable storeを参照する。

返る`coverage.json`は旧/追加train clip数、clipごとのnative frame数・座標教師数・FPS別窓数、
除外状態を保存する。追加clipから利用可能な32frame窓が一つも得られなければ
`no_additional_coordinate_teachers`となり、学習manifestを作らない。
`require_additional_teachers`はこの状態を拒否する。旧データだけで次の追加学習を実行しない。
確認済み不在だけのclipはヒートマップ教師として公開できても、座標回帰の窓にはならない。
その事実を記録し、位置教師の捏造・文脈への延長・frame反復は行わない。

`validate_training_windows`は保存物をhash照合し、同一policyからの再計算と一致することを確認する。
不足した保存物を修復したり、入力変更を同じrunへ取り込んだりしない。

`freeze_holdout`は既存の固定window manifestを`HoldoutWindows`に結び、`DatasetPartition`を返す。
snapshotの全file hash・clip identityと、宣言したsamplingから再構築したwindow membershipの一致を要求する。
従来のvalidationには既存Common subsetが必要。testは別partitionとして保持する。
新しい累積snapshotを公開しても、元のholdout snapshot/window参照は変わらない。

`BallEvaluationRecipe`をplanに固定すると、公開後の`resolve_evaluation`が新旧の
validation/testを確定する。既存のGT・画素・split・窓membershipを照合したうえで、
累積storeを参照するmanifestへ`existing`/`new`の集合を付ける。元のholdout参照も保持する。
新規側のCommon subsetは要求しない。両集合には各FPSの教師が必要で、不足時に
別集合の平均へ置き換えない。登録後のvalidation/test manifest欠損・改変は、
共通Learningの`register`による再登録時も保存済み参照から検証し、再生成せずエラーとする。

これらはtask用APIであり、モデル推論・学習・モデル採用を実行しない。
production CLIでは資源管理下の工程から呼び、共通学習管理には検証した参照を渡す。
