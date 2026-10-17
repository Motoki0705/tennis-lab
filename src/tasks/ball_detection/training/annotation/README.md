# 注釈公開後のball追加学習

`BallLearningAdapter`は[共通Learning](../../../base/annotation/learning/README.md)へ渡すtask実装。
公開snapshotのtrain所属、固定holdout、採用MDD全重みを検証してから、学習・比較評価を別工程として実行する。
production controllerへの接続は[ワークフロー](../../../annotation/README.md)を参照する。CPU合成試験は実動画受入の代替ではない。

```mermaid
flowchart LR
    P[公開receiptと累積snapshot] --> I[固定LearningIntent]
    M[前採用MDD全重み] --> I
    R[合意したrecipeと元のholdout] --> H[公開後に新旧holdout確定]
    P --> H
    H --> I
    I --> W[MDD-only窓と旧・追加clipのcoverage]
    W --> T[共有queue / 累積train・指定比率のreplay]
    T --> C[完全checkpointと完了根拠]
    C --> E[共有queue / 固定評価でbaseline比較]
    E --> V[taskの採否判定]
    V --> A[共通ModelRegistryの世代照合]
```

## 固定する入力

`BallLearningRecipe`にsampling、augmentation参照、runtime、epoch/update予算と学習率等を明示する。
数値の学習予算や採用条件の既定値は置かない。32frame・upfront入力照合等の構造条件はschemaで固定する。
`FixedEvaluation`のvalidation/testは[holdout binding](../../data/windows/learning/README.md)で検証する。
`evaluation_recipe`がある場合、旧・新validationの平均を等重みでまとめてcheckpointを選ぶ。
学習はtest画像を読み込まない。選んだ一候補だけを旧・新testで現行モデルと比較する。
recipeがない既存経路はclean Common validationによる選択・比較を維持する。

`BallLearningRecipe.replay`は前モデルの学習manifest参照と未学習分の抽出比率を固定する。
そのmanifestのhashを初期checkpointと照合し、GT・画像identityが同じclipを学習済み群とする。
全train窓を候補に残し、FPSごとの均等枠を群別に配分する。各枠の整数quotaを満たさない
epoch予算や空の群は拒否する。同じ群内では全候補を一巡してから再抽選し、
各epochの候補数・抽出数・unique窓数を保存する。比率は入力窓数の割合であり、
全データを各epochに一度ずつ読む保証ではない。学習履歴と今回公開されたかどうかは区別する。

採否時の`retained_references`はvalidation/test双方の`HoldoutWindows`を開き、実際の
window manifestとsnapshot参照を列挙する。`BallLearningRecipe`、`TrainingWindows`、
`BallTrainingProvenance`、`BallEvaluationMetrics`の型付き参照も明示的に収集する。
これらは[共通の採否transaction](../../../base/annotation/learning/README.md)で保護される。
回帰テストは実際のHoldoutWindowsと完了記録を使い、依存を個別に期限切れ登録してGC候補に
ならないことを検証する。入力・来歴schemaに参照を持つ文書型が増えた場合も検証対象への追加を要求する。

登録時に窓とcoverageを生成し、登録後は保存した入力が欠けていれば停止する。
追加clipに座標教師を含む窓が無い場合はcoverageを保存し、共通管理へ`NoAdditionalTeachers`を返す。公開を保持した終端結果となり、旧clipだけで追加学習しない。
初期モデル・recipe・累積snapshot・intentのdigestを通常のfull-model学習recipeへ組み込む。
新runは全重みを転送し、optimizer/scheduleは初期化する。

## 工程と再開

共通管理が返すコマンドは`python -m src.tasks.ball_detection.training.annotation`。
`--input`、`--input-sha256`、`--output`を受け取り、supervisorの入力digestと
`campaigns/<campaign>/learning/ball/<train|evaluate>/<attempt>/`の所有範囲を照合する。
CLIはproduction固定rootを使う。自身でqueue投入やモデル採用を行わない。

重みは`outputs/ball_detection/train/annotation/<intent-sha256>/attempt-<番号>/`へ保存する。
最終epochと更新回数、bestの指定validation指標、親全重みhash、config、source hash、
候補specとcheckpointの一致を確認して`TrainingReceipt`を返す。
stressを指定したrecipeでは3 profileの保存結果も必要となる。

中断したepochは、共通管理の理由付きretryで同intent・同recipeの最新完了checkpointを明示する。
途中runはそのoptimizer/scheduleを再開し、既に完了したrunは追加更新せずreceiptを回収する。
完成したreceiptの再回収も同じ参照を返す。別snapshotへの入力差替えや自動retryは行わない。
最終checkpointだけがあり`COMPLETED.json`が無い場合も、同じ明示retryで再開できる。
重み・optimizer・bestを照合して更新を追加せず、指定したstress評価と完了記録の保存を実行する。

## 比較と採否

同じ固定評価データ・runtime・モデル構造でbaseline/candidateを比較する。
全体・Common・各source、設定時は既存・新規について、採点FPS/clip/frame集合hash、支持数、有限な誤差と
macro平均の整合性を検証する。実checkpointのstate hash、学習manifest hash、
成功した学習工程との対応も`EvaluationReceipt`受理前に確認する。

`BallSelectionPolicy`はCommonの最小支持数・最大誤差・必要改善量、Fullの許容悪化量、
任意のsource別支持数・許容悪化量を持つ。source条件はFullのsource別macroを使う。
必要なsource/FPSが欠けていれば不合格。条件はすべて満たす必要があり、境界値は含める。
`BalancedSelectionPolicy`は最終testの既存・新規を50:50で集計し、平均誤差の厳密な改善と
既存側の非悪化を要求する。同値の平均は改善と扱わず、クリップ数の多い側で重みを増やさない。
各集合内はFPS別平均の等重み平均とし、元動画pixel座標の誤差を用いる。
policy未設定なら共通管理が学習・評価を保存したうえで`not_configured`を返し、現行モデルを保持する。
合格後の世代照合とpointer切替は共通ModelRegistryの責務。
