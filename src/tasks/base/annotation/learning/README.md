# 公開後の学習とモデル採用

`Learning` は共通の状態管理を担当し、`LearningAdapter` がtask固有のdataset、
checkpoint、recipe、評価指標と合否判定を担当する。baseからball/playerをimportしない。
具体taskのadapterとproduction CLIへの接続は各taskとcomposition rootが所有する。

`register(completion, adapter)` はplanにあるすべてのtask/productを照合する。
ball、player bbox、player poseのいずれかが未公開・失敗・保留なら学習を始めない。
理由付き対象外は公開済みとは別の完了状態である。adapterの学習対象productに
追加教師がなければ `no_additional_teachers`、学習設定がなければ `not_configured` を保存する。
公開clipがあっても、adapterの`validate_intent`が`NoAdditionalTeachers`を返した場合は理由付きで同じ終端結果を保存し、学習intentやqueue jobを作らない。

追加教師がある場合は、公開receipt、累積train snapshot、前採用モデルの世代と重み、
recipe、固定validation/test membership、採用条件を `LearningIntent` に固定する。
snapshotの内容とmembershipはadapterが検査し、baseが元動画・clipのsplit間重複を拒否する。
評価ジョブの `EvaluationReceipt.validation` は固定validationに一致する必要があり、
testを代わりに使うことはできない。test membershipは学習との分離検査に使う。

`tick(intent, adapter, runtime)` は学習→評価→採用を進め、一回ごとの状態を返す。

- 学習・評価コマンドは `StageRunner` とmainの共有training queueを通る。scope、queueの
  script hash・owner、工程別の資源・timeoutも不変保存する。同じ公開receiptは一つの
  intent、同じ工程試行は一つのlaunch hashで登録し、受付後の親停止でジョブを増やさない。
- taskは `StageInput` とsupervisorの入力digestを照合し、出力先へ `TrainingReceipt` または
  `EvaluationReceipt` を書く。成功した実行receiptとtaskの内容検証が揃ってからbaseが
  `CompletedStage` を保存する。checkpointがあるだけでは学習完了にならない。
- 学習失敗・評価失敗は `failed`、不合格は `rejected`、採用条件なしは `not_configured`。
  いずれも現行モデルを維持する。条件未設定でも学習と固定validationの結果は保存できる。
  失敗の保存前には実行状態を照合する。queue観測中の移動は進行中として再確認し、
  `interrupted` など終端を確認できない状態は明示エラーにして失敗を確定しない。
  過去の `failure.json` があっても同じlaunchを再照合し、成功receiptとtaskの内容検証が
  揃えば完了を回収する。失敗記録は履歴として残し、検証済み完了を優先する。
- 自動再試行はしない。`retry(..., reason=..., resume_checkpoint=...)` で理由と新しい試行番号を
  保存する。checkpoint再開はadapterが同じintent・recipe・初期値のものか検査する。
  別snapshotの追加学習と、同一runのoptimizer/schedule再開を混同しない。
  次試行ディレクトリの作成後、再試行情報の保存前に停止した場合も、明示的な`retry()`で
  直前の失敗を照合し、同じ試行番号へ復旧する。`tick()`だけでは再試行を開始しない。
  新規・復旧のどちらも旧launchの終了と完了receiptを再照合し、待機中・実行中・
  lease解放待ちには次試行を作らない。旧試行が成功していればその完了を保存して
  retryを拒否し、`tick()`で後続工程へ進める。照合のために旧ジョブを再投入しない。
  次試行のrequest保存前なら、途中までのretry記録を残したまま旧試行の検証済み完了を
  `tick()`でも回収する。新しいrequestが確定済みの試行を巻き戻す処理は行わない。

`ModelRegistry.bootstrap(model, evidence)` で初期モデルを根拠付きで明示登録する。
最終採用は、task判定、成功した学習・評価、固定validation・policyの参照を再検証してから行う。
現行の `ModelState`（モデル全参照と世代）が評価基準と一致するときだけ、新世代へ切り替える。
比較元が変わっていれば `reevaluation_required` を保存し、自動で評価基準を読み替えない。

現行pointerと採用結果の全文はSQLiteの一つのtransactionでcommitする。
`ModelRegistry.select(verdict, adapter)`は、共通の採否根拠に加えて必須の
`LearningAdapter.retained_references(layout, intent, evaluation)`が列挙するtask固有の
固定入力・学習来歴・評価記録の依存参照を同じtransactionで保護する。不採用・再評価要求・
保存済み判断の再送でも列挙し、参照欠落や保護失敗は判断とpointer更新をrollbackする。
直後にreceiptファイルの保存が失敗しても、同じDB行から復元でき、採用を二重実行しない。
進行中campaignのplanを書き換える処理や既存clipを再推論する呼び出しは持たない。
次campaignの計画作成時に `current(task)` を読み、採用版を新planへ固定する。

保存先は `campaigns/<campaign>/learning/<task>/` のintent/runtime、工程試行・出力検証receipt、
判定結果と、`models/<task>/selections/` の採用履歴。学習本体のcheckpoint保存先は
[タスク出力規約](../../../OUTPUTS.md)に従い、receiptからhash付きで参照する。

単体試験は小さな合成ファイルと永続化したqueue代替を使い、公開後・queue受付後・
checkpoint後・評価後・モデル切替途中の停止、二重起動、split混入、競合採用を検証する。
合成試験の2世代更新は、実モデルの2サイクル学習やGPU実測を意味しない。
