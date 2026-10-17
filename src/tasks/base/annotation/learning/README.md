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

`LearningReferences` はintent登録と同じtransactionで入力を保護する。
`LearningAdapter.input_references()` はsnapshot内のindex・metadata・shardや固定評価の
実ファイルを型付き契約から列挙する。queue待機中にも保護し、同じintentの再登録で
対象を変更することはできない。未登録・解除済みの保護は明示的なエラーになる。

`tick(intent, adapter, runtime)` は学習→評価→採用を進め、一回ごとの状態を返す。

- 学習・評価コマンドは `StageRunner` とmainの共有training queueを通る。scope、queueの
  script hash・owner、工程別の資源・timeoutも不変保存する。同じ公開receiptは一つの
  intent、同じ工程試行は一つのlaunch hashで登録し、受付後の親停止でジョブを増やさない。
- taskは `StageInput` とsupervisorの入力digestを照合し、出力先へ `TrainingReceipt` または
  `EvaluationReceipt` を書く。成功した実行receiptとtaskの内容検証が揃ってからbaseが
  `CompletedStage` を保存する。checkpointがあるだけでは学習完了にならない。
- 学習失敗・評価失敗は `failed`、不合格は `rejected`、採用条件なしは `not_configured`。
  いずれも現行モデルを維持する。条件未設定でも学習と固定validationの結果は保存できる。
- 自動再試行はしない。`retry(..., reason=..., resume_checkpoint=...)` で理由と新しい試行番号を
  保存する。checkpoint再開はadapterが同じintent・recipe・初期値のものか検査する。
  別snapshotの追加学習と、同一runのoptimizer/schedule再開を混同しない。
  次試行ディレクトリの作成後、再試行情報の保存前に停止した場合も、明示的な`retry()`で
  直前の失敗を照合し、同じ試行番号へ復旧する。`tick()`だけでは再試行を開始しない。

工程要求はqueue投入前に固定入力を保持する。`LearningAdapter.resume_references()` は
同じintentの再開checkpointを検証し、その依存を列挙する。retryはその依存を
`Retry.resume_dependencies` に不変保存し、checkpointとともに返却前に保護する。
retry記録やrequestの保存後（保護登録の失敗を含む）に中断した場合は、同じ引数の
明示的な`retry()`で復旧する。理由・再開checkpoint、直前の失敗とそのrequest、
保存済みretry/requestの内容を照合し、一致すれば保存済みの依存を保護登録して
同じrequestを返す。照合が合わなければ明示的に拒否し、試行番号を増やさない。
成功した工程は `LearningAdapter.result_references()` が検査済みcheckpoint・来歴・
評価根拠の実ファイルを列挙し、結果の保護記録と同時に確定する。完了ファイルの保存後に
保護登録が失敗しても、再送では保存結果を再検証して登録し、学習を重複起動しない。
評価の起動には、学習結果の保護済み依存も渡す。
これらは失敗・完了後も保持する。自動解除、過去intentの一括移行、未完了runの
全checkpoint収集は含まない。[保存台帳](../storage/README.md)のGCは引き続きdry-runである。

`ModelRegistry.bootstrap(model, evidence)` で初期モデルを根拠付きで明示登録する。
最終採用は、task判定、成功した学習・評価、固定validation・policyの参照を再検証してから行う。
現行の `ModelState`（モデル全参照と世代）が評価基準と一致するときだけ、新世代へ切り替える。
比較元が変わっていれば `reevaluation_required` を保存し、自動で評価基準を読み替えない。

現行pointerと採用結果の全文はSQLiteの一つのtransactionでcommitする。
直後にreceiptファイルの保存が失敗しても、同じDB行から復元でき、採用を二重実行しない。
進行中campaignのplanを書き換える処理や既存clipを再推論する呼び出しは持たない。
次campaignの計画作成時に `current(task)` を読み、採用版を新planへ固定する。

保存先は `campaigns/<campaign>/learning/<task>/` のintent/runtime、工程試行・出力検証receipt、
判定結果と、`models/<task>/selections/` の採用履歴。学習本体のcheckpoint保存先は
[タスク出力規約](../../../OUTPUTS.md)に従い、receiptからhash付きで参照する。

単体試験は小さな合成ファイルと永続化したqueue代替を使い、公開後・queue受付後・
checkpoint後・評価後・モデル切替途中の停止、二重起動、split混入、競合採用を検証する。
合成試験の2世代更新は、実モデルの2サイクル学習やGPU実測を意味しない。
