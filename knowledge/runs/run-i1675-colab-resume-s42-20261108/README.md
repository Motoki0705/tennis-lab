# Colab CLIによる容量比較の再開

対象は固定campaign `s42-20261103-main`。入力・モデル・学習条件は[元の実験](../run-ball-capacity-scaling-20261103/README.md)を参照する。

[復元証拠](recovery.json)は全datasetの独立hash検証、4段のcheckpoint/optimizer/RNG/sample位置/log prefix、固定sourceとruntimeの照合を記録する。実Colab session/jobとqueue ID・実行commandも同JSONに保持する。source bundleのbaseと運用adapterのcommitを区別する。

起動は共通runnerの有限Codex子が担当し、親がNotebook kernel配下に置いた外部queue serviceへresource allで一つだけ登録した。Colab workerはrunner-outputを宣言し、学習中のcampaign treeはartifact storeが保存する。

[6,000更新のDrive保存](checkpoint6000-drive-verification.json)はreceipt・marker・全9 payloadと再開状態の独立復元検証を完了した。全8段52k更新、残る容量条件の比較と最終集計は継続中。checkpointと予測配列はDriveに保持する。
