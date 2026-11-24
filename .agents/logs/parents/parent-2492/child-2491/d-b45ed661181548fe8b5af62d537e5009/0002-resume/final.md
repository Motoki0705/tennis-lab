FastHMRの公式checkoutと追加Python依存を準備し、公式推論モジュール6つのCPU読み込みを確認しました。`J_regressor_h36m.npy` は取得し、形状とSHA-256を照合済みです。残る公開重み4ファイルは、取得後にサイズとSHA-256を検証するローカルジョブを起動しました。 [公式重み](https://huggingface.co/SoroushMehraban/FastHMR/tree/898332e1059fc0579f986e03cab11471d4e27b65)

前回CIの設定クラスの既定値エラーを修正し、ローカルで76件成功、1件skip、Ruff・mypy通過を確認しました。依存2ブランチは最新HEADを取り込み済みです。今回の差分は公開資産の検証結果を受け取ってから、既存の [Draft PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501) にpushします。

CameraHMRのライセンス付き3ファイルと #2496 の固定速度入力・基準値は引き続き待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T124209Z_8a4890ab19eb_fasthmr-public-assets"]}}