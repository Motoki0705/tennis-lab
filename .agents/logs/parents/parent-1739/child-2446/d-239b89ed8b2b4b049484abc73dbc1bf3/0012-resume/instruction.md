ローカルジョブ `20261126T145128Z_d85353459e97_i2446-review-r1-123-check` は完了した（1,938 passed、staged tree `89c74e45` と、順に合成したtree `56130c62` を #2462 `497195ff` に対して照合済み）。結果は `[local path]` の同名ディレクトリの `completion.json` と `output.log` にある。

注意: 前回の PARENT_RESULT の job_id は `d893534e97` と誤記されていた（正しくは `d85353459e97`）。そのため親CLIがエラーにした。今後、PARENT_RESULT の job_id は正確に写す。

結果を確認し、pre-commit と knowledge 検査も成功していれば、コミット・push・PR本文の更新（指摘1〜5の対処の要点を含める）を行う。失敗があれば直してから進める。新HEAD（完全SHA）を報告し、CI待ちで終了する。
