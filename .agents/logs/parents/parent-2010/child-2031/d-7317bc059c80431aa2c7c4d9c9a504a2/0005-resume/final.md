2件とも成立を確認し、修正しました。

- **pr-2056-1**：ハンドラは停止フラグだけを設定し、registry外でrunnerを通常停止・回収してから中断を記録します。
- **pr-2056-2**：最後のGitHub照合後にjob/run・委任・generationを再確認し、以後は外部I/Oを挟まず保存します。acceptも修正しました。

追加回帰は6ケースです。最新mainへrebaseし、main追加テストのclientを同じStoreへ結び付ける修正も含めました。

成果は [PR #2056](https://github.com/Motoki0705/tennis-lab/pull/2056)、branch `codex/dispatcher-triage-io`、HEAD **`8e892cae2bf041937a079ce4c2239e3830323323`**。push済み、worktreeはcleanです。

| 実行コマンド・対象 | 結果 |
|---|---|
| `.venv/bin/python -m pytest -q -n 4 --no-cov`：関連一式 | **344 passed**、297.26秒 |
| 同pytestの追加6ケース集中実行 | **6 passed** |
| pytest `-n 3`：共通runnerの子孫回収 | **3 passed** |
| pre-commit、ruff check/format、mypy（本体21ファイル） | **成功** |

対象HEADの[CI / Pythonはsuccess](https://github.com/Motoki0705/tennis-lab/actions/runs/43666879328/job/132672344341)。PR本文を更新し、[コマンド全文・実測](https://github.com/Motoki0705/tennis-lab/issues/2031#issuecomment-6887994992)を残しました。

実機通しは未実施です。統合レビュー・merge・merge後のmain CI確認は親担当へ引き継ぎます。