旧IDを伏せた条件で `gpt-6.1-terra` / `xhigh` を実行し、69 clip・1,205 frame・34,664 boxを全体用の固定datasetに保存しました。全69件が初回の形式検査を通過し、不採用・未処理は0 frameです。 [固定dataset]([local path])

| 層 | 公開 frame | 人の確認対象 |
| --- | ---: | ---: |
| 選別外れ・コート近く | 399 | 30 |
| 1視点・3Dなし | 71 | 28 |
| 未割当・再投影不一致 | 187 | 30 |
| ID switch・断片化 | 350 | 30 |
| 停止 clip | 66 | 22 |
| 確信層から無作為 | 144 | 30 |
| **合計** | **1,205** | **170** |

最大4並列で経過28分21秒、CLI稼働は4.94秒[local path]。入力3,713,178 token、出力300,823 tokenを記録しました。試行時の単価で **USD 4.21相当（USD 0.00350/frame）**です。実請求額は未取得で、利用枠の通知は0件でした。 [実行記録](https://github.com/Motoki0705/tennis-lab/blob/c1c9b06ab3ebf36b595066bc34377e85c561fd8b/knowledge/runs/run-i2442-meiji-pseudo-association-full/README.md)

全69 clipの公開ラベルを固定計画と照合し、seed `2442` の抜き取り170 frameをreview APIで取得できることを確認しました。新HEADでpytest 45件、Ruff・mypy、knowledge検査が通過しました。残作業は170 frameの人手確認と層別誤り率の記録です。 [抜き取りリストと手順](https://github.com/Motoki0705/tennis-lab/blob/c1c9b06ab3ebf36b595066bc34377e85c561fd8b/knowledge/runs/run-i2442-meiji-pseudo-association-full/human-review.md)

`codex/parent-1739-issue-2442-primary` に `c1c9b06ab3ebf36b595066bc34377e85c561fd8b` をpushしました。`Closes #2442` と親の注記を維持して本文を更新し、新HEADのCI確認を親に引き渡します。 [PR #2448](https://github.com/Motoki0705/tennis-lab/pull/2448)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2448,"head":"c1c9b06ab3ebf36b595066bc34377e85c561fd8b","resume_on_success":false}}