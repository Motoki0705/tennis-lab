# cross-1: coverage増減点の誤差を分ける集計のテスト設計

方式・設定・基盤scoringは変更しない。保存済みv2のbaselineとtop-Kの3 LOVO foldを
`load_run`で読み、SHA-256をpinしたbundle/annotationsに対して基盤`error_samples`を呼ぶ。
分布統計も基盤`distribution(threshold=20)`を使い、独自の再投影・採点は作らない。
比較専用scriptと小さな記録はこのrunへ置き、実データの新しい出力はcohort_v1へ保存する。

| 守る契約 / 検出する故障 | 層・再利用/追加するnode | 予算 |
|---|---|---|
| 共通/追加/喪失をheld-out cameraとframeの組で分割し、行順やframeだけでjoinする誤りを拒否 | `tests/unit/tennis_scene/evaluation/ball3d/test_topk_cohorts.py`の非対称fixtureで追加。504/505/1009、同frameの別camera、observed/estimated、20pxちょうどを含む | 1case、CPU、実資産不要、call 1秒以内目標 |
| 区間・camera・reference kindの分解を保持し、空集合を0pxの良い結果として出さない | 上と同じfixture/契約nodeで集計の出力を検証 | fixtureは小さい固定配列のみ |
| 重複camera/frameや共通点のreference kind不一致で停止する | 入力の異常分類ごとにunit 1caseずつ | 各1秒以内目標 |
| 共通+追加=方式全点、共通+喪失=baseline全点を検算し、欠落したpartitionで停止する | 検算の2つの不一致をunitでparameterize | 2case、直積は作らない |
| 基盤scorerのbundle identity、held-out、参照kindの意味を維持 | 既存`tests/unit/tennis_scene/evaluation/ball3d/test_scoring.py`を再利用 | 保存方式の再実行なし、pytest -n 0 |

実データでは既存`paired-scores.json`の全点・2方式共通点のcount/median/p90/>20率とも
区間別に照合する。新JSON/表とcamera-frame別の採点CSVを保存し、既存成果物のchecksum不変を確認する。
pytestは`--collect-only`と`--durations=0 --durations-min=0`を記録する。
ruff/mypyとknowledge validatorを通し、push後の全CI成功を確認する。
代表的な故障として共通点の方式誤差をbaseline誤差へ取り違える変更を一時注入し、
非対称fixtureの具体的な中央値比較で失敗すること、元bytesへ戻すと成功することを記録する。
