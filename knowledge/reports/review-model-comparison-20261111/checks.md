# 実行後の検証記録

2026-11-11。実モデル14回の終了・cleanup確認後に実行した。
比較中にpytestを並行実行していない。今回の差分は実測記録、文書、topic登録と
公開済み集計JSONに限ったignore例外であり、harnessの実行コード・固定入力・正解表・protocolは変更していない。

| 検証 | 実行と結果 |
|---|---|
| 固定snapshot | `PYTHONPATH=.agents/ops .venv/bin/python -m review.evaluation verify --repo .`。7ケースを照合、AI起動0回 |
| オフラインの契約 | `.venv/bin/python -m pytest -n 0 tests/unit/agents_ops/test_review_evaluation.py`。19 passed、2.72秒。将来資料の排除、未登録endpoint拒否、親判定待ち、counterの重複除外、pilot停止条件を含む |
| Knowledge登録 | `.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main`。638ノード、0 error、0 warning |
| 実測と公開版の照合 | [validation.json](validation.json)。14回の元出力・runtime・final・eventの指紋、使用量と公開reportの全数値、逐次実行と別native session、判定未記入、公開ファイル指紋を照合 |
| 公開の安全性 | 既存の`worker_logs.safety.Guard`で公開資料の全値と既知のnative識別子を検査。元のruntimeにあるsource位置は公開版で省略。数値と元出力の照合値を保持した |
| 秘匿化の境界 | [projection-checks.json](projection-checks.json)。大文字schemeのuserinfo URL、資格情報代入の値、UUID形式、既知のopaque識別子、ローカル絶対パスの除去と、比較標本ID・repo相対パス・公開GitHub URLの維持の8確認が成功 |
| ignore例外 | `git check-ignore --no-index`で今回の集計2件とrun観測が追跡可能であること、別の未登録report配下のJSONは除外されることを確認。4確認が成功 |
| 差分 | `git diff --cached --check`成功 |
| pre-commit | 今回の追加・変更ファイルを指定して成功。Python変更がないためruffとmypyはno files to checkでskip。実装準備時の検査記録はPR本文に残す |

一致判定と採否は検証結果に含めない。モデルが出した17件は全て親の判定待ちで、
正式レビューの実機受入、gate確認、段階導入は未実施。
