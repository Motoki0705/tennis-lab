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

## 親判定反映とeffort修正の検証設計

親が14 run・17指摘の判定を記入した。判定の内容を変更せず、schema、元出力と正解表の照合、
指摘の分類漏れ・重複を検証してから、非公開stateへ既存report CLIを適用する。
実行コードの固定を検査する仕組みを維持するため、再計算はeffort修正前に行う。
元の実行証拠・使用量・公開projectionの照合値を保持し、新しい判定の来歴を別に記録する。

effort修正で守る契約は、人間の明示effortを保持し、親との矛盾と許可外の組を拒否すること。
検出する故障は、human modelを補う際に明示effortを親のeffortで上書きすること。
既存unitファイルのsettings fixtureを再利用して、保持・矛盾・許可外の3 nodeを追加する。
さらに実際のCLI引数登録とselection関数で、`--human-review-effort`を渡す1 nodeを追加する。
引数からpolicy選択までで検出できるため、親registry・git履歴・fake runtimeを作るE2Eは増やさない。
計4 nodeを修正前に実行し、実装の問題を原因として失敗することを確認する。
pytestは`-n 0`、1回1〜2 nodeで逐次実行し、setup・call時間を測る。
既存のfixed既定とhuman model指定の拒否契約も少数nodeで確認する。

親判定の検証と再計算は成功し、書式修正は0件、追加AI起動は0回だった。
元の入力・出力・使用量・runtimeの指紋、CLI reportの全数値、公開ファイルの指紋を照合した。
親が記入したadjudications.jsonはバイト単位でも変更していない。
検証結果は[adjudication-validation.json](adjudication-validation.json)と
[validation.json](validation.json)に保存した。

修正前の4 nodeは、2件ずつの実行でそれぞれ0.11秒と0.20秒で失敗した。
保持のnodeでは`human.effort`が`xhigh`から`None`へ失われ、残り3件は
期待するValueErrorが発生しなかった。importやfixture起動の失敗ではない。

正式レビューの実機受入、gate確認は親の残作業。追加の実LLM・実agent起動は行わない。
