# #2030 の実装（PR作成まで）

担当Issue #2030 の本文が要件と受入条件の正本です。全数調査の報告（親 #2010 の証拠: https://github.com/Motoki0705/tennis-lab/issues/2013#issuecomment-6875909177 ）の「全Issue探索・要求数・cache案」の節が前提です。人間の判断は「ラベル付きだけ一覧」です（ローカル binding 番号だけで個別GETする高速経路は採らない）。

## 進め方

- `issues?state=all&labels=parent-evidence` の全ページ取得へ変え、marker・投稿者・ラベル集合・ラベル付きどうしの重複・binding 照合・creation lock・曖昧POST契約は維持する。
- 1コマンド内で同じ探索を繰り返さない工夫は、書込みや明示的な鮮度確認の前に必ず取り直す範囲に限る。
- 子A（#2029）が並行して `parent/github.py` の client/cache 契約（transaction用とロック外用の分離）と review CLI を変更します。client/cache 契約には手を入れず、衝突した場合は後からmergeする側がrebaseします。
- 変更前後の要求数を `PARENT_METRICS_FILE` で測り、PR本文に表で載せる。

## 完了の条件

- 受入条件のテストを含むPRを `Closes #2030` 付きで作成し、関連テストと ruff/mypy（pre-commit）が通り、PRの対象HEADでCI（Python）が成功していること。
- 時間の上限は4時間。全受入条件がそろった時点で終了する。途中で止めてよいのは、(1) 人間の判断（ラベル限定）では保証が足りないと判明し親の判断が要る、(2) 4時間を超えた、のどちらかだけ。止める場合は、済んだ範囲・未了の受入条件・次の一手を報告する。
