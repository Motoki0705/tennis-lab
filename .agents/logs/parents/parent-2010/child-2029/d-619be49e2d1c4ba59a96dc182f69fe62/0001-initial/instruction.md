# #2029 の実装（PR作成まで）

担当Issue #2029 の本文が要件と受入条件の正本です。全数調査の報告（親 #2010 の証拠: https://github.com/Motoki0705/tennis-lab/issues/2013#issuecomment-6875909177 ）の「storeを渡したときの副作用と競合契約」「admissionの追加不具合」「P表」「回帰テスト計画」を前提にしてください。調査時の再現スクリプト（probe.py / supplemental.py / io-probe.py / review_driver.py と fixture）が `[local path]` にあります。読んで再利用してよいですが、このworktreeへコピーしてcommitしないでください。

## 進め方

- 最初に、transaction用clientとロック外用clientの分け方（APIの形・cacheの寿命・fresh読取の指定方法）を決め、README の該当契約を更新してから各入口へ適用する。
- 静かなフォールバック禁止: ロック不在時に読み取りを黙って許す、競合を無視する、空データへ変える、といった変更はしない。
- 子B（#2030、証拠用Issue探索のラベル限定）が並行して `parent/evidence.py` の探索と `parent/github.py` の `evidence_rows` を変更します。そこには手を入れず、衝突した場合は後からmergeする側がrebaseします。
- 規模が大きい場合は、1PRのまま複数commitに分けてよい。PRを分ける必要があると判断したら、実装を止めて理由を報告する。

## 完了の条件

- 受入条件のテストを含むPRを `Closes #2029` 付きで作成し、関連テスト（変更した範囲と `tests/e2e/agents_ops/` の関連ファイル）と ruff/mypy（pre-commit）が通り、PRの対象HEADでCI（Python）が成功していること。
- 時間の上限は8時間。全受入条件の実装とテストがそろった時点で終了する。途中で止めてよいのは、(1) 既存契約の大きな変更が必要と判明し親の判断が要る、(2) 8時間を超えた、のどちらかだけ。止める場合は、済んだ範囲・未了の受入条件・次の一手を報告する。
