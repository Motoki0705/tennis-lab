major 修正（HEAD c4c9143b）を親が確認した。一覧欠落・部分可視の扱いは良いが、新たに次の問題があるため同じbranch・PRで直す。

1. workflow は `parent-evidence` 以外の**すべての** Issue の `issues.closed` で起動する。証拠用Issueを持たない Issue（子・standalone・proposal・ops-report、担当登録されなかった親など、クローズの大半）では `--parent N evidence-close --apply` が `binding_not_found` → 終了コード3 → 3回試行後に workflow が失敗する。Actions が日常的に赤くなり、本当の照合失敗と区別できなくなる。
2. 1回のイベントごとに `repos/.../issues?state=all` でリポジトリ全Issue（PRを含む数千件）をページングして取得しており、クローズのたびに重い。

求める修正（方式は担当が判断し、理由をPRに書く）:
- 「この Issue には証拠用Issueが存在しない」ことと「存在するはずだが一覧の反映遅れで見えない」ことを区別する。たとえば証拠用Issueの本文が親を参照していることを使い、親の timeline の cross-reference など反映遅れの影響を受けない経路で候補の有無・番号を確定し、ラベル一覧との一致を確認する。証拠を持たない Issue のクローズは成功（何もしない）で終わり、候補が存在するのに確認できない場合だけ再照合・失敗にする。
- 1イベントあたりの API 取得を対象の親に関係する範囲に抑える（全Issueの一覧取得を避ける）。一括処理（`--parent` 無し）は従来どおりでよい。
- 誤クローズを防ぐ条件（確認済みの一意な紐付け・親がclosed）は弱めない。
- 回帰テスト: 証拠用Issueを持たない Issue のクローズで成功・何もしないこと、存在するが一覧に未反映のケースで3（再照合）になること、全一覧を取得しないこと。
- 検証・pre-commit・actionlint を通してcommit・pushし、PR番号と完全なHEADを宣言して終了する。
