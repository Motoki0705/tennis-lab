# PR #1927 レビュー指摘の修正（review run parent-1881-bb3d555d1c4e42fba80c631ad35220fa、verdict request_changes）

固定HEAD 92a95098 に対する major 2件を、親が妥当と判断した。同じ branch・PR で修正する。

## pr-1927-1（`.agents/ops/parent/history.py:320`）取消記録の欠落でfail-openになる
旧親コメントに `agreed` があり、証拠用Issueへ保存した `basis_revoked` のコメントが削除されると、readerは取消のない履歴を有効として返し、`execution._current_basis` が旧合意を採用して委任等を通す。

- 読取時に、このstateが送信・照合済みとして持つ記録（comment outbox 等の journal）がremoteにすべて存在し、内容が一致することを検査する。欠落・改変があれば、取消に限らずすべての記録種別で明示エラーにする（静かに無視しない）。合意・basis・取消・停止・レビュー・通知など、権限やgateに効く記録が対象から漏れないこと。
- この検査が効かない範囲（別ホスト/別stateが書いた記録、journal導入前の旧親コメント）を明記する。旧親コメントだけの場合に同じ問題が main でも既存だったかを確認し、PR本文に事実として書く。
- 回帰テスト: 新旧混在の履歴で、証拠用Issueの取消コメントだけを削除すると委任・レビュー起動・merge判定が拒否されること。改変でも同様。正常時は従来どおり通ること。

## pr-1927-2（`.agents/ops/parent/notifications.py:167`）完了通知が親へ未公開のまま完了できる
`completion()` は証拠側の notification_record だけを確認する。証拠保存後・親POST前に中断すると、`workflow-report completed` が成功し `release --ended` で担当も終了でき、その後は owner 検査で notify の再試行もできない。

- 完了の条件に、対応する親側の通知コメントが公開済みで内容が一致することを含める。未公開なら完了・担当終了を拒否し、同じ notify の再実行で親側だけを補えること（二重投稿しない、応答喪失時はremote照合）。
- 他の通知種別でも、証拠記録だけが残って親側が未公開になる状態を、status等で検出できるようにする。
- 回帰テスト: 証拠保存後・親POST前の中断 → workflow-report completed と release --ended が拒否 → notify 再実行で親通知1件 → 完了・終了が通る。応答喪失で二重投稿しない。

## 共通
- 指摘以外の変更は、修正に必要な範囲に限る。
- 関連テストと pre-commit を通し、push 後の対象HEADのCI結果（Pythonを含む全ジョブ）を報告する。
- 作業時間の上限は3時間。止めてよいのは、2件の修正とテストを終えてCI結果が出たとき、または親の判断が必要な矛盾が見つかったとき（独立部分はコミット済みにして報告）。

## 報告
修正HEAD、各指摘の修正内容と回帰テストのnode ID、実行したテストと結果、変更した全パス、CI結果、検査が効かない範囲。
