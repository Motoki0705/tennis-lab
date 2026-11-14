## 親の具体的な指示（#1704、mainへの追従）

- あなたのPR #1998 は、他のPRのmergeでmainが進んだため、mainとの間に競合が出た。最新の `origin/main` にrebaseし、競合を解消して force-with-lease でpushする。
- 競合の解消では、main側に入った変更（他のPRの機能・テスト）を落とさず、あなたのPRの意図を保つ。どちらを採るか判断できない競合は推測で解消せず、その箇所と理由を最終報告に書いて止める。
- #1975（`parent wait-worker`）・#1977（要約の事前確認）がparent README・parent-session skillを変えている。merge済みの内容と矛盾しない案内にし、同じ規則を2か所に書かない。
- rebase後に、PRの変更に関係するテストを `.agents/ops/pytest_locked.sh` で実行し、pre-commitを通す。対象HEADのCI（`Python` を含む全job）の成功まで見届ける。
- 機能の追加や、競合解消に不要な変更はしない。PR本文の検証結果が変わった場合は本文を更新する。
- 最終報告には、rebase前後のHEAD、競合したファイルと解消内容、テスト・CIの結果を書く。
