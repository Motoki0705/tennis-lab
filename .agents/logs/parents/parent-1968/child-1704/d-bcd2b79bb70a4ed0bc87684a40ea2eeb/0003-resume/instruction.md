## 親の具体的な指示（#1704、mainへの再追従）

- PR #1998 は、前回のrebase後に他のPRのmergeでmainが進み、`.agents/skills/parent-session/SKILL.md` で再び競合した。最新の `origin/main` にrebaseし、競合を解消して force-with-lease でpushする。
- 競合の解消では、main側に入った変更（他のPRの案内・規則）を落とさず、あなたのPRの意図（判断点から正本への案内）を保つ。同じ規則を2か所に書かない。skillの行数制限などの既存の形式検査を守る。どちらを採るか判断できない競合は推測で解消せず、その箇所と理由を最終報告に書いて止める。
- PRの変更はレビュー（run parent-1968-3644ca5fbb0a43e1afdca67b3872d14a）で承認済み。競合解消に不要な変更はしない。main側でも変更されたファイル以外を編集すると、レビューの継承が外れる。
- rebase後に、関係するテストを `.agents/ops/pytest_locked.sh` で実行し、pre-commitを通す。対象HEADのCI（`Python` を含む全job）の成功まで見届ける。PR本文の検証結果が変わった場合は本文を更新する。
- 最終報告には、rebase前後のHEAD、競合したファイルと解消内容、テスト・CIの結果を書く。
