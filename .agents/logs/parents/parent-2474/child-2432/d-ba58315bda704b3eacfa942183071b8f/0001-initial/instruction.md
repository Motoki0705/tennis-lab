担当: #2432（親 #2474）。Issue本文の Acceptance checklist をすべて満たす実装・テスト・README更新を行い、1つのPR（本文に `Closes #2432`）を作る。

- 親のクローズ（手動・PRマージの両経路）に連動して、確認済みの紐付けを持つ open の `parent-evidence` 証拠用 Issue をクローズする経路を実装する。紐付けを確認できないもの・親がopenのもの・通常の作業Issueには及ばないこと（静かなフォールバック禁止）。
- 親がクローズ済みの既存の残存分を、冪等に解消できる手段（dry-run既定・明示apply等、repoの既存CLI流儀に合わせる）を用意する。実GitHubへの一括クローズ（apply）は、このターンでは実行しない。dry-runで対象一覧を出し、その結果（対象番号と根拠）をPR本文に載せる。apply実行はマージ後に親が判断する。
- クローズ後も証拠コメントのpermalinkと親CLIの履歴読み取りで同じ記録を参照できることをテストで示す。
- 同じ親の別の子（#2392）も `.agents/ops/parent/evidence.py` を変更する可能性がある。変更は必要最小限にし、競合時は最新mainに追従する。
- 検証・pre-commitを通してcommit・pushし、PRを作ったら、PR番号と完全なHEADを宣言して終了する（CIの成功確認は結果を受けた後続ターンで行う。`resume_on_success` はCI成功後に残作業があるときだけtrue）。
