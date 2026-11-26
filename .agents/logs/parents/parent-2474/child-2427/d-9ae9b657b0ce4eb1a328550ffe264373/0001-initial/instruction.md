担当: #2427（親 #2474）。Issue本文の Acceptance checklist 4項目をすべて満たす実装・テスト・README更新を行い、1つのPR（本文に `Closes #2427`）を作る。

- `wait-worker` など外部状態（queue job・CI・空き枠）を待つ長い待機の間、checkoutの共有lock（`ops-code.lock`）を保持しないようにする。待機から戻ってmainのコードを使う前にlockを取り直し、コードの版が食い違えば理由を出して非ゼロで止める（黙って古いコードで続けない）。
- `review/merge_gate.py` のCI待機など、他の長い待機の入口を一覧にし、各入口の待機中のlockの扱いを `.agents/ops/lib/README.md` の表とテストで示す。
- git-syncが続けて見送ったとき、回数・時間・最後の `lock_holders` を含む通知を出す。しきい値と通知先をREADMEに書く（通知先は既存の運用通知の仕組みを優先して再利用する）。
- `.agents/ops/lib/README.md` と `.agents/ops/parent/README.md` のlock説明を実装と一致させる。
- 実機の journal 確認（queue待ちの `wait-worker` 中にgit-syncが `updated`[local path]`unchanged` になること）は、main checkoutの常駐unitに影響するため、このターンでは行わない。代わりにテストで同等の状況を再現し、実機確認の手順をPR本文に書く。実機確認はマージ後に親が行う。
- 検証・pre-commitを通してcommit・pushし、PRを作ったら、PR番号と完全なHEADを宣言して終了する（CIの成功確認は結果を受けた後続ターンで行う。`resume_on_success` はCI成功後に残作業があるときだけtrue）。
