## 親からの継続指示（正式レビューの指摘修正）

PR #1867 の正式な親レビュー（run `parent-1797-41cc122760e141e9b87029ab80d1231a`、[結果](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6766611577)）は request_changes で、major 2件。親は両方を妥当と判断した。同じbranch・PRで修正する。

### pr-1867-1（major）古い採用元で後続runの未解決指摘を隠さない
`.agents/ops/review/parent_review.py` 769付近。同じsnapshotで、run Aに承認の保存final、後続run Bに未解決majorがある場合でも、新しいrun CからAをrecoverでき、gateは最新requestのCだけを見るため、Bの指摘をresolveせずに通せる。
- recover時とgateの両方で、元run以後に同じPRを対象とした全run（失敗・incomplete・request_changes・採用を含む）を照合し、未解決の指摘・失敗・取消・矛盾があれば、古い結果の採用を拒否するか、その指摘をgateの未解決として残す。どちらにするかは安全側で設計し、READMEに契約を書く。
- A→B→C のケースのテストを追加する（Bの指摘がresolveされるまでgateが止まること、採用でBを消せないこと）。

### pr-1867-2（major）実行証拠の内容不一致を復旧で承認へ変換しない
`.agents/ops/review/review_recovery.py` 50付近。共通ランナーは `final.md` と完了イベントの本文が異なると `invalid_output` で失敗し `agent_output.md` へコピーしない（`.agents/ops/lib/README.md` の `final.md`・`invalid_output` / `metadata_error` の契約）。現在の `runtime_evidence` は失敗理由とイベントを照合せず、`saved_final` は `final.md` 単独を受理するため、完了イベントが request_changes で `final.md` がschema上有効なapproveでも、候補保存→recover→gateで承認を採用できる。
- 保存済みイベント（events.jsonl）の最終メッセージと `final.md` の対応を再検証し、不一致は拒否する。runtimeの失敗理由を区別し、`invalid_output`・`metadata_error`（実効設定・実行先の不一致を含む）は復旧の対象にしない。復旧してよい失敗理由を許可リストで明示する（例: 通信エラー後に完了イベントとfinalが一致しているもの）。
- fake runtimeで、イベントとfinalの不一致、`invalid_output`、`metadata_error` を表現し、拒否を確認するテストを追加する。

### 共通
- 他にも同じ種類の抜け道（古い証拠や食い違う証拠を承認に使える経路）が無いか、変更した範囲を見直す。
- pytestは引き続き `-n 0` で、対象のnode・ファイルだけを順に実行する（WSL停止の再発防止）。全体はCIで確認する。
- CIのレーン0はapt取得の基盤障害（#1857 / PR #1858）で失敗が続いている。pushごとのCIは確認するが、同じ準備失敗の盲目的な再実行はしない。PR本文のCI欄を、新しいHEADの結果で更新する。
- 修正後、PR本文の受入条件と検証表を更新し、最終報告で各指摘への対応、追加テスト、故障注入（少なくとも各指摘1件）の結果を返す。作業時間の目安は最大3時間。
