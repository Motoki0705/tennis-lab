担当: #2392（親 #2474）。Issue本文の Acceptance checklist 3項目をすべて満たす実装・テストを行い、1つのPR（本文に `Closes #2392`）を作る。

- `parent register` が保存後の証拠Issue確保・投稿で失敗しても、agyの `status-line` hookが未確定状態を原因に毎回 `exit 1` を返し続けて自動無効化されないようにする。表示先の不整合を残さない方式か、hook側で未確定を安全に扱う方式かは調査のうえ判断し、理由をPRに書く。未確定を黙って成功扱いにする静かなフォールバックは避け、状態が観測できるようにする。
- `agy_hook.py` の失敗時に、ローカル識別子などの機密を漏らさずに原因区分（owners未確定・Herdr未接続・boardエラー等）を識別できる出力または記録を追加し、テストで示す。
- `evidence.ensure()` のPOST直後の照合が一覧APIの反映遅延で誤って不確定エラーにならないことを、一覧が新Issueをまだ返さないモック条件のテストで示す。
- 同じ親の別の子（#2432）も `.agents/ops/parent/evidence.py` を変更する可能性がある。変更は必要最小限にし、競合時は最新mainに追従する。
- 検証・pre-commitを通してcommit・pushし、PRを作ったら、PR番号と完全なHEADを宣言して終了する（CIの成功確認は結果を受けた後続ターンで行う。`resume_on_success` はCI成功後に残作業があるときだけtrue）。
