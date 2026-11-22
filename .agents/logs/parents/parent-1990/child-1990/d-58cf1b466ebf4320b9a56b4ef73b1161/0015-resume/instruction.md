# PR #2023 7回目レビュー（parent-1990-4466ee5fb427494a87395a61a3365899, HEAD 0a297c434）の指摘修正

前回の境界検査と修正は親が確認済み（ローカル3,095件成功）。CIの所要時間は前回とほぼ同じ（CPU shardは各11〜20分、変化なし）だったので、件数の増加は問題にしない。7回目のレビューでmajor 1件が出た。今回も配置切替・実削除・unit操作・sudoの操作は行わない。

## pr-2023-1（tests/support/agents_ops/storage_crashes.py:285 付近）traceが、ファイル内の書込みや既存ファイルへの追記を捕捉していない
指摘: traceが記録するのは、Path.openの排他的作成、fsyncの後、append_jsonの完了後だけ。`stream.write`[local path]`flush`、`os.open` による作成・切詰め、既存ファイルへの追加のwriteは捕捉していない。copy_treeは8MiBごとに書き込むので、複数chunkの部分コピーの状態（作成直後から最終fsyncまでの間）が検査から漏れる。大きなJSONやjournalの途中書込みも同じ。そのためtraceの完全一致テストでは、新しい永続化ステップの追加を検出できない。
- 求めること:
  1. 実際の書込み（`os.open` のO_CREAT/O_TRUNC/O_APPEND、`os.write`、file objectの `write`、`os.truncate`[local path]`ftruncate` など、storage_tierが使う経路すべて）を捕捉するようにtraceを広げる。同じファイルへの連続したwriteを、1つの境界の種類（例: 「最初のchunkの後」「最終chunkの前」）にまとめて、件数を抑えてよい。境界ごとに全chunkを列挙する必要はない。
  2. 部分書込みの代表入力を加える。複数chunk（8MiB超）のファイルの部分コピー、途中まで書いたJSON（manifest・transaction・receipt）、末尾の行が途中で切れたjournal（audit.jsonl・removals・publication journalなど、追記するファイルすべて）。
  3. 契約(a)〜(d)を再照合する。特に、**末尾が途中で切れたjsonlや、途中まで書いたJSONを、recoverや日次cycleがどう扱うか**を決めて実装する。黙って読み飛ばして成功にしない。照合済みでない状態を正としない。必要なら、切れた末尾を退避してから明示的に復旧する。
  4. 境界表（fixture）と、traceの完全一致テストを更新する。新しい書込みの種類を足したときに、テストが失敗することを1件で示す（例: 未登録のwriteを足すと失敗する検査テスト）。
- 修正前（現HEAD）で、新しく捕捉した境界のうち失敗するものがあれば報告する。

## 完了条件
- テスト、修正、README（3言語。jsonlの切れた末尾などの扱いを書く）、PR本文（`Closes #1990` を維持）。
- 対象HEADのCIで `Python` 成功まで確認する。GitHub APIのrate limit（HTTP 403）が出たら、resetを待って再試行する。待機のために作業を終了しない。CIの所要時間が大きく増える（CPU shardが25分を超える）場合は、件数の削減を優先する。
- 最終報告: 追加した境界の種類と件数、修正前に失敗した境界、jsonl・JSONの途中書込みの扱い、テスト件数。作業時間の上限は2.5時間。
