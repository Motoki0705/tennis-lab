# #2368 hot copyの規則とVHDX縮小の手順を文書化する

親issue #2361 の「全体方針・合意事項」と、担当issue #2368 の本文（変更内容・Acceptance checklist）が仕様の正本です。まず両方を読んでください。

## 人間と合意済みの要点
- VHDXの上限を物理D:（931.5GiB）未満へ縮める。実施の時期は人間が決めるので、手順の文書化だけを行い、実施はしない。
- outputsはE:、`data/`・`ckpt/` はD:に残す。storage_tierの削除条件（dataの14日atime、ckptの保持）は変えない。
- 物理D:の空きが20GiB未満になると新規起動を止めるwatchdogを、#2367 で別に作っている。hot copyの規則はこれと矛盾しないように書く。

## 進め方
1. 置き場所は `.agents/README.md` のファイル地図と「ドキュメントを二重管理しない」規則に従う。storage_tierのREADMEに書くなら英語・日本語・中国語の3つを揃える。
2. hot copyの規則は、既存のstorage_tierの仕組みで扱える形を優先する。新しい台帳ツールは実装しない。11/14に作られた `[local path]`（147G、登録も期限もなく残った）を失敗例として、登録先・期限・解放方法を具体的に決める。
3. VHDX縮小の手順は、現環境で実際に使えるコマンドに基づいて書く。`wsl.exe --version` と、使う各コマンドのhelpの出力を確認して証拠に残す。VHDXの所在は `[local path]`。ディスクを変更するコマンド（fstrim、resize2fs、`wsl --manage`、diskpart、`wsl --shutdown` など）は実行しない。
   - 必ず書くこと: 事前のfstrimとcompact、ext4の縮小、VHDXの最大サイズの変更、目標サイズの決め方、所要時間の見積り、D:に必要な作業用の空き、途中で失敗したときの戻し方、`fstrim.timer` がWSLで `ConditionVirtualization=!container` によりskipされる事実と手動fstrimの手順。
4. 文書テスト（`tests/e2e/agents_ops/test_agents_docs.py` など、関係するもの）を通す。
5. PR本文は日本語で、`Closes #2368` を書く。受入条件ごとの証拠を載せる。
6. 検証してcommit・pushし、PR番号と完全なHEADを宣言して終了する。CIの結果は後続のターンで受け取る。
