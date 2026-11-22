# PR #2023 3回目レビュー（parent-1990-1ebe43132af643d0ac3972bb5bbd7b4a, HEAD 53e5e2812）の対応

前回の修正は親が確認済み（新テストが修正前で失敗・修正後に全件成功、CI成功）。3回目のレビューで、major 1件と、証拠待ちの論点1件が出た。今回も配置切替・実削除・unit操作は行わない。

## pr-2023-1（leases.py:48 付近）unlink後のF_GETLEASEからlease解除までに起きたbreakを読み捨てている
指摘: removal.py:246 のunlink後の確認から、`write_lease` がF_UNLCKするまでの間に、directory fsync・journal記録・fstatがある。この間に外部hardlinkを `O_WRONLY|O_NONBLOCK` で開かれると、EAGAINとSIGIOが発生してもmetadataは変わらない。`write_lease` は解除時に届いたSIGIOを読み捨てて正常終了するため、`file_removal_failed` を残さずに evicted／mirror_retired として成功扱いになる。
- 求める動作: lease解除の直前（F_UNLCKの直前）にもbreak待ちを確かめ、解除時に回収したbreak通知とあわせて失敗へ伝播する。journal・auditには `phase=after_unlink`、`unlinked=true` を記録する。break検出の後で成功扱いになる経路（transactionの削除、evicted／mirror_retiredの返却）を残さない。
- テスト: 既存のblocking-writerテストを、unlink後のF_GETLEASEを通過した後（解除前）にwriter（`O_NONBLOCK` を含む）が開く時点にも広げる。修正前は成功扱いになり、修正後は失敗として記録されることを確かめる。

## 証拠待ちの論点: O_PATH参照と、別mount namespaceの外部file bind alias
レビューの問い: O_PATHのfdはread/writeのopen数に加算されず、lease取得を妨げない。保持したO_PATH参照（や `[local path]` 経由）から、lease解除後に削除済みinodeを再openして書き込める。token名を知らなくても書込みが失われ得るのではないか。
- 親の判断: これは、使用中検査の後に新しく参照を得て、削除の数マイクロ秒の窓に当てる使い方で、通常のconsumerの範囲外の前提として扱ってよい。ただし、事実を実測し、前提を正確にREADMEへ書くこと。
- 求めること:
  1. 隔離した一時環境（このWSL2 6.6）で、次を同期させて実測する。O_PATHで参照を持つ → lease取得の可否 → unlink → 解除後に `[local path]` から再openして書けるか。nlink=1の場合で確かめる。
  2. 使用中検査（同じUIDのプロセスのfdの列挙）が、検査時点で既に開いているO_PATHのfdを、device/inodeの照合で検出して「使用中」と判定するかを確かめる。検出しない場合は、検出するように直してテストする。
  3. 別mount namespaceの単一ファイルbind aliasについて、同じ実測を行う。単位内のファイルの外部bind aliasは、同じUIDのプロセスのmountinfoから検出できるかを調べる。検出が現実的なら実装し、そうでなければ前提として記す。
  4. 結果をREADME（3言語）の「残る前提」に正確に反映する。最終報告に、実測のコマンド・出力の要点を含める。

## 完了条件
- 再現テスト（修正前に失敗すること）、修正、README（3言語）、PR本文の更新。PR本文の閉じ指示は `Closes #1990` のままにする（親が直した。`Refs` に戻さない）。
- 対象HEADのCIで `Python` 成功まで確認する。
- 最終報告: 指摘の再現と修正、O_PATH・bind aliasの実測結果、テスト件数、残る前提。作業時間の上限は2時間。
