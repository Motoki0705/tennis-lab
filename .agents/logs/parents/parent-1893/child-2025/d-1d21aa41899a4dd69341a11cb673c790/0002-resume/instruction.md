PR #2034 の統合レビュー（run parent-1893-f86fe5d603ba464bab11c520f519cc37、verdict request_changes）の major 1件を修正してほしい。有効と親が判断した。

### pr-2034-1 .agents/ops/worktree_cleanup/processes.py:133（major）
静的推論（再現テスト未実行）。cwd読取をsnapshot内へ移しましたが、対象の親Pについてstatと対象cwdを読んだ直後にPが終了すると、process_identityがNoneとなり、Pの観測全体が捨てられます。cwd読取中のFileNotFoundErrorでも同様です。Pの子Cが既に[local path]、同じsnapshotに含まれている場合、Pはprepare_nativeのlostにも残らず、Cはuncertain_processesでbaseline所属として除外されます。そのためCが稼働中でも停止確認とworktree削除が成功し得ます。観測済みのPID/start ticks・親子関係を消滅時にも保持し、対象または未確定の親の子孫を追跡してください。既存・追加の競合テストはsnapshot自体を置き換えており、この内部境界を検出できません。

## やること
- snapshot 内で観測した PID・start ticks・親子関係・cwd を、その後の消滅（process_identity が None、cwd 読取の FileNotFoundError 等）でも捨てずに保持する。対象 worktree を cwd に持つと観測した親、または所属が未確定の親の子孫は、子が別の cwd へ移っていても追跡し、消滅を確認できなければ削除を成功扱いにしない。
- snapshot を丸ごと置き換えるのではなく、snapshot 関数の内部境界（stat と cwd 読取の後に親が終了する、cwd 読取中に消える）を再現する回帰テストを追加し、修正を外すと失敗することを確認する。
- 同じ種類の窓が snapshot 内の他の箇所（子の列挙・親子関係の読取など）にもないか確認し、あれば同じ方針で閉じて PR 本文に書く。
- 関連テスト（前回と同じ範囲）と pre-commit を通し、commit して通常の push をする（force push しない）。PR 本文を更新し、対象HEADのCIを確認する。pytest の並列は `-n 4` 以下。
- 実ホストでの削除・unit 操作・状態ファイルの変更・シグナル送信はしない（読み取りだけの診断はよい）。
- 作業時間の目安は最大3時間。止めてよいのは、修正をpushしCIが成功した場合か、親の判断なしには進めない矛盾を見つけた場合だけ。
