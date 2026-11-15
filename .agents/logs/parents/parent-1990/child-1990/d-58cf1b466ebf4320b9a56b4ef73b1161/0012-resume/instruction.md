# PR #2023 4回目レビュー（parent-1990-744d9009deb04ba1bfbc9673c1258817, HEAD f17898523）の指摘修正

前回の作業は親が確認済み（ローカル176件成功、新テスト20件が修正前で失敗、CI成功）。4回目のレビューでmajor 2件が出た。どちらも静的推論なので、**先に再現テストを書いて失敗を確かめてから**直すこと。今回も配置切替・実削除・unit操作・sudoの操作は行わない。

## pr-2023-1（usage.py:334 付近）単位内mountのidentityが確認できないのに、削除を通す
指摘: 単位内ファイルのbind先に別のmountを重ねると、下層のbindはmountinfoに残る。しかし `[local path]` のstatは上層の別inodeを返すため、下層の参照を無視して、evict・旧ミラーの退役が成功する。上層を外したconsumerは、lease解除後に削除済みのinodeへ書き込める。
- 求める動作: mountのroot（mountinfoの4列目）が単位の内部にあるのに、statしたinodeが単位のentryと一致しない場合は、unknownとして削除を止める（fail closed）。祖先・単位外をrootとするbind（CIランナー用のbindなど）は、これまでどおり除外のままにする（実機のE:照合を再び止めないこと）。
- 既存の `test_mount_reference_inspection_uses_identity_and_fails_closed` の `other-inode` ケースは、削除の成功を期待している。これを失敗を期待する形に直し、重ねたmountのケースを追加する（実mountのe2eで可能なら実装し、CIでskipされる場合はintegrationの擬似mountinfoでも固定する）。

## pr-2023-2（checkpoint_files.py:107 付近）ckpt直下ファイルのhydrateで、保存済みのSHA-256と比較していない
指摘: checkpoint-filesのhydrateは、E:を再ハッシュしても、previousとの比較が `same_metadata` だけになっている。metadataが一致してhashだけが食い違う場合に、その内容をD:へ公開し、205行目の `record` で既存の証拠も上書きして成功する。
- 求める動作: 通常の `Engine.hydrate` と同じく、保存済みmanifestのSHA-256との一致を要求する。不一致なら、D:へ公開せず、manifestも更新せずに失敗する。checkpoint_files.py の他の経路（mirror・verify・index・退役）でも、保存済みhashとの比較が抜けている箇所がないか確認し、あれば同様に直す。
- テスト: `test_hydrate_checks_e_hash_against_existing_manifest` の契約を、ckpt直下ファイルにも広げる。

## 完了条件
- 再現テスト（修正前に失敗すること）、修正、README（3言語。前提の記述に変更があれば）、PR本文（`Closes #1990` を維持）。
- 対象HEADのCIで `Python` 成功まで確認する。
- 実機の読取専用QAで、E:側の照合がunknown 0のままであることを確かめる（CIランナー用のbindで止まらないこと）。
- 最終報告: 各指摘の再現と修正、テスト件数、実機QAの要点。作業時間の上限は1.5時間。
