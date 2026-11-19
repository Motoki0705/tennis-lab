# #1990 第1段の修正（PR #2023）

報告ありがとう。あなたが挙げた第2段前の判断事項について、人間との合意を更新した（#1990 の合意通知 https://github.com/Motoki0705/tennis-lab/issues/1990#issuecomment-6878903405 が正本）。同じbranch・PR #2023 に次を反映すること。今回も**配置の切替・実削除・unitの導入/有効化は行わない**。

## 人間の決定（合意済み。聞き直さない）

1. **training queueを監視しない。** `jobs/*.job`・`running/*.job` があれば全単位を保持する処理を削除する。使用中の判定は、開いているfd・cwd・mmap（device/inode照合、単位外hardlink別名の検出を含む）と、atime（14日）だけにする。待機中・予定のjobが参照するデータが2週間で消え、取得CLIを促すエラーになるのは許容する。`--queue-dir` など不要になった引数・設定・テスト・README記述も取り除く。
2. **プロセス検査は実行ユーザー自身のプロセスだけ。** 他ユーザー（root・CIランナー `tennis-actions` 等）のプロセスは検査対象外とし、読めないことを失敗にしない。ただし自分（同じuid）のプロセスの fd/maps が読めない場合は、従来どおり失敗として止める（静かなフォールバック禁止）。対象外にした範囲（他ユーザーのプロセス数など）は監査記録・ログに残し、READMEに前提として明記する。
3. **失敗通知はしない。** journalと監査記録（audit.jsonl）に残すだけ。`tennis-lab-storage-tier-failure.service` と `OnFailure=` を削除し、README・ops_controlの対象定義・テストを合わせる。

## 親の判断（合意の範囲内）

4. **worktreeのoutputsはD:の永続領域に置く**（あなたの推奨どおり）。メインcheckoutのD:上の決まった場所（例: `<main>[local path]<name>[local path]`。名前・配置はあなたが決め、規約の正本READMEに書く）を実体とし、worktreeの `outputs` はそこへのD:内symlinkにする。336時間の期限削除がworktreeを消しても実体は残り、storage_tierのミラー・照合・未使用削除だけがそれを消す。E:側の対応パスも決めて記録する。`prepare-worktree` をこの方式に合わせ、`data/` からのhardlinkがD:内で成立することをテストする。期限削除のツール（worktree_cleanup）がsymlinkの先を辿って消さないことを、コードで確認するかテストで固定する。worktree_cleanup の規約自体は変えない。
5. **メインoutputsを指す既存のworktree alias** は、メインoutputsへの参照として扱い、独立した単位にしない。第2段の移行では、D:のメインoutputsを指すように付け替える。
6. **未分類の `data/` 配下**（`ACCAD`・`smplh`・`smplx`・`samples`・`synthetic_data_generation`・`tennis_multivew`・`tennis_multivew_linux_retained` など）は今回は掃除の対象外のままでよい。ただし第2段の判断材料として、`scan` で未分類のトップレベルごとのサイズ（E:上の実測。可能なら `du` 相当でpayloadは読まない）を出し、報告に含める。

## 完了条件

- 上の1〜6を実装・テスト・README（3言語）に反映し、PR #2023 を更新する。対象HEADのCIで `Python` 成功まで確認する。PR本文も更新後の内容に合わせる。
- 実機の読取専用dry-runを、第1段と同様に小さな単位だけで再実行し、atime等が不変であることと、他ユーザーのプロセスを対象外にしてもcycleのdry-runが失敗しないことを確かめる。
- 最終報告: 変更点、テスト結果、dry-run結果、未分類data/のサイズ一覧、第2段の手順案（更新があれば差分）。
- 作業時間の上限は4時間。合意の範囲外になる分岐が出たときだけ、選択肢と推奨を書いて止まってよい。
