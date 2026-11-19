# #1990 PR #2023 正式レビュー（Astra）の指摘対応

正式レビュー（run `parent-1990-61716d7fa4ef47c28dd6d02e0dc8d4b3`、HEAD 2b3ddeefa）で verdict=request_changes、major 2件。どちらも静的推論で、再現テストは未実行。まず各指摘を**再現する失敗テスト**を書いて実在を確かめ、実在すれば修正する。実在しない場合は、その根拠（テストとコード上の理由）を報告する。

## pr-2023-1（filesystem.py:134）照合していないpayloadを削除しない
Engine.evict は隔離後snapshotのroot情報だけを remove_tree へ渡し、clear は現在の配下を列挙して削除するため、子ファイルの内容・ctime・atimeが照合されない。隔離後snapshotの直後に、producerが保持していたdirectory FDから子ファイルを更新し、プロセス検査の前にcloseすると、使用中と判定できずrootのdev/inoも一致するので、E:にない更新内容を削除して成功扱いにできる。
→ 照合済みの全entry（manifest・最終snapshot）と実際に削除するentryを結び付け、削除時点で更新・追加・置換されたentryがあれば削除を止めて失敗にする（既に消したentryがある場合の記録と復旧手順も含む）。

## pr-2023-2（filesystem.py:238）管理境界のmountを検証し、ckptを削除対象にしない
real_directory は祖先を st_dev だけで確認し、snapshotのmount ID検査は単位rootより下に限られる。同じfilesystemの `R/ckpt` を `R/data/court_detection` へbind mountすると、ckpt内の版ディレクトリがdatasetとして発見され、14日条件とミラー照合を満たすとbind先経由でckpt配下が消える。
→ repo／mirrorの基準mountから単位rootまでの全経路でmount IDを検証する。また削除対象の実体がckpt配下でないことを、パス文字列ではなくdevice/inode等で確かめる。

## 共通
- 既存の故障注入テストを壊さず、上の再現テストを追加する。README（3言語）に関係する記述があれば合わせる。
- PR #2023 を更新し、対象HEADのCIで `Python` 成功まで確認する。
- 最終報告: 各指摘の再現有無、修正内容、追加テスト、テスト結果。
- 実機の配置切替・実削除・unit操作はしない。作業時間の上限は3時間。
