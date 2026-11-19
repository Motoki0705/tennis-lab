# PR #2148 3回目レビュー（parent-1990-de0df596adac406f91ae448ed770d2a8）の指摘の修正

前回2件の解消は確認した（修正前HEADで4件失敗→修正HEADで82件成功、CI全14ジョブ成功）。レビューでは、常駐単位・未管理9参照・container検査・dry-run契約・#2023の経路に新たな問題はなかった。残りはmajor 1件。先に再現テストを書いて修正前に失敗することを確かめてから直すこと。PR本文は作業中 `Refs #1990` のまま変更しない。実機の配置切替・index・hydrate・実削除・unit・sudoは行わない。

## pr-2148-1（local_storage.py:100）通常のDATA包含検査がGitの利用可否に依存する
指摘: POLICY_CONFIGが存在すると、root名が `data` で `path != resolved` の通常の内部リンクにも、custom root判定より先に `_standard_repository()`（git subprocess）を実行する。例: custom root `[local path]` 内の `alias→real` を通る `alias/input.bin` は、解決先が同じroot内でも、git未導入なら FileNotFoundError、`.git` のない展開済みcheckoutなら CalledProcessError から PathContractError になる。従来成功した入力が、E:参照の例外を必要としないのに失敗する。
- 求める動作: 解決先がroot内に収まる通常の包含の成功を、Gitから完全に切り離す（例: `resolved` がroot内なら例外判定に入らず従来の包含規則に任せる、など。宣言リンクがroot内を指していた場合の扱いは、storage_tierのscan/activateの検証との責務分担として明示し、READMEの契約と矛盾させない）。root外への解決（E:参照の追加権限）が必要な場合だけ、標準rootの判定とリンクの完全検証を行い、そこでGitが使えなければ従来どおり明確なエラーにする。
- Gitの照会を解決ごとに繰り返さない（プロセス内でcommon dirは不変なので、結果のcacheは可。policy・alias内容はcacheしない現行方針は維持）。
- テスト: 既存のcustom rootテストを拡張し、Git不在（PATHにgitがない）・Git metadata不在（`.git` のない展開checkout）でも内部リンクが成功し、外部リンク（E:形状を含む）が拒否されることを検証する。標準root・worktreeの既存テストは維持する。

## 完了条件
- 再現テスト（修正前に失敗）、修正、必要ならREADME（3言語）とPR本文の更新。
- PathResolverを使うconsumerの既存テストの成功を確認する。
- 対象HEADのCIで `Python` 成功まで確認する。rate limitが出たらresetを待って再試行。
- 最終報告: 再現と修正、テスト件数、CIのURLと最長shard時間。作業時間の上限は1.5時間。
