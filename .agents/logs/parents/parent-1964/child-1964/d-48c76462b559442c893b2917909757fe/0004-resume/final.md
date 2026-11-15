PR [#2089](https://github.com/Motoki0705/tennis-lab/pull/2089) を作成・push済みです。`Closes #1964` を記載しています。

branch: `codex/proposal-expiry-registry`  
修正コミット: `7af022778`、`55fb9bb2c`  
HEAD: `55fb9bb2cb99fc41f3cf7b0e48dcfd0be979e8c4`

- 原因は、期限処理がregistry全体の再照合に束縛され、別親の更新も競合と判定していたことです。案(a)を採用し、検証済みsnapshotとdetached clientへ分離しました。実行・dry-runとも改善操作ロックで直列化します。
- expiryはregistry保存・`append`・判断ログ更新を行いません。初回dry-runの空ロックファイル作成だけを許容します。
- 追加テストは `test_expiry_preserves_peer_registry_updates_during_get` と `test_expiry_serializes_with_competing_improvement` の計8ケース。既存の取り出し復旧3ケースも拡張しました。
- CLIを修正前へ一時的に戻すと **4 failed / 4 passed**。観測されたregistryエラーとdry-runの直列化漏れを再現しました。
- 関連テスト **148 passed**、最終差分 **16 passed**。pre-commitのruff/mypyも成功しました。
- [最終HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43661797674/attempts/2) は、`Python` を含む全14ジョブ成功です。

CIのannotation側で単発のprocfs `PermissionError` が発生しましたが、失敗枠の1回再実行で成功しました。mainとの該当コード差分はなく、単独実行も両方成功しており、根本原因は未確定です。

残作業はmerge後の既存timer実機確認です。なお、最後のgh REST照会はrate limit（403）でした。CI成功とPR更新はGitHub連携経由で確認済みです。