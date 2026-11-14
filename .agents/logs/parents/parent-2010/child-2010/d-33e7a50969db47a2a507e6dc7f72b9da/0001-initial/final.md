全数調査を完了しました（約41分）。**全51箇所の表・再現結果・修正計画**を[調査報告](https://github.com/Motoki0705/tennis-lab/blob/a6ac281e3ec167429ddc92dff0417905ea60e956/tmp/registry-lock-audit/report.md)へ保存しました。

| 箇所 | 確認した問題 | 修正方針 |
|---|---|---|
| `review/parent_review.py` 6入口 | store漏れ。直接gh・gitもロック内 | transaction内外のclientを明示して分離 |
| `dispatcher/` | store漏れ、runner status/stop待機 | 観測・予約・結果保存を段階化 |
| `opsreport/triage.py:182` | 整理runnerをtimeoutなしでロック内実行 | 計画実行をロック外へ移動 |
| `parent/snapshot.py`、merge gate | git fetch/diff等が保持中 | 固定入力でロック外検証 |
| `board/` | 通知・Herdr更新・sessions走査が保持中 | 担当固定を維持するfenceを設計 |
| `project_progress/github.py` | mutation・REST POSTが保持中 | Project操作の排他を残して通信を退避 |
| `manual_workers.py:475` | rollbackもGitHubへ接続し、継続403で幽霊runが残る | 通信不要の新transactionで巻き戻す |

fake ghで1,000件の無関係Issueを加えた初回`resolve`は、**146要求すべてで別プロセスがlockを取得できず**、11ページの全Issue走査を9回繰り返しました。store結合を仮適用すると37要求・2巡になりましたが、git処理は依然として保持中です。

`store=store`追加だけでは通常runのロック外collectが失敗します。また、dispatcherではschedule変更の再検査、focus範囲、観測間cacheの更新も必要です。6つのreview入口では受付中の競合を終了値2で拒否できましたが、run後段のcatchは競合を終了値1へ変換し得ます。

実装は次の**3PRへの分割**を推奨します。

1. transaction/client契約、review、手動admissionの修正
2. dispatcher・triageのI/Oと状態更新の分離
3. board・Projectの残る外部I/Oと担当固定の整理

SH/snapshot導入、証拠Issueの別番号重複を毎回検出する保証、board配送中の担当固定方式は判断事項です。まずEXを維持して長時間I/Oを退避する方針を推奨します。

検証は既存関連テスト**45 passed**。一時再現スクリプトと結果JSONも報告と同じディレクトリにあります。

branchは`codex/registry-lock-io`、HEADは`a6ac281e3`。追跡ファイルの差分・新規commit・PRはありません。実装、全関連テスト・lint/typecheck、Issueへの集約、merge・全体受入は未実施です。