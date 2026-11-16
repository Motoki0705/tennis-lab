# parent.run-worker / 担当issue &#35;1750

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 3 件。記録時刻 2026-11-15T15:05:13.915106+00:00 〜 2026-11-15T15:05:44.199624+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

3 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

公開workerログ: 1 ターン（分母）。対応イベントあり: 1 ターン / 対応イベントなし: 0 ターン。イベントのみ: 0 ターン。和集合: 1 ターン。最終報告なし（終了・欠落の根拠あり）: 0 ターン。稼働中: 0 ターン。終了未確認: 0 ターン。最終報告あり・公開保留: 0 ターン。

対応数は公開worker3ファイルのあるターンを分母にしたものです。イベントのみのターンを加えた和集合とは分けます。対応イベントの存在は、親判断の記録・受入・網羅性を保証しません。

直近の観測（受入の判定とは別）: [結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919) rejected / machine: parent.run-worker: rejected before execution / GitHubHTTPError / details in private evidence<br>[行動 2492](README.md#record-de5ca497ac9e4e5c925b7bcd0d85594e) 新しいAIの起動 / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-5ee10a788e704bd98ec06e8f57698794"></a>[行動 2490](README.md#record-5ee10a788e704bd98ec06e8f57698794)<br>2026-11-15T15:05:13.915106+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / [担当 &#35;1750](../child-1750/README.md) / [ターン 1](../child-1750/d-9aef71789d794a93a5e32a40d679ad61/turn-0001/README.md) | [結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: requested<br>新しいAIの起動 / 要求済み・実行未確認 | [記録本文](event-5ee10a788e704bd98ec06e8f57698794.json)<br>次の記録: [結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919)、[行動 2492](README.md#record-de5ca497ac9e4e5c925b7bcd0d85594e) |
| <a id="record-7eb876759fdb4e059d851d72c52b9919"></a>[結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919)<br>2026-11-15T15:05:44.157540+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / [担当 &#35;1750](../child-1750/README.md) / [ターン 1](../child-1750/d-9aef71789d794a93a5e32a40d679ad61/turn-0001/README.md) | parent.run-worker: rejected before execution<br>GitHubHTTPError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-7eb876759fdb4e059d851d72c52b9919.json)<br>根拠非公開・記録不足: capture&#95;source=parent.run-worker; artifact=command response; content&#95;sha256=79aa4f194b49275ad67cb3a4acba30c568c486cbb17ccf566e3ddfcd5297c408; private evidence<br>前の記録: [行動 2490](README.md#record-5ee10a788e704bd98ec06e8f57698794)<br>次の記録: [行動 2492](README.md#record-de5ca497ac9e4e5c925b7bcd0d85594e) |
| <a id="record-de5ca497ac9e4e5c925b7bcd0d85594e"></a>[行動 2492](README.md#record-de5ca497ac9e4e5c925b7bcd0d85594e)<br>2026-11-15T15:05:44.199624+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / [担当 &#35;1750](../child-1750/README.md) / [ターン 1](../child-1750/d-9aef71789d794a93a5e32a40d679ad61/turn-0001/README.md) | [結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: rejected&#95;before&#95;start<br>新しいAIの起動 / 起動前拒否 | [記録本文](event-de5ca497ac9e4e5c925b7bcd0d85594e.json)<br>前の記録: [行動 2490](README.md#record-5ee10a788e704bd98ec06e8f57698794)、[結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919) |

<details><summary>反映した記録と補助情報</summary>

- [結果 2491](README.md#record-7eb876759fdb4e059d851d72c52b9919)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-15T15:05:44.157526+00:00 / 記録時刻: 2026-11-15T15:05:44.157540+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

