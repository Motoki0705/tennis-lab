# parent.run-worker / 担当issue &#35;1961

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 3 件。記録時刻 2026-11-14T06:15:28.885041+00:00 〜 2026-11-14T06:16:12.318005+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

3 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

公開workerログ: 1 ターン（分母）。対応イベントあり: 1 ターン / 対応イベントなし: 0 ターン。イベントのみ: 0 ターン。和集合: 1 ターン。最終報告なし（終了・欠落の根拠あり）: 0 ターン。稼働中: 0 ターン。終了未確認: 0 ターン。最終報告あり・公開保留: 0 ターン。

対応数は公開worker3ファイルのあるターンを分母にしたものです。イベントのみのターンを加えた和集合とは分けます。対応イベントの存在は、親判断の記録・受入・網羅性を保証しません。

直近の観測（受入の判定とは別）: [結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e) rejected / machine: parent.run-worker: rejected before execution / ValueError / details in private evidence<br>[行動 1084](README.md#record-059f1d510feb4d478b83d17c670e8b8e) 同じAIの継続 / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-68dee177b4684cc59f28645dfa7840a2"></a>[行動 1082](README.md#record-68dee177b4684cc59f28645dfa7840a2)<br>2026-11-14T06:15:28.885041+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / [担当 &#35;1961](../child-1961/README.md) / [ターン 4](../child-1961/d-e2ca66b37b014611b0863cfecdc87b04/turn-0004/README.md) | [結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: requested<br>同じAIの継続 / 要求済み・実行未確認 | [記録本文](event-68dee177b4684cc59f28645dfa7840a2.json)<br>次の記録: [結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e)、[行動 1084](README.md#record-059f1d510feb4d478b83d17c670e8b8e) |
| <a id="record-23450f05ad8b4b17893ea80297736d2e"></a>[結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e)<br>2026-11-14T06:16:12.152071+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / [担当 &#35;1961](../child-1961/README.md) / [ターン 4](../child-1961/d-e2ca66b37b014611b0863cfecdc87b04/turn-0004/README.md) | parent.run-worker: rejected before execution<br>ValueError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-23450f05ad8b4b17893ea80297736d2e.json)<br>根拠非公開・記録不足: capture&#95;source=parent.run-worker; artifact=command response; content&#95;sha256=9ebe1d017fa6efe032b31692d3a28cf3f7fa36ccf73482b3459ce791708e432e; private evidence<br>前の記録: [行動 1082](README.md#record-68dee177b4684cc59f28645dfa7840a2)、[結果 1059](../activity-d5ed580da763436bbd6be4b5c556663b/README.md#record-c5ba9e2b8de646e4a781bd19fe87c6e4)<br>次の記録: [行動 1084](README.md#record-059f1d510feb4d478b83d17c670e8b8e) |
| <a id="record-059f1d510feb4d478b83d17c670e8b8e"></a>[行動 1084](README.md#record-059f1d510feb4d478b83d17c670e8b8e)<br>2026-11-14T06:16:12.318005+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / [担当 &#35;1961](../child-1961/README.md) / [ターン 4](../child-1961/d-e2ca66b37b014611b0863cfecdc87b04/turn-0004/README.md) | [結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: rejected&#95;before&#95;start<br>同じAIの継続 / 起動前拒否 | [記録本文](event-059f1d510feb4d478b83d17c670e8b8e.json)<br>前の記録: [行動 1082](README.md#record-68dee177b4684cc59f28645dfa7840a2)、[結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e) |

<details><summary>反映した記録と補助情報</summary>

- [結果 1083](README.md#record-23450f05ad8b4b17893ea80297736d2e)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-14T06:16:12.152055+00:00 / 記録時刻: 2026-11-14T06:16:12.152071+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

