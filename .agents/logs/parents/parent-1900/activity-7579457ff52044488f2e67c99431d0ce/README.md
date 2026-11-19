# parent.create-child / 親 &#35;1900 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-14T23:14:52.751416+00:00 〜 2026-11-14T23:15:06.932094+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed) failure / machine: parent.create-child: post-start failure / ValueError / details in private evidence<br>[行動 1860](README.md#record-b27d27c376ff42a484416f3bf67bc932) 記録 / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-568dd0af3bb34fd3b681f4deef5bd474"></a>[行動 1857](README.md#record-568dd0af3bb34fd3b681f4deef5bd474)<br>2026-11-14T23:14:52.751416+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | 結果記録なし | 記録不足・判断: parent decision not recorded for this operation | parent.create-child: requested<br>記録 / 要求済み・実行未確認 | [記録本文](event-568dd0af3bb34fd3b681f4deef5bd474.json)<br>次の記録: [行動 1858](README.md#record-47de005f297045e280296869a004a787) |
| <a id="record-47de005f297045e280296869a004a787"></a>[行動 1858](README.md#record-47de005f297045e280296869a004a787)<br>2026-11-14T23:14:53.017324+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed) | 記録不足・判断: parent decision not recorded for this operation | parent handler entered; no AI launch implied<br>記録 / 実行中 | [記録本文](event-47de005f297045e280296869a004a787.json)<br>前の記録: [行動 1857](README.md#record-568dd0af3bb34fd3b681f4deef5bd474)<br>次の記録: [結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed)、[行動 1860](README.md#record-b27d27c376ff42a484416f3bf67bc932) |
| <a id="record-6cfcf30ba3124a52848f94652716d5ed"></a>[結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed)<br>2026-11-14T23:15:06.652329+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | parent.create-child: post-start failure<br>ValueError<br>details in private evidence<br>failure / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-6cfcf30ba3124a52848f94652716d5ed.json)<br>根拠非公開・記録不足: capture&#95;source=parent.create-child; artifact=command response; content&#95;sha256=3e40fec8fde94f4b63abddedf83fc58d8c2d36925fccc91b46736e4b1cfb78f4; private evidence<br>前の記録: [行動 1858](README.md#record-47de005f297045e280296869a004a787)<br>次の記録: [行動 1860](README.md#record-b27d27c376ff42a484416f3bf67bc932) |
| <a id="record-b27d27c376ff42a484416f3bf67bc932"></a>[行動 1860](README.md#record-b27d27c376ff42a484416f3bf67bc932)<br>2026-11-14T23:15:06.932094+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed) | 記録不足・判断: parent decision not recorded for this operation | parent.create-child: ended<br>記録 / 終了（成果受入とは別） | [記録本文](event-b27d27c376ff42a484416f3bf67bc932.json)<br>前の記録: [行動 1858](README.md#record-47de005f297045e280296869a004a787)、[結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed) |

<details><summary>反映した記録と補助情報</summary>

- [結果 1859](README.md#record-6cfcf30ba3124a52848f94652716d5ed)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-14T23:15:06.652314+00:00 / 記録時刻: 2026-11-14T23:15:06.652329+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

