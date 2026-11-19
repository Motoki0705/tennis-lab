# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;1797 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-13T03:19:46.405513+00:00 〜 2026-11-13T03:29:01.926613+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 2 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980) rejected / machine: merge&#95;gate.merge: rejected before execution / ValueError / details in private evidence<br>[行動 778](README.md#record-6f72956eb2ba41db887f8c7783579535) merge / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-88ccffc509464c578cc458d233b8736d"></a>[行動 770](README.md#record-88ccffc509464c578cc458d233b8736d)<br>2026-11-13T03:19:46.405513+00:00（記録時刻）<br>[親 &#35;1797](../README.md) / 親の直接活動 | [結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)、[結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-88ccffc509464c578cc458d233b8736d.json)<br>次の記録: [結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)、[結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980)、[行動 778](README.md#record-6f72956eb2ba41db887f8c7783579535) |
| <a id="record-0bab7acbaed84dc9993a84d0f75837f1"></a>[結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)<br>2026-11-13T03:25:10.788967+00:00（記録時刻）<br>[親 &#35;1797](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=1949<br>判断: blocked<br>操作: blocked<br>状態: 記録値なし<br>理由: pr&#95;outside&#95;review&#95;targets、ci&#95;python&#95;missing、review&#95;missing<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-0bab7acbaed84dc9993a84d0f75837f1.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=9c050d427cd86aae9a9f79cecb74c333bd39c0f63501abbb9c49951c5614ded2; private evidence<br>前の記録: [行動 770](README.md#record-88ccffc509464c578cc458d233b8736d)、[結果 768](../activity-068354d75f6844839b28d5599a00e468/README.md#record-74d6ec71295e40e89ccbc3ed33d29e67)<br>次の記録: [結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980)、[行動 778](README.md#record-6f72956eb2ba41db887f8c7783579535) |
| <a id="record-dea384d91dbb4a539abe0d7a41dbc980"></a>[結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980)<br>2026-11-13T03:29:01.558883+00:00（記録時刻）<br>[親 &#35;1797](../README.md) / 親の直接活動 | merge&#95;gate.merge: rejected before execution<br>ValueError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-dea384d91dbb4a539abe0d7a41dbc980.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=df4b1770ff4feaaeb6cb1d50fa8cc7f25709b7a0a143ffbe14586fec2a3fff71; private evidence<br>前の記録: [行動 770](README.md#record-88ccffc509464c578cc458d233b8736d)、[結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)<br>次の記録: [行動 778](README.md#record-6f72956eb2ba41db887f8c7783579535)、[結果 782](../activity-2ec519e5bf4a4b4f983079f4991065dc/README.md#record-331dd61ee73b441db1ba29f31616a251) |
| <a id="record-6f72956eb2ba41db887f8c7783579535"></a>[行動 778](README.md#record-6f72956eb2ba41db887f8c7783579535)<br>2026-11-13T03:29:01.926613+00:00（記録時刻）<br>[親 &#35;1797](../README.md) / 親の直接活動 | [結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)、[結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: rejected&#95;before&#95;start<br>merge / 起動前拒否 | [記録本文](event-6f72956eb2ba41db887f8c7783579535.json)<br>前の記録: [行動 770](README.md#record-88ccffc509464c578cc458d233b8736d)、[結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)、[結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980) |

<details><summary>反映した記録と補助情報</summary>

- [結果 771](README.md#record-0bab7acbaed84dc9993a84d0f75837f1)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-13T03:25:10.788954+00:00 / 記録時刻: 2026-11-13T03:25:10.788967+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 776](README.md#record-dea384d91dbb4a539abe0d7a41dbc980)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-13T03:29:01.558867+00:00 / 記録時刻: 2026-11-13T03:29:01.558883+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

