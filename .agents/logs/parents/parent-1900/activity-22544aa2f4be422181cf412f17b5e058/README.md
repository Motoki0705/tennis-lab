# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;1900 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-15T05:59:09.393734+00:00 〜 2026-11-15T06:29:07.227989+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9) rejected / machine: merge&#95;gate.merge: rejected before execution / GitHubHTTPError / details in private evidence<br>[行動 1932](README.md#record-2c16497154204a5fb6de0fde7c9bd83a) merge / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-e5c9c7a3026f4113a94eca7932303cc6"></a>[行動 1929](README.md#record-e5c9c7a3026f4113a94eca7932303cc6)<br>2026-11-15T05:59:09.393734+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)、[結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-e5c9c7a3026f4113a94eca7932303cc6.json)<br>次の記録: [結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)、[結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9)、[行動 1932](README.md#record-2c16497154204a5fb6de0fde7c9bd83a) |
| <a id="record-06f61b17ee384f7fb7b54992fc764c80"></a>[結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)<br>2026-11-15T06:28:13.634018+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=2087<br>判断: merge<br>操作: eligible<br>状態: 記録値なし<br>理由: なし（空の一覧）<br>success / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-06f61b17ee384f7fb7b54992fc764c80.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=0c591b60779863098aa893a2c8347d76f53ba271fcf1ec51b99d4dc730db1dcd; private evidence<br>前の記録: [行動 1929](README.md#record-e5c9c7a3026f4113a94eca7932303cc6)<br>次の記録: [結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9)、[行動 1932](README.md#record-2c16497154204a5fb6de0fde7c9bd83a) |
| <a id="record-60cf4186f55640919cd4f85bcf57e0f9"></a>[結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9)<br>2026-11-15T06:29:06.522991+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | merge&#95;gate.merge: rejected before execution<br>GitHubHTTPError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-60cf4186f55640919cd4f85bcf57e0f9.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=c9c041b6951475b214eb2d6a6bbcb6fec7264f2842d018c260d53f6835b6fbbb; private evidence<br>前の記録: [行動 1929](README.md#record-e5c9c7a3026f4113a94eca7932303cc6)、[結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)<br>次の記録: [行動 1932](README.md#record-2c16497154204a5fb6de0fde7c9bd83a) |
| <a id="record-2c16497154204a5fb6de0fde7c9bd83a"></a>[行動 1932](README.md#record-2c16497154204a5fb6de0fde7c9bd83a)<br>2026-11-15T06:29:07.227989+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)、[結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: rejected&#95;before&#95;start<br>merge / 起動前拒否 | [記録本文](event-2c16497154204a5fb6de0fde7c9bd83a.json)<br>前の記録: [行動 1929](README.md#record-e5c9c7a3026f4113a94eca7932303cc6)、[結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)、[結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9) |

<details><summary>反映した記録と補助情報</summary>

- [結果 1930](README.md#record-06f61b17ee384f7fb7b54992fc764c80)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-15T06:28:13.633954+00:00 / 記録時刻: 2026-11-15T06:28:13.634018+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 1931](README.md#record-60cf4186f55640919cd4f85bcf57e0f9)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-15T06:29:06.522971+00:00 / 記録時刻: 2026-11-15T06:29:06.522991+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

