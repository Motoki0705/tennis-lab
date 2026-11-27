# parent.merge-continue / 親 &#35;2627 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-27T02:46:47.307624+00:00 〜 2026-11-27T02:46:47.812097+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017) failure / machine: parent.merge-continue: post-start failure / ValueError / details in private evidence<br>[行動 6675](README.md#record-755d6a4cbe0840d88d9578ae9f7dfed4) 再開 / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-22f41d10fb9a4f8ba3da04b4e0b10906"></a>[行動 6672](README.md#record-22f41d10fb9a4f8ba3da04b4e0b10906)<br>2026-11-27T02:46:47.307624+00:00（記録時刻）<br>[親 &#35;2627](../README.md) / 親の直接活動 | 結果記録なし | 記録不足・判断: parent decision not recorded for this operation | parent.merge-continue: requested<br>再開 / 要求済み・実行未確認 | [記録本文](event-22f41d10fb9a4f8ba3da04b4e0b10906.json)<br>次の記録: [行動 6673](README.md#record-d7c54cc54085446cacff72798b6ab8fc) |
| <a id="record-d7c54cc54085446cacff72798b6ab8fc"></a>[行動 6673](README.md#record-d7c54cc54085446cacff72798b6ab8fc)<br>2026-11-27T02:46:47.465901+00:00（記録時刻）<br>[親 &#35;2627](../README.md) / 親の直接活動 | [結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017) | 記録不足・判断: parent decision not recorded for this operation | parent handler entered; no AI launch implied<br>再開 / 実行中 | [記録本文](event-d7c54cc54085446cacff72798b6ab8fc.json)<br>前の記録: [行動 6672](README.md#record-22f41d10fb9a4f8ba3da04b4e0b10906)<br>次の記録: [結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017)、[行動 6675](README.md#record-755d6a4cbe0840d88d9578ae9f7dfed4) |
| <a id="record-14d674afa80b42099c43ee1c72426017"></a>[結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017)<br>2026-11-27T02:46:47.660947+00:00（記録時刻）<br>[親 &#35;2627](../README.md) / 親の直接活動 | parent.merge-continue: post-start failure<br>ValueError<br>details in private evidence<br>failure / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-14d674afa80b42099c43ee1c72426017.json)<br>根拠非公開・記録不足: capture&#95;source=parent.merge-continue; artifact=command response; content&#95;sha256=86e89aa9cb693bb0b68ea649b9e093e9b466c6527a2b460f9ff076b83f3da107; private evidence<br>前の記録: [行動 6673](README.md#record-d7c54cc54085446cacff72798b6ab8fc)<br>次の記録: [行動 6675](README.md#record-755d6a4cbe0840d88d9578ae9f7dfed4) |
| <a id="record-755d6a4cbe0840d88d9578ae9f7dfed4"></a>[行動 6675](README.md#record-755d6a4cbe0840d88d9578ae9f7dfed4)<br>2026-11-27T02:46:47.812097+00:00（記録時刻）<br>[親 &#35;2627](../README.md) / 親の直接活動 | [結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017) | 記録不足・判断: parent decision not recorded for this operation | parent.merge-continue: ended<br>再開 / 終了（成果受入とは別） | [記録本文](event-755d6a4cbe0840d88d9578ae9f7dfed4.json)<br>前の記録: [行動 6673](README.md#record-d7c54cc54085446cacff72798b6ab8fc)、[結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017) |

<details><summary>反映した記録と補助情報</summary>

- [結果 6674](README.md#record-14d674afa80b42099c43ee1c72426017)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-27T02:46:47.660933+00:00 / 記録時刻: 2026-11-27T02:46:47.660947+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

