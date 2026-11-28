# parent.run-worker / 担当issue &#35;2700

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 3 件。記録時刻 2026-11-28T07:45:34.981158+00:00 〜 2026-11-28T07:45:35.114411+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

3 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

公開workerログ: 1 ターン（分母）。対応イベントあり: 1 ターン / 対応イベントなし: 0 ターン。イベントのみ: 0 ターン。和集合: 1 ターン。最終報告なし（終了・欠落の根拠あり）: 0 ターン。稼働中: 0 ターン。終了未確認: 0 ターン。最終報告あり・公開保留: 0 ターン。

対応数は公開worker3ファイルのあるターンを分母にしたものです。イベントのみのターンを加えた和集合とは分けます。対応イベントの存在は、親判断の記録・受入・網羅性を保証しません。

直近の観測（受入の判定とは別）: [結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d) rejected / machine: parent.run-worker: rejected before execution / ValueError / details in private evidence<br>[行動 7152](README.md#record-2558d761c0694c098618d39cf2d105be) 同じAIの継続 / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-a5743336848a45ddbb25f440ec6b4922"></a>[行動 7150](README.md#record-a5743336848a45ddbb25f440ec6b4922)<br>2026-11-28T07:45:34.981158+00:00（記録時刻）<br>[親 &#35;2602](../README.md) / [担当 &#35;2700](../child-2700/README.md) / [ターン 3](../child-2700/d-cc9c8a3583f040f3a8aef807f749d011/turn-0003/README.md) | [結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: requested<br>同じAIの継続 / 要求済み・実行未確認 | [記録本文](event-a5743336848a45ddbb25f440ec6b4922.json)<br>次の記録: [結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d)、[行動 7152](README.md#record-2558d761c0694c098618d39cf2d105be) |
| <a id="record-42a13232e58344bfa404c3dd12ffa50d"></a>[結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d)<br>2026-11-28T07:45:35.053651+00:00（記録時刻）<br>[親 &#35;2602](../README.md) / [担当 &#35;2700](../child-2700/README.md) / [ターン 3](../child-2700/d-cc9c8a3583f040f3a8aef807f749d011/turn-0003/README.md) | parent.run-worker: rejected before execution<br>ValueError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-42a13232e58344bfa404c3dd12ffa50d.json)<br>根拠非公開・記録不足: capture&#95;source=parent.run-worker; artifact=command response; content&#95;sha256=203cda31eaa2808d718637865b87dca52c1aae24e37eb6546210a566b3a33eea; private evidence<br>前の記録: [行動 7150](README.md#record-a5743336848a45ddbb25f440ec6b4922)、[結果 7132](../activity-614feec72e1745ce9b21908063ae062e/README.md#record-daed96ba5ecb4708a028711be6c40466)<br>次の記録: [行動 7152](README.md#record-2558d761c0694c098618d39cf2d105be) |
| <a id="record-2558d761c0694c098618d39cf2d105be"></a>[行動 7152](README.md#record-2558d761c0694c098618d39cf2d105be)<br>2026-11-28T07:45:35.114411+00:00（記録時刻）<br>[親 &#35;2602](../README.md) / [担当 &#35;2700](../child-2700/README.md) / [ターン 3](../child-2700/d-cc9c8a3583f040f3a8aef807f749d011/turn-0003/README.md) | [結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: rejected&#95;before&#95;start<br>同じAIの継続 / 起動前拒否 | [記録本文](event-2558d761c0694c098618d39cf2d105be.json)<br>前の記録: [行動 7150](README.md#record-a5743336848a45ddbb25f440ec6b4922)、[結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d) |

<details><summary>反映した記録と補助情報</summary>

- [結果 7151](README.md#record-42a13232e58344bfa404c3dd12ffa50d)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-28T07:45:35.053637+00:00 / 記録時刻: 2026-11-28T07:45:35.053651+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

