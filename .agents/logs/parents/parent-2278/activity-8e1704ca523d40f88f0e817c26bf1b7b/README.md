# parent.run-worker / 担当issue &#35;2355

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 3 件。記録時刻 2026-11-21T23:59:27.769264+00:00 〜 2026-11-21T23:59:44.151945+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

3 記録 / 判断 0 件。失敗・拒否の結果: 1 件。未判断の結果: 1 件。親の確認未記録: 1 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 1 件（結果イベント単位）。

公開workerログ: 0 ターン（分母）。対応イベントあり: 0 ターン / 対応イベントなし: 0 ターン。イベントのみ: 1 ターン。和集合: 1 ターン。最終報告なし（終了・欠落の根拠あり）: 0 ターン。稼働中: 0 ターン。終了未確認: 0 ターン。最終報告あり・公開保留: 0 ターン。

対応数は公開worker3ファイルのあるターンを分母にしたものです。イベントのみのターンを加えた和集合とは分けます。対応イベントの存在は、親判断の記録・受入・網羅性を保証しません。

直近の観測（受入の判定とは別）: [結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa) rejected / machine: parent.run-worker: rejected before execution / ValueError / details in private evidence<br>[行動 4409](README.md#record-a539c24a272845cc9909f80d28e50904) 新しいAIの起動 / 起動前拒否

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-78784b4d67fb458b91f372e30b866483"></a>[行動 4407](README.md#record-78784b4d67fb458b91f372e30b866483)<br>2026-11-21T23:59:27.769264+00:00（記録時刻）<br>[親 &#35;2278](../README.md) / [担当 &#35;2355](../child-2355/README.md) / [ターン 3](../child-2355/d-f3942ae96ead4458a93f577be6fea2df/turn-0003/README.md) | [結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: requested<br>新しいAIの起動 / 要求済み・実行未確認 | [記録本文](event-78784b4d67fb458b91f372e30b866483.json)<br>次の記録: [結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa)、[行動 4409](README.md#record-a539c24a272845cc9909f80d28e50904) |
| <a id="record-15ca81c225d846c68e79731a943c25aa"></a>[結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa)<br>2026-11-21T23:59:43.862303+00:00（記録時刻）<br>[親 &#35;2278](../README.md) / [担当 &#35;2355](../child-2355/README.md) / [ターン 3](../child-2355/d-f3942ae96ead4458a93f577be6fea2df/turn-0003/README.md) | parent.run-worker: rejected before execution<br>ValueError<br>details in private evidence<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-15ca81c225d846c68e79731a943c25aa.json)<br>根拠非公開・記録不足: capture&#95;source=parent.run-worker; artifact=command response; content&#95;sha256=8f0c824dbf5f42f3d70127ee710fdcca54e4603ece314c8780ebf36840f8f7ff; private evidence<br>前の記録: [行動 4407](README.md#record-78784b4d67fb458b91f372e30b866483)<br>次の記録: [行動 4409](README.md#record-a539c24a272845cc9909f80d28e50904) |
| <a id="record-a539c24a272845cc9909f80d28e50904"></a>[行動 4409](README.md#record-a539c24a272845cc9909f80d28e50904)<br>2026-11-21T23:59:44.151945+00:00（記録時刻）<br>[親 &#35;2278](../README.md) / [担当 &#35;2355](../child-2355/README.md) / [ターン 3](../child-2355/d-f3942ae96ead4458a93f577be6fea2df/turn-0003/README.md) | [結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa) | 記録不足・判断: parent decision not recorded for this operation | parent.run-worker: rejected&#95;before&#95;start<br>新しいAIの起動 / 起動前拒否 | [記録本文](event-a539c24a272845cc9909f80d28e50904.json)<br>前の記録: [行動 4407](README.md#record-78784b4d67fb458b91f372e30b866483)、[結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa) |

<details><summary>反映した記録と補助情報</summary>

- [結果 4408](README.md#record-15ca81c225d846c68e79731a943c25aa)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-21T23:59:43.862287+00:00 / 記録時刻: 2026-11-21T23:59:43.862303+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

