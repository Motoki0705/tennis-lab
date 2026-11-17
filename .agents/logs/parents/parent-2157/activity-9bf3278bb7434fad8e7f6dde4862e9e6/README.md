# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;2157 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-16T02:29:08.963425+00:00 〜 2026-11-16T02:29:41.284110+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 2 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec) rejected / machine: merge&#95;gate.merge: auto&#95;merge=True / results: 2<br>[行動 2875](README.md#record-cc14c0db10e64dd39e71196ba408833a) merge / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-0a5a30a95cd740098671024cf08be9bb"></a>[行動 2872](README.md#record-0a5a30a95cd740098671024cf08be9bb)<br>2026-11-16T02:29:08.963425+00:00（記録時刻）<br>[親 &#35;2157](../README.md) / 親の直接活動 | [結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)、[結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-0a5a30a95cd740098671024cf08be9bb.json)<br>次の記録: [結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)、[結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec)、[行動 2875](README.md#record-cc14c0db10e64dd39e71196ba408833a) |
| <a id="record-1927ccd5009a49c6a9433e9ae7ee2d86"></a>[結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)<br>2026-11-16T02:29:41.029441+00:00（記録時刻）<br>[親 &#35;2157](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=2178<br>判断: blocked<br>操作: blocked<br>状態: 記録値なし<br>理由: pr&#95;2178&#95;unverified&#95;head、unresolved&#95;findings、review&#95;missing<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-1927ccd5009a49c6a9433e9ae7ee2d86.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=0862a0d569d9b8c1425d8f731ad1b7d39811052df029ae932cbe80f4cdf67d20; private evidence<br>前の記録: [行動 2872](README.md#record-0a5a30a95cd740098671024cf08be9bb)<br>次の記録: [結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec)、[行動 2875](README.md#record-cc14c0db10e64dd39e71196ba408833a) |
| <a id="record-8263a884658641e4ae257e95ececb5ec"></a>[結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec)<br>2026-11-16T02:29:41.155254+00:00（記録時刻）<br>[親 &#35;2157](../README.md) / 親の直接活動 | merge&#95;gate.merge: auto&#95;merge=True<br>results: 2<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-8263a884658641e4ae257e95ececb5ec.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=1fd127c61f2db4e740d401abf4d34b6aee8e2f5d30cf7f3bc2ed0e91388caf75; private evidence<br>前の記録: [行動 2872](README.md#record-0a5a30a95cd740098671024cf08be9bb)、[結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)<br>次の記録: [行動 2875](README.md#record-cc14c0db10e64dd39e71196ba408833a) |
| <a id="record-cc14c0db10e64dd39e71196ba408833a"></a>[行動 2875](README.md#record-cc14c0db10e64dd39e71196ba408833a)<br>2026-11-16T02:29:41.284110+00:00（記録時刻）<br>[親 &#35;2157](../README.md) / 親の直接活動 | [結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)、[結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: ended<br>merge / 終了（成果受入とは別） | [記録本文](event-cc14c0db10e64dd39e71196ba408833a.json)<br>前の記録: [行動 2872](README.md#record-0a5a30a95cd740098671024cf08be9bb)、[結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)、[結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec) |

<details><summary>反映した記録と補助情報</summary>

- [結果 2873](README.md#record-1927ccd5009a49c6a9433e9ae7ee2d86)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T02:29:41.029428+00:00 / 記録時刻: 2026-11-16T02:29:41.029441+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 2874](README.md#record-8263a884658641e4ae257e95ececb5ec)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T02:29:41.155240+00:00 / 記録時刻: 2026-11-16T02:29:41.155254+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

