# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;1734 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-16T00:48:58.928678+00:00 〜 2026-11-16T00:49:41.964080+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 2 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65) rejected / machine: merge&#95;gate.merge: auto&#95;merge=True / results: 2<br>[行動 2906](README.md#record-6e10d6ee17ce426ba5a734005c26cd68) merge / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-61818d25e16b44dca3af8b74fc43ce24"></a>[行動 2903](README.md#record-61818d25e16b44dca3af8b74fc43ce24)<br>2026-11-16T00:48:58.928678+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | [結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)、[結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-61818d25e16b44dca3af8b74fc43ce24.json)<br>次の記録: [結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)、[結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65)、[行動 2906](README.md#record-6e10d6ee17ce426ba5a734005c26cd68) |
| <a id="record-7f85960b19f14c209f476f03fbb5ac7d"></a>[結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)<br>2026-11-16T00:49:41.667638+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=2153<br>判断: blocked<br>操作: blocked<br>状態: 記録値なし<br>理由: parent&#95;review&#95;not&#95;confirmed<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-7f85960b19f14c209f476f03fbb5ac7d.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=a5ef0c6fe15622b47a3b4d5228719f377b9ea4f153d5e744b4a918a7983126e1; private evidence<br>前の記録: [行動 2903](README.md#record-61818d25e16b44dca3af8b74fc43ce24)<br>次の記録: [結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65)、[行動 2906](README.md#record-6e10d6ee17ce426ba5a734005c26cd68) |
| <a id="record-b1af2b47abb04cfdadc1a76adca6ee65"></a>[結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65)<br>2026-11-16T00:49:41.805188+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | merge&#95;gate.merge: auto&#95;merge=True<br>results: 2<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-b1af2b47abb04cfdadc1a76adca6ee65.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=05ed8a8bd8cf5b8cda0c5d0dcbf4c945be78c5122a9989a7e35abfd36112b406; private evidence<br>前の記録: [行動 2903](README.md#record-61818d25e16b44dca3af8b74fc43ce24)、[結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)<br>次の記録: [行動 2906](README.md#record-6e10d6ee17ce426ba5a734005c26cd68) |
| <a id="record-6e10d6ee17ce426ba5a734005c26cd68"></a>[行動 2906](README.md#record-6e10d6ee17ce426ba5a734005c26cd68)<br>2026-11-16T00:49:41.964080+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | [結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)、[結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: ended<br>merge / 終了（成果受入とは別） | [記録本文](event-6e10d6ee17ce426ba5a734005c26cd68.json)<br>前の記録: [行動 2903](README.md#record-61818d25e16b44dca3af8b74fc43ce24)、[結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)、[結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65) |

<details><summary>反映した記録と補助情報</summary>

- [結果 2904](README.md#record-7f85960b19f14c209f476f03fbb5ac7d)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T00:49:41.667627+00:00 / 記録時刻: 2026-11-16T00:49:41.667638+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 2905](README.md#record-b1af2b47abb04cfdadc1a76adca6ee65)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T00:49:41.805176+00:00 / 記録時刻: 2026-11-16T00:49:41.805188+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

