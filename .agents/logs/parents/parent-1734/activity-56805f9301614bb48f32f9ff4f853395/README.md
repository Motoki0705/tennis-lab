# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;1734 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-16T00:48:04.675400+00:00 〜 2026-11-16T00:48:49.064284+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 2 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af) rejected / machine: merge&#95;gate.merge: auto&#95;merge=True / results: 2<br>[行動 2898](README.md#record-dc222e833bc74bc28ed7fef3a281ca1d) merge / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-504d8fdcfe274c93a1482982fc7df00a"></a>[行動 2895](README.md#record-504d8fdcfe274c93a1482982fc7df00a)<br>2026-11-16T00:48:04.675400+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | [結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)、[結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-504d8fdcfe274c93a1482982fc7df00a.json)<br>次の記録: [結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)、[結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af)、[行動 2898](README.md#record-dc222e833bc74bc28ed7fef3a281ca1d) |
| <a id="record-5554f2d56240417c8a85a96a5ae6ef5e"></a>[結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)<br>2026-11-16T00:48:48.772978+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=2153<br>判断: blocked<br>操作: blocked<br>状態: 記録値なし<br>理由: parent&#95;review&#95;not&#95;confirmed<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-5554f2d56240417c8a85a96a5ae6ef5e.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=a5ef0c6fe15622b47a3b4d5228719f377b9ea4f153d5e744b4a918a7983126e1; private evidence<br>前の記録: [行動 2895](README.md#record-504d8fdcfe274c93a1482982fc7df00a)<br>次の記録: [結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af)、[行動 2898](README.md#record-dc222e833bc74bc28ed7fef3a281ca1d) |
| <a id="record-20fd8e20d5a947fc91842895129e22af"></a>[結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af)<br>2026-11-16T00:48:48.909168+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | merge&#95;gate.merge: auto&#95;merge=True<br>results: 2<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-20fd8e20d5a947fc91842895129e22af.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=05ed8a8bd8cf5b8cda0c5d0dcbf4c945be78c5122a9989a7e35abfd36112b406; private evidence<br>前の記録: [行動 2895](README.md#record-504d8fdcfe274c93a1482982fc7df00a)、[結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)<br>次の記録: [行動 2898](README.md#record-dc222e833bc74bc28ed7fef3a281ca1d) |
| <a id="record-dc222e833bc74bc28ed7fef3a281ca1d"></a>[行動 2898](README.md#record-dc222e833bc74bc28ed7fef3a281ca1d)<br>2026-11-16T00:48:49.064284+00:00（記録時刻）<br>[親 &#35;1734](../README.md) / 親の直接活動 | [結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)、[結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: ended<br>merge / 終了（成果受入とは別） | [記録本文](event-dc222e833bc74bc28ed7fef3a281ca1d.json)<br>前の記録: [行動 2895](README.md#record-504d8fdcfe274c93a1482982fc7df00a)、[結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)、[結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af) |

<details><summary>反映した記録と補助情報</summary>

- [結果 2896](README.md#record-5554f2d56240417c8a85a96a5ae6ef5e)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T00:48:48.772965+00:00 / 記録時刻: 2026-11-16T00:48:48.772978+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 2897](README.md#record-20fd8e20d5a947fc91842895129e22af)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-16T00:48:48.909156+00:00 / 記録時刻: 2026-11-16T00:48:48.909168+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

