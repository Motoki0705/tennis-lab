# merge&#95;gate.merge・merge&#95;gate.evaluation / 親 &#35;1900 の直接活動

結果 → 親の判断と理由 → 実際の行動 → 次の結果を、参照で辿る表示です。

反映範囲: このHEADの記録集合 4 件。記録時刻 2026-11-15T09:58:45.439069+00:00 〜 2026-11-15T09:59:43.825700+00:00。

公開待ち: このHEADのmain反映と、まだGitに無い非公開記録は `decision-log-status` で照合してください。ここでは未公開件数や最新状態を推測しません。

[上の階層](../README.md) · [親一覧](../../README.md)

4 記録 / 判断 0 件。失敗・拒否の結果: 2 件。未判断の結果: 2 件。親の確認未記録: 2 件。判断済み未実行の計画: 0 件。記録不足の最終報告: 0 件（結果イベント単位）。最終報告の観測対象外: 2 件（結果イベント単位）。

直近の観測（受入の判定とは別）: [結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa) rejected / machine: merge&#95;gate.merge: auto&#95;merge=True / results: 4<br>[行動 2541](README.md#record-c24527cb37644f6b84c4e45e96859205) merge / 終了（成果受入とは別）

## 全記録の詳細

| 時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠 |
| --- | --- | --- | --- | --- |
| <a id="record-f80bc3e6c7c6407f90ed48b614c3e11f"></a>[行動 2535](README.md#record-f80bc3e6c7c6407f90ed48b614c3e11f)<br>2026-11-15T09:58:45.439069+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)、[結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: requested<br>merge / 要求済み・実行未確認 | [記録本文](event-f80bc3e6c7c6407f90ed48b614c3e11f.json)<br>次の記録: [結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)、[結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa)、[行動 2541](README.md#record-c24527cb37644f6b84c4e45e96859205) |
| <a id="record-9ba016d60b5e447ea0ff7ac2a88e61ef"></a>[結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)<br>2026-11-15T09:59:42.273747+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | merge&#95;gate.evaluation: pr=2087<br>判断: blocked<br>操作: blocked<br>状態: 記録値なし<br>理由: mergeable&#95;unknown<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-9ba016d60b5e447ea0ff7ac2a88e61ef.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=evaluation; content&#95;sha256=bd79476e69a206ae9240a5305481401cf06c8af28b84e568b3a2f6429c3cb937; private evidence<br>前の記録: [行動 2535](README.md#record-f80bc3e6c7c6407f90ed48b614c3e11f)<br>次の記録: [結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa)、[行動 2541](README.md#record-c24527cb37644f6b84c4e45e96859205) |
| <a id="record-3bb5ca634f104a749505eb85e0793aaa"></a>[結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa)<br>2026-11-15T09:59:43.146325+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | merge&#95;gate.merge: auto&#95;merge=True<br>results: 4<br>rejected / machine<br>親の確認未記録<br>最終報告の観測対象外（記録された理由による）<br>前回との差: コード: 未確認 / 実行条件: 未確認 / 指示: 未確認 / 要件: 未確認 / 確認済み範囲: 未確認 | 未判断 | 判断待ち | [記録本文](event-3bb5ca634f104a749505eb85e0793aaa.json)<br>根拠非公開・記録不足: capture&#95;source=merge&#95;gate.merge; artifact=command response; content&#95;sha256=19a21dc77b059e39c69a1c9257e1883201af2eb13373b24659756dee53d6b706; private evidence<br>前の記録: [行動 2535](README.md#record-f80bc3e6c7c6407f90ed48b614c3e11f)、[結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)<br>次の記録: [行動 2541](README.md#record-c24527cb37644f6b84c4e45e96859205) |
| <a id="record-c24527cb37644f6b84c4e45e96859205"></a>[行動 2541](README.md#record-c24527cb37644f6b84c4e45e96859205)<br>2026-11-15T09:59:43.825700+00:00（記録時刻）<br>[親 &#35;1900](../README.md) / 親の直接活動 | [結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)、[結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa) | 記録不足・判断: parent decision not recorded for this operation | merge&#95;gate.merge: ended<br>merge / 終了（成果受入とは別） | [記録本文](event-c24527cb37644f6b84c4e45e96859205.json)<br>前の記録: [行動 2535](README.md#record-f80bc3e6c7c6407f90ed48b614c3e11f)、[結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)、[結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa) |

<details><summary>反映した記録と補助情報</summary>

- [結果 2539](README.md#record-9ba016d60b5e447ea0ff7ac2a88e61ef)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-15T09:59:42.273729+00:00 / 記録時刻: 2026-11-15T09:59:42.273747+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source
- [結果 2540](README.md#record-3bb5ca634f104a749505eb85e0793aaa)<br>時刻: 発生時刻: 未取得: source occurrence time unavailable / 受信時刻: 観測: 2026-11-15T09:59:43.146303+00:00 / 記録時刻: 2026-11-15T09:59:43.146325+00:00<br>補助情報（欠測と累積値を保持、加算なし）: 実効モデル: 未取得: not reported by this source / 実行秒数: 未取得: not reported by this source / 選択モデル: 未取得: not reported by this source / 使用量: 未取得: not reported by this source / 待機秒数: 未取得: not reported by this source

</details>

