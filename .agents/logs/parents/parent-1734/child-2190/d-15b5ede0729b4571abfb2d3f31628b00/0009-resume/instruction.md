## 親の指示: 正式レビューのmajor 1件への対応（PR #2200）

レビュー（gpt-6-astra、xhigh）の指摘は次のとおりです。
> 既定の設定（legacy/model）のままでも、追加した方式・品質・paintの設定がcourtのidentityに入る（definition.py:142）。さらに、src全体の実装fingerprint（orchestrator.py:50）も変わる。そのため、既存のclipを同じ設定で再実行すると、保存済みのcourt成果物がcache hitせず、全cameraで再計算になる。load指定では、identityの不一致で停止する（execution_plan.py:201）。この意図と影響がPRに説明されていない。

### 親の判断
**既定のidentityを維持する対応はしません。** 既存のcourt cacheが無効になることを、意図した変更として明記します。理由は次の3つです。
- meiji_4camの全件処理はまだ行っておらず、無効になるcacheは計測・評価用のものだけです。
- identityに方式と設定を含めるのは、#2191で異なる版が混ざるのを防ぐための契約です。
- load指定がidentityの不一致で停止するのは、明示的な失敗なので、静かなフォールバックではありません。

### やること
1. **影響を実測で確認する（CPUのみ、GPU不要）。**
   - 既定の設定で、court nodeのidentityがmainと異なることを、小さな実例またはテストで確かめる。
   - src全体の実装fingerprintの変化が、court以外のcomponentに波及するかを、コードで確認する。src全体のfingerprintが、srcのどの変更でも変わる既存の仕組みなら、このPRに固有の影響ではないことも書く。
   - load指定のときに、どのエラーで停止するかを確認する。
2. **PR #2200の本文に「cacheへの影響」の節を追加する。**
   - 無効になる成果物の範囲（court、および波及するcomponent）。
   - 再計算が必要になること。
   - load指定では停止すること（その理由とエラー）。
   - これが意図した変更であること。
3. 必要なら、knowledge 75またはcourt_detection READMEに1〜2文を追記します。二重管理にならない場所を選んでください。
4. コードを変えた場合は、関連テストとCIの成功まで確認してください。本文の変更だけなら、commitは不要です。

merge・issue closeは行いません。
