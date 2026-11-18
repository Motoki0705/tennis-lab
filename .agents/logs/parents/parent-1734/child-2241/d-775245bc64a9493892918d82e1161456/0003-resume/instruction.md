# PR #2244 再レビューのmajor指摘への対応（親 #1734 から #2241 へ）

再レビュー（gpt-6.2-astra / xhigh）で、前回の5件への対応は受け入れられた。新たにmajor指摘が3件出た。3件とも妥当と判断する。直してください。

## 1. driftの厳格設定で棄却されるclipが、別clipの共有判定に寄与する

静的な推論。`video_court_side/prepare.py` の `calibration_digest` と `artifact.py` の `check_settings` は共有コートとdrift診断のdigestを照合しますが、証拠に使う各clipが現在の `court_kp.video_shared.unobservable_policy` で採用可能かを確認しません。例えばAはdrift検査に合格し、Bは `drift_unobservable` のため `warn` でのみ校正できた場合、`fail` に切り替えてAを実行しても、Bのball観測を含む判定をbuild・loadできます。Bの観測がAの向きを決め得ます。共有判定を採用する前に、選択した全証拠clipに現在のコート採否規則を適用し、棄却対象を含む判定は理由付きで拒否する必要があります。

親の判断: **直す。** 共有判定を採用する前（build・loadとも）に、証拠に使う全clipへ、現在のコートの採否規則（unobservable_policy）を適用する。棄却対象のclipを含む判定は、理由を付けて拒否する。

## 2. 共有判定のbuildが、別clipの入力配列が破損していてもcache hitする

静的な推論。`video_court_side/prepare.py` のbuild再利用分岐は `pin.assert_current()` を呼びますが、`inputs.py` の `from_store` が検証するのはscene index、descriptor、依存参照までです。実際の `.npy` のchecksumは `assemble()` で初めて検証されます。A・Bでbuildした後、Bのball座標配列だけが欠損・破損しても、Aの再buildは成功し、datasetの完成marker用identityも取得できます。同じ入力での新規buildは失敗します。判定の再採点は省略したまま、build再利用時にも固定した校正・ball入力の実体を検証する必要があります。

親の判断: **直す。** buildで再利用するときも、再採点はしなくてよい。ただし、固定した校正とballの入力の実体（配列のchecksum）は検証する。指摘の手順（Bの配列を破損させてからAを再build）を、テストにする。

## 3. court_sideの明示loadで共有digestと矛盾判定を迂回できる

静的な推論。`definition.py` は共有digestとclipの採否をnode設定に入れますが、`execution_plan.py` の `_decide` がload時に実行identityを照合するのは `origin=component` の場合だけです。既存の `bind_import`・`publish_import` APIで、同じ校正・ball入力参照を持つ `origin=import` のcourt_sideを公開したstoreでは、`execution.court_side=load` がその値を採用します。`SharedCourtSideModule.process` は通らず、共有診断が `court_side_video_shared_conflict` でも別の向きが下流に渡ります。共有モードのloadでは、共有digestと方針に一致する共有nodeの出力だけを受理し、失敗判定のclipがloadを通過しないことを検証する必要があります。

親の判断: **直す。** 共有モードでloadするときは、共有digestと方針が一致する、共有node（origin=component）の出力だけを受理する。import由来の出力と、失敗と判定されたclipは拒否する。

## 再レビューを繰り返さないために
指摘3件は、どれも「共有判定を採用するときに、どこかの経路で照合が抜ける」という同じ種類の問題である。修正の前に、共有判定がclipへ届く全経路を洗い出し、各経路で何を照合するかの表をPR本文に載せる。経路には、build・cacheの再利用・load・import・overrides・公開の再開・dataset marker、ほかに見つかった経路を含める。照合する項目は、request、入力の実体、コートの採否、共有digest、矛盾した判定とする。漏れた経路をまとめて塞ぐ。

## 共通の条件
- 静かなフォールバックを作らない。既定（clip）の結果は変えない。
- meiji_4camの判定（[F,F,T,T]、margin 0.267333、一致51・警告18・矛盾0）を、保存済みの成果物を使ってCPUで再現する。
- CPUの関連テスト、pre-commit、CIを通し、PR本文に追記する。merge・closeは親が行う。

