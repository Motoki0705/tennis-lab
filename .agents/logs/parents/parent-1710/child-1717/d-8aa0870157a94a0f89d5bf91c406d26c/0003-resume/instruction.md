PR #1787 の統合レビュー（run parent-1710-89504094758f4489bb236ee8c81b679f）は、レビュアーの出力がPR #1786 を含む横断指摘だったため形式検査で失敗し、PRには公開されていない。親が生出力を読み、次の指摘を妥当と判断した。同じbranchで修正し、push後のCI結果まで確認してほしい。

### 指摘（major, `.agents/ops/explain_html/editor.py:266`付近）
writer出力のままdigestと意味判定を確定している。一方、#1718（PR #1786、未merge）の契約では、図slotを `render_document()` で展開し、CSPとmanaged scriptを挿入してから変更一覧を計算する。例えば、空のfigure slotと正しい図定義を持つ候補は、editorの構造検査を通っても `validate_visuals()` で拒否される。返却後に展開するとHTMLとdigestが変わるため、元の意味判定は `store.commit()` で `validation_mismatch` になる。digestだけを書き換えて判定を流用することもできない。

### 修正の方針
- 生成と意味検査の間に、信頼するcompile処理（候補Documentを受け取り、展開後のDocumentを返すか、`ContractError` で拒否する）を注入できる入口を設ける。意味検査・digest・ValidationResultは、compile後の最終候補に結び付ける。
- compile処理は呼出元（#1719 のservice、#1721 の統合）が明示的に渡す。省略時に暗黙の恒等変換へ落とさない（静かなフォールバックの禁止）。図を使わない呼出しでも、恒等を使うことを呼出側が明示する形にする。
- #1786 のコードはこのPRに取り込まない。テストでは、HTMLとassetを実際に書き換える偽compile（例: slotへ要素とassetを追加）を使い、(a) 意味検査がcompile後の内容を受け取る、(b) compileの拒否で候補がcommit不可になり現行版が変わらない、(c) 最終候補のdigestとValidationResultがstore.commitで一致する、を検証する。
- 公開済みのエラーコード・状態名は変えない（追加は可）。追加・変更した公開面（引数・型）は最終報告に列挙する。#1719 がこのbranchの上で作業しているため、その影響も書く。

目安は最大2時間。最終報告: HEAD・CI結果、変更した公開面、回帰テスト（コマンドと結果、修正前に失敗することの確認）。
