PR #1786 の統合レビュー（run parent-1710-351be80884b543ae9db92564ad577513、PRへ公開済み）が request_changes でした。3件を同じbranchで修正し、push後のCI結果まで確認してください。

1. pr-1786-1（blocker, visuals.py:413付近）: Python HTMLParserとブラウザでコメント解析が食い違う（`<!-->`、`<!--->`、`--!>`、閉じないコメント、`<!` で始まる不正宣言など）。食い違いを起こしうる構文は、許容して解釈を合わせるのではなく明示的に拒否する（静かな補正をしない）。この問題はmain上の `.agents/ops/explain_html/document.py`（HTMLIndex、#1716でmerge済み）にも及ぶ。共通の解析で拒否し、document検査とvisual検査の両方で同じ規則が効くようにする。公開済みの型名・エラーコードは変えない（既存の `invalid_document` 等を使う）。CSPの挿入位置も、head内の最初のscript・style・リソース参照より前になることを確かめる。ブラウザ回帰（実Chromiumで、拒否されること・該当入力がexportに出ないこと）を追加する。
2. pr-1786-2（major, visuals.py:397付近）: inline styleの `image-set(...)`、`-webkit-image-set(...)` の文字列URLなど、`url(` 以外のURL記法を拒否する。style要素内も同じ規則にする。この検査も document.py 側の参照所有の規則と一致させ、既存の参照・変更申告テストにこの境界を追加する。
3. pr-1786-3（major, renderers/visuals-v1.js:88付近）: `data-managed-graph`・`data-definition` など描画用の属性を予約し、正規のfigure slot外では拒否する。実行時のrendererも、検証済みでfigure slot内の描画対象だけを処理する。1つの不正な要素で他のgraphの初期化が止まらないことをNode/ブラウザで検証する。

renderer bytesを変える場合は、managed rendererの版（v1→v2など）と旧版の再表示の契約を守る（既存の版に保存されたv1が壊れないこと）。
修正後、HEAD・CI結果・各指摘への対応と回帰テスト（コマンドと結果、修正前に失敗することの確認）を最終報告に書く。#1719 が並行して #1787 の上で作業しているため、document.py の公開関数の引数・戻り値は変えない。目安は最大3時間。
