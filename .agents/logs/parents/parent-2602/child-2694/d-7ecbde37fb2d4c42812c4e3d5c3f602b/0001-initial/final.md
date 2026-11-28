#2694 の棚卸しと実表示比較を完了しました。Wiki の公開コミットは `b054562b72e69fa122edecaf4966a8465529329f` です。差分は一時ページ3件とアセット44件で、比較ページは選定に使える状態です。[公開比較ページ](https://github.com/Motoki0705/tennis-lab/wiki/_Preview-Wiki-Style-Comparison)

- **棚卸し**では本文38ページとサイドバー・フッターを確認しました。Markdown表178件のうち、冒頭定型表は Home を除く37件です。定型戻りリンク30件、定型基準日6件を確認し、ページごとの出典移動先を記録しました。改訂案は Tier 1 が Meiji と人物比較5ページ、Tier 2 が21ページ、Tier 3 が10ページです。
- **公開表示**では Markdown・HTML・AsciiDoc・reStructuredText の各2表が正本の数値・状態と一致しました。Chromium の通常幅・390px幅、ライト・ダークの4条件で各50画像を確認し、SVGのraw・Camo配信、折りたたみ、Mermaidも表示できました。`picture` はブラウザの配色に追従し、テーマフラグ付きの相対画像は両方表示されました。
- **採用案**は、表に **Typst**、グラフに **Matplotlib**、画像に背景色を明示した **アウトライン化SVG** と互換用PNGです。390px幅では検出の全列表が約5px相当まで縮み、4列に分けると約11px相当を保てました。Markdown表は、狭幅で横スクロールせず読める2〜4列の索引・対応表・小規模な比較に残す案です。

初回のHTML表SVGには日本語見出しとモデル名の欠落がありました。Poppler/Cairoに切り替えて修正し、最終版のSVG・PNG 22組を照合しました。別ブラウザエンジン、モバイル実機、ログイン済み設定画面からのテーマ切替は未確認です。

成果物は `tmp/wiki-2694/` にあります。

- [棚卸し・全表の用途分類](https://github.com/Motoki0705/tennis-lab/blob/8d629a35289f475e514291167435c333876afdd2/tmp/wiki-2694/inventory.md) · [集計CSV](https://github.com/Motoki0705/tennis-lab/blob/8d629a35289f475e514291167435c333876afdd2/tmp/wiki-2694/inventory-tables.csv)
- [改訂設計と採用案](https://github.com/Motoki0705/tennis-lab/blob/8d629a35289f475e514291167435c333876afdd2/tmp/wiki-2694/design-and-comparison.md)
- [実表示・公開前検査の記録](https://github.com/Motoki0705/tennis-lab/blob/8d629a35289f475e514291167435c333876afdd2/tmp/wiki-2694/render-evidence.md)
- [試作スクリプトと再生成手順](https://github.com/Motoki0705/tennis-lab/blob/8d629a35289f475e514291167435c333876afdd2/tmp/wiki-2694/prototypes/README.md)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}