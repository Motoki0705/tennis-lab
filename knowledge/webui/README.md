# Research Library — Web UI

タスクごとの実験・知識グラフ・関連論文・横断summaryを閲覧するNext.js / React Flow UI。
保存形式は [knowledge/README.md](../README.md) が正本。

![研究ライブラリの実験一覧](assets/library.png)

![Papersの一次資料リンク・要約・採録理由](assets/papers.png)

```bash
cd knowledge/webui
npm ci
npm run dev                     # localhost:3000
npm run build && npm start      # production
```

- タスクはノードと論文metadataから自動発見。新しいトピックにも対応。
- 実験一覧: 実験日順、6桁連番、全文/設定検索、status/provider/tag/issue/論文の絞り込み、30件ずつのページ表示。
- グラフ: 前提から後続への有向線、グループの囲み、任意の関連線、zoom/minimap。
- 詳細: 考察、実測metrics、config、曲線、再現bundleのパス、前提・後続・関連実験、論文参照。Escapeで閉じる。
- 実験比較: 最大4件の条件・metricsを並置しCSV出力。異なる評価条件に自動の優劣を付けない。
- Papers: 論文情報・要約・採録理由・読解ノート・一次資料URL・参照実験への逆引き。
- 引用探索: Papersから保存済み探索の起点を選び、役割付きマップと役割別ランキングを閲覧。件数から引用元ごとの判断理由・根拠位置へ移り、外部IDで対応した登録済みPapersの詳細を開ける。問題箇所は役割の絞り込み中も表示する。
- 研究サマリー: `summary.md` を表示し、根拠ノードへ遷移。
- URLにタスク・検索・フィルタ・表示・選択ノードを保持。`/?node=<id>`、`/?paper=<id>` で直接開ける。実験の論文絞り込みには別キー `paperFilter` を使う。
- 引用探索のURLは `citationSearch`（探索ID）・`citationRole`（役割）・`citationPaper`（論文文書ID）・`citationView`（`map` / `ranking`）。例: `/?citationSearch=search-example&citationRole=comparison`。探索やPapers登録はCLIで行い、Web UIは読み取り専用。
- モバイル対応、キーボード操作、MarkdownのHTMLサニタイズ。

`KNOWLEDGE_DIR` でknowledge rootを変更可能（既定はアプリの親）。既存の
`KNOWLEDGE_NODES_DIR` / `KNOWLEDGE_RUNS_DIR` もサブディレクトリ指定に使える。
リクエストごとに正本Markdownを読むため、編集後の再読み込みで反映される。
読み込み失敗や不正なID・連番を空のライブラリに置き換えず、エラーとして表示する。
Papersは文書だけを保持し、本文は一次資料URLから閲覧する。
`citation_map/` が無い、または保存済み探索が0件なら「探索なし」を表示する。
不正な引用文書、参照や世代の不整合、外部IDの矛盾、読み込み失敗はエラーとして表示する。

## 検証・コードの入口

```bash
npm run lint
npm run test
npm run build
npx playwright install chromium  # 初回のみ
npm run test:e2e                 # 専用dev serverを起動
```

e2eは既存ライブラリ（3137）と、共有固定例を複製した専用knowledge root（3138）を使う。
架空の引用論文や登録済みPapersは `.citation-e2e/` にだけ作成する。固定例で開発確認する場合:

```bash
npx tsx tests/citation-fixture.ts
KNOWLEDGE_DIR="$PWD/.citation-e2e" npm run dev
```

![固定例の引用マップ](assets/citation-map.png)

- `src/lib/{content,nodes}.ts`: Markdown/Papers/summary読込とリンク解決。
- `src/lib/citation-documents.ts`: 引用frontmatter・文書間整合性の検証と読込、外部IDによるPapers結合。
- `src/lib/citations.ts`: 探索範囲の引用関係と、異なる引用元数による集計。
- `src/lib/explorer.ts`: 絞り込み・並び順・CSV。
- `src/components/ResearchExplorer.tsx`: ライブラリの状態と各ビュー。
- `src/components/{GraphView,DetailPanel}.tsx`: グラフとノード詳細。
- `src/components/{CitationExplorer,CitationMap}.tsx`: 引用マップ・ランキング・根拠と問題箇所の表示。
- `src/app/api/curves/[id]/route.ts`: 曲線。

一覧のDOMはページ単位、グラフは可視範囲だけを描画する。現在の200ノード規模では
サーバがMarkdownを読み、検索用本文もclientへ渡す。数千〜数万ノードに拡大する際は
本文取得と検索をサーバ側に分離する必要がある。別途生成したindexを正本として維持しない。
