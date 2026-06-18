# Knowledge Control — Web UI

`knowledge/nodes/*.md`（学習知識グラフ）を Next.js + React Flow で可視化する閲覧 UI。

## 実行

```bash
cd knowledge/webui
npm install
npm run dev          # http://localhost:3000
```

ビルド時 / リクエスト時に `../nodes/*.md` を直接読み込む（別途 graph.json 生成は不要）。
ノードを追加・編集したら再読込で反映される。

## 機能

- run ノード = カード（provider 色 / issue / status / 主要 metrics）。
- group ノード = 破線ボックス。
- 有向エッジ: 実線 = parent→child、破線(黄) = group→member、点線(青) = relation（ラベル付き）。
- ノードクリックで右パネルに config / metrics / artifacts / 考察(Markdown) を表示。
- 上部チップで issue / provider / tag フィルタ。
- レイアウトは dagre で自動配置（rankdir=TB）。

## 構成

- `src/lib/nodes.ts` — server 側で `.md` を gray-matter + marked でパースしグラフ化。
- `src/lib/layout.ts` — dagre レイアウト。
- `src/components/GraphView.tsx` — React Flow 本体 + フィルタ。
- `src/components/nodeTypes.tsx` — run/group カスタムノード。
- `src/components/DetailPanel.tsx` — 詳細パネル。

ノード仕様は `../README.md` を参照。
