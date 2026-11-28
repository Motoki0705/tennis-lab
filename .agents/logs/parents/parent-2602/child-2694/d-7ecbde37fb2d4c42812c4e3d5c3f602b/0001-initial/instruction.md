# #2602 子Issue #2694 担当: 現行Wikiの棚卸しと階層・高密度表・グラフ表示方式の実レンダリング比較

親Issue #2602 の方針・仕様確定に先立ち、現行 GitHub Wiki の棚卸し、情報階層化・数値可視化・アセット配置の設計、および複数レンダリング候補の実 GitHub Wiki 上での比較検証を行ってください。

## 厳守事項・境界
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください**（共通生成基盤コードやガイドラインの実装・PR は、本比較を人間が確認・選定した後の次タスクで行います）。Issue へのコメント投稿も行わず、親への報告は最終メッセージと worktree 内 `tmp/wiki-2694/` の成果物にまとめてください。
- GitHub Wiki は `https://github.com/Motoki0705/tennis-lab.wiki.git`（**公開リポジトリ**）です。clone はこの worktree の `tmp/wiki/` に置いてください（`tmp/` は gitignore 済み）。
- **既存の公開 Wiki ページ（全38ページ）、`_Sidebar.md`、`_Footer.md` は一切変更しないでください。**
- 公開 Wiki へ追加・push してよいのは、ユーザー承認済みの**サイドバー未掲載の一時比較ページ**（`_Preview-Wiki-Style-Comparison.md`、および必要ならマークアップ検証用の一時ページ `_Preview-Wiki-Style-Comparison-*.{asciidoc,rst}`）と、その一時アセット `assets/_preview-style-comparison/` だけです。
- 追加ツール（例: `Typst` 単体バイナリ等）を試す場合は、worktree 内の一時領域（`tmp/tools/`）に配置し、本体リポジトリや共有 `.venv` を汚さないでください。
- **公開禁止情報の厳禁**: ローカル絶対パス、個人情報、非公開データ・動画フレーム、トークン類を Wiki ページ・画像・メタデータに一切含めないでください。push 前に自動検査を行ってください。

## 実施内容

### 1. 現行 Wiki 全38ページ＋`_Sidebar.md`・`_Footer.md` の完全棚卸し
成果物: `tmp/wiki-2694/inventory.md`（および機械集計の補助 JSON/CSV があれば同ディレクトリに保存）
- 全38ページの冒頭定型メタデータ欄（「項目／内容」表の「想定読者」「検索キーワード」「正本ファイル」、冒頭の定型戻りリンク、定型「基準日」ブロック）を全件棚卸しし、**不要として削除する項目**と、**本文・出典節へ移して維持すべき出典・版・測定日情報**をページごとに仕分けてください。
- 全38ページのすべての既存 Markdown 表を抽出し、列数・行数・最大セル文字数・内容を確認したうえで用途別に分類してください：
  1. 階層・ディレクトリ・ファイル仕様・コンポーネント構成の説明
  2. モデル別利用可否・用途マトリクス・意思決定/判断履歴
  3. 精度・推論速度・メモリ等の数値比較表
  4. 簡潔なリファレンス・索引・対応表（Markdown 表を残してよい候補）
- 既存ページの改訂優先順位案（Tier 1: `Meiji-4cam-Benchmarks` と人物モデル比較 5 ページ群、Tier 2: 表示問題が大きい既存ページ、Tier 3: 冒頭定型メタデータ欄の廃止のみで十分なページ、未適用とする箇所があればその理由）を作成してください。

### 2. 情報階層化・数値可視化・アセット配置の設計案
成果物: `tmp/wiki-2694/design-and-comparison.md`
- **`Meiji-4cam-Benchmarks` の階層型再構成案**:
  - データセット全体像（ツリー形式）、ディレクトリ・ファイル仕様（`<details><summary>`＋入れ子リスト等による折りたたみ式ツリー）、派生・依存関係（Mermaid）、モデル別利用可否（階層型リスト）、Hugging Face 配布版セクションの整理案。
  - 旧版の全情報・リンク・出典・注意事項を欠落させないマッピング。
- **人物モデル比較 5 ページ群の数値表・グラフ構成案**:
  - 対象: `Person-Model-Comparison-Overview`, `Person-Model-Comparison-Methods`, `Person-Detector-and-ECPose-Comparison`, `Top-Down-Pose-Model-Comparison`, `Body-Recovery-FastHMR-vs-GVHMR`。
  - 既存正本データ（`knowledge/runs/run-meiji-player-model-comparison/conditions.csv`, `knowledge/runs/run-meiji-topdown-pose-comparison/comparison.csv`, `knowledge/runs/run-person-model-speed-low-precision/cases.csv`, `knowledge/runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json` 等）と各表・グラフの対応表。
  - 欠測・未計測・OOM・変換失敗・疑似GTの限界・非比較可能条件・定性評価のみの HMR に架空精度を与えない明示設計。
- **既存 Markdown 表を残す明確な基準案**（用途・列数・セル幅・情報構造による判断基準）。
- **Wiki アセット配置・再生成規約案**:
  - 生成スクリプト・テンプレートは本体 repo で PR/CI 管理、生成済み SVG/PNG は Wiki repo 内 `assets/<page-slug>[local path]` に配置して相対リンクで参照する前提で、ディレクトリ構造・命名規則・ライト/ダーク対応・`alt` テキスト・元データリンクまたは折りたたみテキスト数値表の併記（検索・コピー・監査性の確保）・再生成コマンド設計をまとめる。

### 3. 同一実正本数値を用いた複数レンダリング候補の試作と実 GitHub Wiki 検証
成果物:
- 試作スクリプト群: `tmp/wiki-2694/prototypes/`
- 実レンダリング検証記録: `tmp/wiki-2694/render-evidence.md`
- 公開一時比較ページ: `tmp/wiki/_Preview-Wiki-Style-Comparison.md` および `tmp/wiki/assets/_preview-style-comparison/`

以下を**すべて同一の実正本数値**で試作・比較してください（人物モデル比較の正本 CSV から、長いモデル名・複数指標・単位・グループ化・最良値強調・未計測/変換失敗/非比較可能条件を含む代表的な実データ表を2種選定。例: 検出モデル比較表＋Top-down 2D pose 比較表）：
1. **Typst**: 単体バイナリを `tmp/tools/` に取得し、高密度学術表（小さい余白・列幅最適化・桁揃え・単位・グループ化・最良値強調・適切な罫線）を SVG および PNG で生成。ライト/ダーク両方、および SVG 内フォントの扱い（アウトライン化/文字崩れ有無）を確認。
2. **LaTeX (`lualatex`) + `booktabs`**: 既存環境の `lualatex` を用いて同等の高密度表をコンパイルし、SVG および PNG へ変換して比較（必要な変換ツール・日本語フォント対応・ビルド時間・依存の重さも記録）。
3. **HTML + CSS 静的レンダリング**: 導入コスト・再現性・出力品質を評価。ローカル環境で実行可能な軽量手段があれば SVG/PNG を生成して比較し、追加ブラウザ依存などが必要な場合は制約と実用性を具体的に記録。
4. **Matplotlib**: 共有 `.venv` の matplotlib で (a) 高密度数値表（SVG/PNG）と、(b) グラフ試作（精度–latency 散布図＋比較可能条件のみの Pareto frontier、backend 別棒グラフ等。軸・単位・指標・計測条件・母集団・疑似GT注記・未計測/変換失敗の明示を含む）を SVG および PNG で生成。
5. **GitHub Wiki ネイティブマークアップ**:
   - 通常 Markdown 表、生 HTML `<table>`（GitHub サニタイズで残る属性・消える `style` 属性を実測）、`<details><summary>`＋入れ子リスト/ツリー（空行要件含む）、Mermaid。
   - AsciiDoc / reStructuredText の Wiki 上でのレンダリング可否・表密度・CSS制約も実検証する。

### 4. 公開一時比較ページへの push と実レンダリング確認
- 試作した比較ページ `_Preview-Wiki-Style-Comparison.md` と `assets/_preview-style-comparison/` について、以下を検査したうえで `git -C tmp/wiki pull --rebase` → commit（日本語メッセージ、`Co-Authored-By` なし）→ push してください：
  - 正本 CSV/JSON の数値・単位・欠測/除外/比較不能条件との完全一致検査
  - 公開禁止情報検査（ローカルパス、個人情報、トークン、非公開フレームなし）
  - 相対リンク・代替テキスト（`alt`）・ファイルサイズ検査
- push 後、以下を実測して `tmp/wiki-2694/render-evidence.md` に記録してください：
  - `https://github.com/Motoki0705/tennis-lab/wiki/_Preview-Wiki-Style-Comparison` の HTTP 200 と、GitHub が実際に返したレンダリング済み HTML におけるサニタイズ挙動（`<details>`、`<picture><source media="(prefers-color-scheme: dark)">`、`#gh-dark-mode-only` / `#gh-light-mode-only`、SVG の `<img>` 配信方式、HTML `<table>` 属性の残存状況）。
  - SVG vs PNG の実表示互換性（日本語フォント描画・文字化けリスク・拡大縮小・ファイルサイズ・GitHub camo/raw 経由でのSVG表示可否）。
  - ライトモード／ダークモード、通常画面幅／狭い（モバイル）幅、長いモデル名、大表の分割方針の比較評価。
- **一時比較ページ `_Preview-Wiki-Style-Comparison` は、人間と親AIが目視確認・選定を終えるまで削除せず残してください**（サイドバー・Home からはリンクしないでください）。AsciiDoc/reStructuredText 検証用に別ページを作った場合も、比較ページから参照するか検証結果を記録したうえで整理方針を報告してください。

### 5. 採用推奨案の整理
`tmp/wiki-2694/design-and-comparison.md` の冒頭（および最終メッセージ）に、人間がそのまま最終決定できる形で以下の比較表と推奨案をまとめてください：
1. 生成ツールの最終選定案（表用・グラフ用）と採用根拠
2. 画像形式（SVG vs PNG）とライト/ダーク対応方式の選定案
3. 既存 Markdown 表を残す基準案
4. 対象ページの改訂優先順位案（全38ページの Tier 分類と冒頭メタデータ廃止対象37ページの確認）
5. 一時比較ページの公開 URL と確認ポイント

## 報告・終了形式
- 最終メッセージに、成果物ファイル一覧、Wiki commit hash、公開一時比較ページ URL、実レンダリング検証の主要結果、採用推奨案、未確認事項・残課題を分かりやすくまとめてください。
- 最終メッセージの末尾に必ず次の1行をそのまま出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
