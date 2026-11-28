# #2602 子Issue #2699 担当: 高密度数値表（Typst）・グラフ（Matplotlib）の共通生成基盤と Wiki 執筆ガイドラインの整備

#2694 で確定した採用方針に基づき、本体リポジトリへ共通生成基盤・テスト・Wiki 執筆ガイドラインを実装し、ready PR を作成してください。

## 確定仕様（ユーザー最終承認済み 2026-11-28）
1. **生成ツール**:
   - 高密度数値表: **Typst 0.15.1**（固定版単体バイナリ。公式アーカイブ SHA-256 `a6d077d0a95eed5a2eba715b2dae06be954f624ccbf85758a03f389ded33118c` を検証してローカルキャッシュへ取得、または環境変数・引数で既存バイナリを指定可能。本体の推論パイプラインや `pyproject.toml` の推論依存には影響させない）。
   - グラフ: **Matplotlib**（既存 `.venv` の matplotlib / Pillow / numpy / pandas 等を使用）。
2. **画像形式・テーマ**:
   - **ライト背景（白背景＋明確な文字色・薄い枠線）の共通 1 版**を標準とする。
   - **日本語アウトライン化 SVG** を標準表示用として生成し、互換用 **PNG** と**同値の折りたたみ用テキスト表（Markdown/CSV参照）**を同時に生成・検証できるようにする。
   - SVG は外部 resource・`<script>`・不要メタデータを含まず、日本語フォントをパス化（アウトライン化）する。PNG は必須チャンク・RGB 画素のみとする。
3. **Markdown 表残存基準**:
   - 2〜4 列・おおむね 12 行以内の平坦な索引・対応表・小規模同条件比較（目安：セル 40 字以内、390px 狭幅で横スクロールなし）は Markdown 表を維持する。
   - 階層・ファイル仕様・長文の判断経緯はツリー[local path]`<details><summary>` 折りたたみ（空行ルール厳守）[local path]。
   - 多列数値比較表は目的別（精度・速度[local path] 等）に分割した Typst 表＋Matplotlib グラフ（＋折りたたみテキスト表／元データリンク）へ移行する。
4. **Wiki アセット配置規約**:
   - 生成コード・テンプレート・仕様・テスト・ガイドラインは **本体リポジトリ** で PR/CI 管理。
   - 生成済み SVG/PNG は **Wiki リポジトリの `assets/<page-slug>[local path]`** に配置し、相対リンクで参照する（本 PR では Wiki repo への push は行わない）。

## 参考となる先行試作（#2694 の worktree）
#2694 の棚卸し・設計・試作スクリプト・検証記録は以下にあります（読み取り参照のみ。直接編集しないでください）：
- `[local path]`
- `[local path]`
- `[local path]`
- `[local path]`
- `[local path]`
  （試作で取得済みの Typst バイナリも同 worktree の `tmp/tools/` にあります）

## 実装する成果物（本体リポジトリ PR）

### 1. `tools/wiki_assets/` 共通生成基盤
モジュラーに設計し、静かなフォールバックを避け、入力欠落・型不正・未定義状態は明示エラーにしてください。
- **パッケージ構成案**（適宜わかりやすく分割）:
  - `tools/wiki_assets/__init__.py`、`__main__.py`（CLI 入口: `generate`, `verify`, `list-pages` 等）
  - `typst_runner.py`: 固定版 Typst 0.15.1 の解決・SHA-256 検証付きキャッシュ取得（`--typst-bin` / `WIKI_ASSETS_TYPST_BIN` 指定可）・SVG/PNG レンダリング（ライト背景・日本語アウトライン化・Noto Sans CJK JP / DejaVu Sans Mono 等のフォント検査）
  - `table_spec.py` / `table_renderer.py`: 高密度数値表の共通データモデル（タイトル、サブタイトル/条件注記、列定義・単位・丸め桁・桁揃え・最良値方向、グループ行、状態表現：実測 `0.00` と欠測 `—`・未計測・ライセンス除外・打ち切り・変換異常・実行不能・重みロードのみ等を厳密分離、丸め前での最良値計算）＋ Typst ソース生成＋同値 Markdown 折りたたみテキスト表スニペット生成。大表の目的別分割（例: 精度表・速度[local path]）にも対応。
  - `chart_renderer.py`: Matplotlib 共通グラフ生成（精度–latency 散布図＋**比較可能条件のみ**の Pareto frontier、診断点 `×` 等の明示分離、拡大パネルのモデル名ラベルまたは明瞭な凡例対応、backend 別棒グラフ、HMR 速度・VRAM 比較棒グラフ等。SVG 出力時は `svg.fonttype = 'path'` で日本語をアウトライン化し、ライト背景・メタデータ除去・PNG 同時出力）。
  - `pages/`（人物モデル比較ページ群の正本ローダと定義）:
    - 対象正本:
      - `knowledge/runs/run-meiji-player-model-comparison/conditions.csv` & `comparison-evidence.json` & `joint-speed/cases.csv`
      - `knowledge/runs/run-meiji-topdown-pose-comparison/comparison.csv` & `accuracy.csv` & `difficulty_pt32.csv` & `sapiens2_attempts.json`
      - `knowledge/runs/run-person-model-speed-low-precision/cases.csv` & `run-person-model-speed-baseline/cases.csv`
      - `knowledge/runs/run-fasthmr-camerahmr-pt16-speed/speed-summary.json`
    - 対象ページ定義（人物モデル比較 5 ページ群のうち数値表・グラフを持つページ群：`person-model-comparison-overview`, `person-detector-and-ecpose-comparison`, `top-down-pose-model-comparison`, `body-recovery-fasthmr-vs-gvhmr`, 必要なら `person-model-comparison-methods`）について、正本 CSV/JSON から表・グラフ一式および `manifest.json`（入力 SHA-256・出力ファイル・行数・除外理由・ツール版）と埋め込み用 Markdown スニペット（`<img ...>` / Markdown 画像＋PNG リンク＋折りたたみテキスト表＋正本リンク）を再現可能に生成できるようにしてください。
    - **重要**: 異なる workload・batch・計時境界を混ぜて順位付けしない。疑似GTの限界、未計測・変換失敗・ライセンス除外・打ち切り・重みロードのみを隠さない。定性評価のみの HMR に架空の精度を絶対に出さない。
  - `verify.py`: 生成物および Wiki Markdown ページ向けの公開前検証：
    - 生成済み `manifest.json` と正本 CSV/JSON の SHA-256・数値・状態一致検査
    - SVG 検証（パス化確認：`<text` 要素なし、外部 URL/image 参照なし、`<script` やイベント属性なし、ファイルサイズ上限、対応する PNG の存在確認）
    - PNG 検証（RGB・破損なし・解像度）
    - 公開禁止情報検査（ローカル絶対パス `[local path]`, `[local path]`, `[local path]`, メールアドレス/トークン、実行ID等）
    - Markdown ページ検査（冒頭定型メタデータ表の残存検知、`<details>` タグ前後の空行検査、相対アセットリンクの実在検査、`alt` テキスト非空検査）
  - `tools/wiki_assets/README.md`: 生成基盤の技術仕様、依存（Typst 0.15.1 の固定版取得）、実行コマンド例、テスト方法を簡潔に記載（ガイドライン本体は `docs/wiki/README.md` へリンクし二重管理しない）。

### 2. `docs/wiki/` 版管理ドキュメント
- `docs/wiki/README.md`: **GitHub Wiki 執筆ガイドライン（正本）**
  - 冒頭定型メタデータ欄（「想定読者」「検索キーワード」「正本ファイル」、定型戻りリンク、定型基準日ブロック）の廃止ルールと、必要な出典・正本参照・版・測定日の配置方針。
  - 情報階層化の標準パターン（ディレクトリツリー、`<details><summary>` 折りたたみ＋空行必須ルール、Mermaid 縦/横の使い分け、階層リスト）。
  - 既存 Markdown 表を残す基準（2〜4 列・おおむね 12 行以内・セル 40 字目安・390px 横スクロールなし）。
  - 高密度数値表（Typst）とグラフ（Matplotlib）の設計規約（単位・桁揃え・丸め前順位・欠測と実測ゼロの区別・疑似GT/非比較条件の明示・HMR 定性のみ規約）。
  - Wiki `assets/<page-slug>[local path]` 配置・命名・ライト背景共通 1 版・アウトライン化 SVG＋PNG リンク＋折りたたみテキスト表/正本リンク規約、`alt` テキスト規約、公開前チェックリスト。
  - 再生成コマンドは `tools/wiki_assets/README.md` へリンクして二重管理を避ける。
- `docs/wiki/rendering-comparison.md`: **#2694 表示制約・レンダリング比較根拠と全 38 ページ棚卸し**
  - `#2694` で実測した比較表（Typst / LuaLaTeX / Matplotlib / HTML+CSS / Markdown / HTML / AsciiDoc / RST の通常幅・390px 幅・生成時間・依存・GitHub サニタイズ・`#gh-*-mode-only`・`picture`・`<details>` 空行要件の知見）と最終採用根拠。
  - 全 38 ページの棚卸しサマリ、Tier 1 (6) / Tier 2 (21) / Tier 3 (10) / Home (1) の分類と未適用・簡素適用ページの理由記録の正本。
  - **ローカル絶対パスや内部 worktree パスは一切含めず、公開可能な記述のみとする。**

### 3. テストと品質検証
- `tests/unit/tools/test_wiki_assets/`（既存の `tests/` 規約に従う配置）に公開可能 fixture を使った意味のあるテストを実装：
  - データ変換・丸め前最良値計算・Pareto 非支配点抽出（非比較可能条件・変換異常の除外）・欠測/ゼロ分離のテスト
  - Typst ソース生成・折りたたみテキスト表生成のテスト
  - Matplotlib グラフ生成（SVG パス化・PNG 生成・外部参照なし）のテスト
  - `verify.py` の正常系・異常系（`<text>` 残存、外部参照、ローカルパス混入、冒頭定型メタデータ残存、`<details>` 空行欠落、PNG 欠落、数値不一致を確実に検出すること）のテスト
  - Typst 実バイナリが利用可能な環境での end-to-end 生成＋検証テスト（CI に Typst がない場合はオフライン固定 fixture でバイナリ未配置時のエラー契約もテストし、ネットワーク依存で CI が不安定にならないよう配慮する！）
- **実正本データでの動作確認**:
  - 実装した `tools/wiki_assets` で人物モデル比較 5 ページ群の全表・全グラフを `tmp/wiki-assets-preview/` に実生成し、`verify` が全件 PASS すること、さらに生成された SVG/PNG の代表例を目視・画素検査して文字欠落・重なり・折り返し崩れがないことを確認し、その結果を PR 本文に記録する。
- `ruff`、`mypy`、`pre-commit`、および関連 `pytest` をすべて通す。

## コミット・PR・引き渡し条件
- PR 本文は日本語で書き、`Closes #2699` を明記する。変更内容、受入条件ごとの証拠、テスト結果、実正本での生成・検証結果を記載する。
- ローカル検証と pre-commit を通して commit・push し、ready PR を作成する。
- PR 番号・完全な HEAD SHA・ローカル検証結果を最終メッセージに報告し、PR CI 完了前に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す（CI 待機とレビュー進行は親が `review-request` で担当する）。
- 注意: `git status` や commit 対象に `tmp/` の一時生成物が混入しないこと。また `pyproject.toml` / `uv.lock` に不要な依存を追加しないこと（既存の `.venv` で動作させる）。
