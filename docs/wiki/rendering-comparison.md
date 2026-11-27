# GitHub Wiki の実表示比較と棚卸し

2026-11-27 に公開 GitHub Wiki の一時比較ページ3件を、匿名 HTTP と Chromium 153 で検証した。棚卸しは一時ページ追加前の Wiki 版、描画に使った本体の正本は固定の公開版を参照した。ユーザーは 2026-11-28 に、Typst の数値表、Matplotlib のグラフ、ライト共通1版、Markdown 表の残存基準を選定した。執筆規約は [Wiki 執筆ガイドライン](README.md) を参照する。 [#2694 の検証・選定記録](https://github.com/Motoki0705/tennis-lab/issues/2694#issuecomment-7224065502)

## 同じ数値表での比較

検出は11条件、Top-down pose は10実測と Sapiens2 の6対象外条件。単位、表示の丸め、欠測、長いモデル名を固定し、ライトとダークで表示を確認した。画面1440px時の本文幅は896px、画面390px時は326px。次表は検出／pose の順。高さと幅は CSS px。生成時間はこの環境での単回バッチの中央値または内訳。

| 方式 | 通常幅の高さ | 390px画面での表示 | 生成時間と依存 |
|---|---:|---|---|
| Typst 0.15.1 | 301／373 | 全列版は横スクロールなし。9pt 文字が約5pxに縮小。検出を4列ずつに分けた試作は同じ幅で約11px | SVG 0.218秒、PNG 0.238秒。単体54 MiB、アーカイブ約16.7 MiB、和文・等幅フォント |
| LuaLaTeX + booktabs | 377／429 | 326pxに縮小。狭い見出しと長い名前は折り返し | PDF 3.792秒、SVG 0.152秒、PNG 0.176秒。TeX Live 2023、luatexja、booktabs、dvisvgm、Poppler |
| Matplotlib 3.11.1 | 371／464 | 326pxに縮小。罫線と文字は描画側で調整 | SVG＋PNG 0.213秒（表4枚・図4枚の中央値）。既存 Python 環境とフォント |
| HTML＋CSS から画像化 | 391／491 | 326pxに縮小。CSS は画像に反映 | Chromium 4ページで3.661秒、各ページ0.43〜0.54秒、Poppler/Cairo SVG化0.065〜0.076秒。Chromium、Playwright core、Poppler |
| Wiki Markdown 表 | 761／1,210 | 高さ1,577／2,050。必要横幅662／669 | GitHub ネイティブ。数値の右寄せが残る |
| Wiki HTML 表 | 785／1,210 | 高さ1,601／2,026。必要横幅636／669 | GitHub ネイティブ。`align=right` が残る |
| Wiki AsciiDoc | 785／1,210 | 高さ1,601／2,026。必要横幅636／669 | GitHub が変換。グループの `colspan` は残り、罫線や右寄せの指定は既定表示になる |
| Wiki reStructuredText | 737／1,234 | 高さ1,601／2,074。必要横幅662／669 | GitHub が変換。`list-table` を描画。指定の列幅比は確認できなかった |

Typst のライト版は検出159／134 KiB、pose 190／184 KiB（SVG／PNG）。4つの画像方式の中で通常幅の表が最も低く、和文と桁揃えを保ち、単体バイナリで生成できた。グラフは散布図、非支配点、診断点、backend 別の棒を描き分けられる Matplotlib を採用した。検出散布図は14点中4点が診断で、同条件10点のうち4点が非支配。backend の単位は固定のバッチ wall P50 を人数で割った ms/人。

公開時は22組の SVG／PNG を Chromium で同寸に描画して照合した。初回の Chromium PDF と dvisvgm の組み合わせでは和文見出しと一部のモデル名・状態が欠けた。HTML 由来の4 SVG を Poppler/Cairo で作り直し、修正後は22組の文字画素比較が通った。Wiki の4形式では表の全セルと状態を正本からの射影に照合した。両テーマと両画面幅で各50画像を表示できた。別のブラウザエンジンや実機は未確認。

## GitHub Wiki の表示制約

- 生 HTML の `style`、`class`、`id`、`data-*`、`border`、`bgcolor` は取り除かれた。`cellpadding="2"` など一部の属性は残ったが、実表示は上下6px・左右13px、文字16pxだった。
- `<details>` と `<summary>` は残り、Enter と Space で開閉できた。`</summary>` の後に空行を置いた例は内側のリストを5つ描画した。空行を省いた対照例のリストは0件で、ハイフン付きのテキストとして残った。
- Mermaid の4ノードと依存線は、両テーマで iframe 内の SVG として表示された。
- `#gh-light-mode-only` と `#gh-dark-mode-only` は相対 `src` に残ったが、4表示条件で両画像が表示された。`picture/source` の `prefers-color-scheme` はブラウザの配色に追従し、GitHub 側で逆の固定テーマにしてもブラウザ側の画像が表示された。白背景を画像に持つ共通1版で両テーマを確認した。
- 相対アセットは Wiki から raw 配信へ転送され、SVG と PNG は対応する MIME と配信バイトが一致した。SVG の原寸リンクは拡大時も輪郭を保った。原寸表示と折りたたみのテキスト表を設け、狭幅で文字が小さくなる表を補う。

[GitHub Wiki の編集](https://docs.github.com/en/communities/documenting-your-project-with-wikis/editing-wiki-content) · [GitHub Markup](https://github.com/github/markup)

## 全38ページの棚卸し

本文38ページと、共通のサイドバー・フッターを確認した。表は178件で、冒頭定型37件、本文141件。表の用途は階層・ファイル・構成24件、利用可否・判断37件、数値比較・集計44件、リファレンス・索引73件（冒頭定型37件を含む）。定型の戻り行は30件、定型の基準日は6件。必要な出典、測定日と判断日は関連節へ移す。サイドバーは39リンクで38ページを網羅し、外部リンクは5件。サイドバーとフッターに表はなかった。

Tier 1（6件）は代表ページの階層化と数値可視化を先行する。Tier 2（21件）は長い表を内容に合わせて順次改訂する。Tier 3（10件）は短い既存の参照構造を維持し、冒頭定型欄だけを撤去する。Home（1件）は入口の構成を維持する。本文表数は冒頭定型欄を除いた件数。以下は棚卸し時点の適用計画。

### Tier 1

| Wiki ページ | 本文表 | 改訂・簡素適用の理由 |
|---|---:|---|
| [Body-Recovery-FastHMR-vs-GVHMR](https://github.com/Motoki0705/tennis-lab/wiki/Body-Recovery-FastHMR-vs-GVHMR) | 6 | 人物比較。9列の速度と定性評価 |
| [Meiji-4cam-Benchmarks](https://github.com/Motoki0705/tennis-lab/wiki/Meiji-4cam-Benchmarks) | 7 | 階層、版・ファイル、利用可否の代表 |
| [Person-Detector-and-ECPose-Comparison](https://github.com/Motoki0705/tennis-lab/wiki/Person-Detector-and-ECPose-Comparison) | 8 | 人物比較。最大13列・18行 |
| [Person-Model-Comparison-Methods](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Methods) | 7 | 人物比較。数値の読み方を集約 |
| [Person-Model-Comparison-Overview](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Overview) | 4 | 人物比較。異なる段の数値と判断が横並び |
| [Top-Down-Pose-Model-Comparison](https://github.com/Motoki0705/tennis-lab/wiki/Top-Down-Pose-Model-Comparison) | 8 | 人物比較。最大11列・18行、B1/B4/B8混在 |

### Tier 2

| Wiki ページ | 本文表 | 改訂・簡素適用の理由 |
|---|---:|---|
| [AI-Ops-Overview](https://github.com/Motoki0705/tennis-lab/wiki/AI-Ops-Overview) | 3 | ロール・介入条件の文章が横並び |
| [Ball-2D-Detector-Baseline-and-Domain-Adaptation](https://github.com/Motoki0705/tennis-lab/wiki/Ball-2D-Detector-Baseline-and-Domain-Adaptation) | 5 | 評価集合と採否の混在。列が5つ |
| [Ball-3D-Smoothing-and-Scene-Evaluation](https://github.com/Motoki0705/tennis-lab/wiki/Ball-3D-Smoothing-and-Scene-Evaluation) | 2 | 方式比較と判断根拠の長い列が一体 |
| [Ball-Detection-Overview](https://github.com/Motoki0705/tennis-lab/wiki/Ball-Detection-Overview) | 3 | プロファイルと評価契約が長い横表 |
| [Ball-Long-Sequence-and-Architecture-Exploration](https://github.com/Motoki0705/tennis-lab/wiki/Ball-Long-Sequence-and-Architecture-Exploration) | 2 | 6列比較と母数の条件分け |
| [Ball-Refiner-2D-Failure-and-3D-Transition](https://github.com/Motoki0705/tennis-lab/wiki/Ball-Refiner-2D-Failure-and-3D-Transition) | 4 | 評価集合同士の違いと採否が埋もれる |
| [Component-3D-Reconstruction-and-Tooling](https://github.com/Motoki0705/tennis-lab/wiki/Component-3D-Reconstruction-and-Tooling) | 2 | 三角測量の5列と周辺ツールの長文 |
| [Component-CourtSide-Association-Alignment](https://github.com/Motoki0705/tennis-lab/wiki/Component-CourtSide-Association-Alignment) | 2 | 証拠モードの実験・採否の最長セル226字 |
| [Component-Person-Detection-and-Tracking](https://github.com/Motoki0705/tennis-lab/wiki/Component-Person-Detection-and-Tracking) | 2 | 実験経緯と数値が長文セル |
| [Court-Detection-Overview](https://github.com/Motoki0705/tennis-lab/wiki/Court-Detection-Overview) | 3 | 採用根拠は最長セル211字 |
| [Court-Geometry-Postprocessing-and-Video-Shared](https://github.com/Motoki0705/tennis-lab/wiki/Court-Geometry-Postprocessing-and-Video-Shared) | 5 | 採否と設定の長いセル、階層の混在 |
| [Court-Model-Evolution-and-Backbones](https://github.com/Motoki0705/tennis-lab/wiki/Court-Model-Evolution-and-Backbones) | 6 | KP・SEG・Pose比較が最大10列 |
| [Glossary-and-Search-Traps](https://github.com/Motoki0705/tennis-lab/wiki/Glossary-and-Search-Traps) | 2 | 定義の最長セル173字、長い用語対比表 |
| [Host-Ops-and-Recovery](https://github.com/Motoki0705/tennis-lab/wiki/Host-Ops-and-Recovery) | 3 | 症状と復旧経路が長い、緊急時の視認性 |
| [NHT-Rendering-and-Foreground-Compositing](https://github.com/Motoki0705/tennis-lab/wiki/NHT-Rendering-and-Foreground-Compositing) | 2 | JSON契約を長文で並べ、数値と判断が混在 |
| [Research-Models-PLCS-BLCS-SLCS](https://github.com/Motoki0705/tennis-lab/wiki/Research-Models-PLCS-BLCS-SLCS) | 5 | モデルの契約・到達点と比較数値が密集 |
| [Shared-Memory-Catalog](https://github.com/Motoki0705/tennis-lab/wiki/Shared-Memory-Catalog) | 5 | 4本の索引が長く、確認手掛かりも横方向 |
| [Synthetic-3DGS-Alignment-and-Generation](https://github.com/Motoki0705/tennis-lab/wiki/Synthetic-3DGS-Alignment-and-Generation) | 5 | 6列のアライメント条件と外観数値 |
| [Synthetic-Scaling-Diversity-and-Pair-Studies](https://github.com/Motoki0705/tennis-lab/wiki/Synthetic-Scaling-Diversity-and-Pair-Studies) | 4 | 全6ペアの9列。契約の長文セル |
| [Tennis-Scene-History-and-Optimizations](https://github.com/Motoki0705/tennis-lab/wiki/Tennis-Scene-History-and-Optimizations) | 3 | 年表の最長セル237字、最適化条件が混在 |
| [Tennis-Scene-Pipeline-Overview](https://github.com/Motoki0705/tennis-lab/wiki/Tennis-Scene-Pipeline-Overview) | 4 | 20部品×6列と長い入出力契約 |

### Tier 3

| Wiki ページ | 本文表 | 改訂・簡素適用の理由 |
|---|---:|---|
| [Ball-Datasets-Annotation-and-Test-Contracts](https://github.com/Motoki0705/tennis-lab/wiki/Ball-Datasets-Annotation-and-Test-Contracts) | 1 | 版別の小さな仕様表 |
| [Communication-Visibility-and-Audit-Logs](https://github.com/Motoki0705/tennis-lab/wiki/Communication-Visibility-and-Audit-Logs) | 4 | 2〜3列で問い合わせ単位が明確 |
| [Installation-Storage-and-Outputs](https://github.com/Motoki0705/tennis-lab/wiki/Installation-Storage-and-Outputs) | 3 | パス・資産表は2〜3列。既存用途単位で参照可 |
| [Issues-Adoption-and-Dispatcher](https://github.com/Motoki0705/tennis-lab/wiki/Issues-Adoption-and-Dispatcher) | 3 | 2列のラベル・状態・実行判断参照 |
| [Knowledge-Library-and-Annotation](https://github.com/Motoki0705/tennis-lab/wiki/Knowledge-Library-and-Annotation) | 4 | 正本へ引く2〜3列の索引が主体 |
| [Parent-Session-Lifecycle](https://github.com/Motoki0705/tennis-lab/wiki/Parent-Session-Lifecycle) | 3 | 7フェーズと状態を横に照合する3列 |
| [Research-and-Infra-Overview](https://github.com/Motoki0705/tennis-lab/wiki/Research-and-Infra-Overview) | 1 | 4列のカテゴリ索引のみ |
| [Review-and-Merge-Gate](https://github.com/Motoki0705/tennis-lab/wiki/Review-and-Merge-Gate) | 2 | 設定と例外の2列だけ |
| [Shared-Memory-and-AutoDream](https://github.com/Motoki0705/tennis-lab/wiki/Shared-Memory-and-AutoDream) | 3 | 2〜5行の索引と契約参照 |
| [Worker-Delegation-and-Waits](https://github.com/Motoki0705/tennis-lab/wiki/Worker-Delegation-and-Waits) | 2 | 2列・3行。操作と条件に対応 |

### Home

| Wiki ページ | 本文表 | 改訂・簡素適用の理由 |
|---|---:|---|
| [Home](https://github.com/Motoki0705/tennis-lab/wiki/Home) | 3 | 入口を維持。定型メタデータ表なし |
