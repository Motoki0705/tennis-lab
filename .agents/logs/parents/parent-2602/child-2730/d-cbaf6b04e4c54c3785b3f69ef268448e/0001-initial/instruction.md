# #2602 残31ページ展開 担当 `pillar-scene-court`: Pillar I・II（11ページ・全Tier 2）の改訂と公開前検証

`docs/wiki/README.md` および `docs/wiki/rendering-comparison.md`（および `#2694` の `inventory.md`）に従い、担当する Pillar I・II の **11 ページ（すべて Tier 2）** を改訂し、公開前全検査を完了してください。

## 厳守事項・境界
- **本番 GitHub Wiki への `git push` は行わないでください**（後段で統合担当 `primary` が全 31 ページをまとめて最新 Wiki `master` へ公開します）。
- **本体リポジトリのファイルは変更・commit・PR 作成しないでください。** Issue コメント投稿も行わず、成果物は worktree 内 `tmp/wiki/`（担当 11 ページのみ変更）と `tmp/rollout-scene-court/` に保存してください。
- GitHub Wiki（`https://github.com/Motoki0705/tennis-lab.wiki.git`）はこの worktree の `tmp/wiki/` に clone し、以下の担当 11 ページ**以外**は一切触らないでください。

## 担当 11 ページ（すべて Tier 2）
- Pillar I（`tennis_scene` 標準パイプライン 5 ページ）:
  1. `Tennis-Scene-Pipeline-Overview.md`
  2. `Tennis-Scene-History-and-Optimizations.md`
  3. `Component-Person-Detection-and-Tracking.md`
  4. `Component-CourtSide-Association-Alignment.md`
  5. `Component-3D-Reconstruction-and-Tooling.md`
- Pillar II（コート検出・3DGS 合成・NHT 6 ページ）:
  6. `Court-Detection-Overview.md`
  7. `Court-Model-Evolution-and-Backbones.md`
  8. `Court-Geometry-Postprocessing-and-Video-Shared.md`
  9. `Synthetic-3DGS-Alignment-and-Generation.md`
  10. `Synthetic-Scaling-Diversity-and-Pair-Studies.md`
  11. `NHT-Rendering-and-Foreground-Compositing.md`

## 参照すべき規約・棚卸し
- `docs/wiki/README.md`（執筆ガイドライン正本）
- `docs/wiki/rendering-comparison.md`（各ページの改訂理由）
- `[local path]`（各ページの出典移設先・全表の改訂案）

## 改訂要件（全 11 ページ共通）
1. **冒頭定型メタデータ欄・定型戻り行の廃止**:
   - 冒頭の「項目／内容」表（「想定読者」「検索キーワード」「正本ファイル」）と定型戻り行（`> ← ...へ戻る`）を削除し、タイトル直後から本題に入れる構成にする。
   - 「正本ファイル」にあった全リンク・必要な出典や時期情報は、該当本文または末尾の「出典・関連ページ」節へ確実に移設する（1 リンクも落とさない）。
2. **Tier 2 の表改善（`docs/wiki/README.md` 準拠）**:
   - 横幅過多・長文セル（目安 40 字超）・階層・構成・年表・判断経緯・採否根拠を詰め込んだ表は、ツリー・`<details><summary>` 折りたたみ（タグ前後・`<[local path]>` 後の空行必須）・項目別／フェーズ別リストへ再構成する。
   - 多列の数値比較表（例: `Court-Model-Evolution-and-Backbones` の 10 列比較、`Synthetic-Scaling-Diversity-and-Pair-Studies` の 9 列全 6 ペア比較、`Synthetic-3DGS-Alignment-and-Generation` の 6 列比較など）は、読者の目的に応じて 3〜5 列以内の簡潔な小表へ分割し、長い採否・判断根拠・注記は表の直下本文または箇条書きへ分離する（390px 狭幅で横スクロールしない／破綻しない密度にする）。
   - 2〜4 列・おおむね 12 行以内・短セルで平坦な索引・対応表・同条件小比較は Markdown 表のまま残してよい。
   - Mermaid 図がある場合は維持し、狭幅で見づらい横長図は必要に応じ `flowchart TD` 等へ改善する。
3. **情報・数値・リンク・アンカーの完全保全**:
   - 旧版 11 ページのすべての数値・評価集合・母数・採否判断・Issue/PR/ファイルリンク・見出しアンカーを欠落なく維持する。
   - 数値や評価条件を変造・省略しない。

## 必須の公開前検証（成果物: `tmp/rollout-scene-court/`）
1. **共通検証 CLI の実行**: 全 11 ページに対して `.venv/bin/python -m tools.wiki_assets verify --wiki-page tmp/wiki/<Page>.md` を実行し、冒頭定型欄 0 件・`<details>` 空行・公開禁止情報 0 件・相対リンク検査が全件 PASS することを確認する。
2. **移行・リンク突合監査 (`tmp/rollout-scene-court/audit.md` / `.json`)**:
   - 旧版 11 ページの全リンク（URL・Wiki 相対リンク）・主要数値・見出しアンカー・表数の変化（旧→新）を突合し、リンク・情報欠落 0 件を確認する。
3. **ローカル表示チェック (`tmp/rollout-scene-court/render-check.md`)**:
   - 少なくとも最も表が大きかった代表 3 ページ（`Tennis-Scene-Pipeline-Overview`, `Court-Model-Evolution-and-Backbones`, `Synthetic-Scaling-Diversity-and-Pair-Studies`）について、ローカル Chromium（896px / 390px × ライト/ダーク）で変更前後の高さ・横スクロールブロック数・折りたたみ開閉・Mermaid 表示を確認し、スクリーンショットを目視確認する。

## 報告・終了形式
最終メッセージに、改訂した 11 ページのパス（`tmp/wiki/*.md`）、表数・横スクロールの変化、`tools.wiki_assets verify` 結果、旧版リンク・アンカー突合結果、代表ページ実表示結果を簡潔にまとめ、末尾に次の 1 行を出力してください：
`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}`
