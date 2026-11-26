親 #2627 は、テニスシーン3D再構成のアーキテクチャー（標準pipelineの手法＋実行システム）を日本語の技術報告として論文化する。親issue本文と #2630 本文を読む。#2630 の調査は人間の指示で項目ごとに並列化した。あなたは下の「担当項目」だけを調べる。他の項目は別の担当が並行して調べるので扱わない。

## 進め方

- コード・knowledge・paper配下は変更しない。commit・PRは作らない。GPU・training queueは使わない。保存済みの結果（json・npz・csvなど）を読む短時間のCPU集計は可。
- 出典は `knowledge/summary.md` → 該当トピックの `knowledge/nodes/<topic>[local path]` → `knowledge/runs/` の順に辿る。`kg_query.py`（.agents/skills/knowledge-control/SKILL.md）で絞り込める。関係issue・PRは `gh` で読む。
- 数値は記録の原文から写し、条件（dataset・clip・view数・frame数・metric・split・seed）を必ず添える。記録にない数値を推定で補わない。
- 主張を記録の範囲を超えて強めない。`knowledge/summary.md` の「結果を解釈するための規則」に照らして、比較できない組み合わせを明記する。

## 報告

調査結果を #2630 へ1件のコメントとして投稿する（日本語）。先頭行を `## 調査: <担当項目名>` とし、次の4項目で構成する。

- 採用候補の結果の表（主張・結果・数値・条件・出典・支持の判定〔支持／部分的／不足〕・比較可能性の注意）
- 図表の候補（内容と、元データまたは既存図版のパス）
- 不足する根拠と追加実験の候補（内容・必要データ・GPU/CPUの見込み・依存）
- 人間の主張と記録の食い違い

#2630 の本文は編集しない（統合は親が行う）。最終メッセージにはコメントのURLと、人間の判断が必要な点の要約を書き、`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終える。

## 担当項目
**コート検出**（担当名 primary）

範囲変更の通知: 先の指示の5項目のうち、このターン以降はコート検出だけを担当する。

人間の主張: 3DGSを用いたデータセット生成と、レンダリング視点の軌道の多様性によるモデルの堅牢性。

- `paper/court_robustness/` の報告（report.tex・evidence・tables）とcourt_detection・synthetic_data_generationのknowledgeから、3DGSでのデータ生成の手順と規模（シーン数・レンダリング枚数・視点軌道の作り方）を整理する。
- 「視点軌道の多様性が堅牢性を上げる」と言える比較（視点の多様性そのものを変えたアブレーション、TCDのみ学習とmixed学習の比較など）の有無と数値。
- 実画像testでの精度、収録を分けた評価、標準pipelineの下流（校正・E2E）への影響の記録。
