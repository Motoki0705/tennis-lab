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
**E2Eの3D精度と実行システム**（担当名 e2e-system）

- E2E 3D精度: ボール3DのLOVO（tennis_scene 000032〜000041）、3Dボールイベント（000096・000097）、court side（court_side 000007・000008）、標準pipelineの動作確認（000028・000030・000031）。人物3D精度の評価の有無。
- 実行システム: 宣言型clip pipeline・component store・横断バッチ・モデル寿命の設計（src/tennis_scene/pipeline/README.md）と、それを示す実測（69 clip一括処理 000064・000083、TensorRT 000075〜000081、速度計測 000085・000086、横断バッチの効果）。
- 論文の第5章（実行システム）と第6章（実験）に載せる表・図の候補。
