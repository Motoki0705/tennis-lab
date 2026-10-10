# サイクルの回し方

1サイクル＝1子issue。template: [05_research_cycle.md](../../../../.github/ISSUE_TEMPLATE/05_research_cycle.md)
（タイトル `[研究サイクル] <テーマ名> #<n>: <今回の焦点>`、ラベル `research`）。
状態の正本はissueとknowledgeであり、セッションをまたいで再開できるように書く。

## 0. 開始

テーマissue本文（固定条件）、過去のサイクルissueの結論、関連knowledgeノードを読む。
サイクルissueを作り、テーマのsub-issueとして登録する。

```bash
CHILD=$(gh issue create --title "[研究サイクル] ..." --label research --body-file body.md | grep -o '[0-9]*$')
gh api -X POST "repos/{owner}/{repo}/issues/$THEME/sub_issues" \
  -F sub_issue_id="$(gh api "repos/{owner}/{repo}/issues/$CHILD" --jq .id)"
```

前サイクルの「次サイクル提案」があれば、それを今回の起点にする。

## 1. 調査

[literature.md](literature.md) に従い、文献とknowledgeの既存証拠を調べる。
issueの「調査」欄に、検索クエリ・ソース・候補一覧（出典リンクつき）と、
knowledge上で既に試して効かなかったもの（ノードID）を書く。

## 2. 選定

候補から今回試すものを選び、理由を書く。観点の例:

- 目標指標への期待効果と、それを支える根拠（論文・既存ノード）
- 予算内で、テーマの評価条件のまま検証できるか
- 結果が出たとき、次の判断が変わるか（どちらに転んでも学びがあるか）
- 既に否定された方向と実質的に同じでないか

ablationを含めるのは、主張する効果がどの要素によるかを分けて示す必要があるときだけ。

## 3. 実験

- テーマの評価条件（split・seed数・評価コマンド）をそのまま使う。
- GPUを使うものは小さなsmokeも含めて [training-queue](../../training-queue/SKILL.md) 経由で
  enqueueする（`--provider`/`--session`/`--issue <サイクルissue>` を付ける）。
- job名・job ID・コマンドをサイクルissueにコメントする。学習待ちの間はセッションを
  終えてよい。再開時はqueueの状態とissueコメントから続ける。
- 失敗・中断したjobも結果として扱う（黙って再投入・除外しない）。

## 4. 記録と考察

1. 各runを [knowledge-control](../../knowledge-control/SKILL.md) でノード化する
   （`issue` にサイクルissue番号）。サイクル内の比較は1つのgroupノードにまとめ、
   baselineへの `parents`/`relations` を張る。
2. [evidence-checklist.md](evidence-checklist.md) で、主張がノードの実測値から言える
   範囲に収まっているか確かめる。
3. サイクルissueの「結果」「考察」を書く。数値はknowledgeノードが正本で、issueには
   ノードIDと要点だけを書く（二重管理しない）。
4. knowledge summaryの更新・検証は knowledge-control の完了手順に従う。

## 5. 結論

サイクルissueの結論欄に次を書いて、knowledge変更のPRと合わせて閉じる。

- **停止条件の判定**: テーマの各停止条件について、満たした/満たしていないと根拠ノード。
- **次サイクル提案**: 仮説・やること・期待する観測。停止するなら「なし」と理由。
- **skill改善メモ**: この手順で詰まった点、不要だった手順、追加したい部品。
  無ければ「なし」。週次レポートがこれを集めてskillの改訂を提案する。

停止条件を満たしたら、テーマissueに最終報告（到達点、根拠ノード、残った不確実性、
未着手の候補）をコメントし、`needs-human` ラベルを付けて止まる。満たしていなければ
次サイクルへ進む。
