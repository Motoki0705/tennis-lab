[PR #1999](https://github.com/Motoki0705/tennis-lab/pull/1999) を作成・push済みです。未mergeです。  
branch: `codex/review-model-select`[local path]`f1fbf8ac6`

- **受入1:** 通常設定をselectへ変更。`test_review_choice_handoff.py` で代表4ケース・自律入口・指定漏れ拒否が成功。
- **受入2:** runner契約・保存選択・実効model/effort、人間指定、再利用、handoff・coverage・gateを検証。
- **受入3:** 運用手順をparent READMEへ集約。文書126件、fixed復帰、使用量集計例の検証が成功。

unit 1,225件成功。E2E 1,189ノードは修正後の再実行を含め成功を確認。pre-commit成功。[CI #4109](https://github.com/Motoki0705/tennis-lab/actions/runs/43572394651) はPythonを含む14ジョブすべて成功しました。

残る観測は、並列停止テストの一過性失敗1件（branch/main単独成功、原因未確定）と既存ops mypyの5件です。詳細はPR本文に記載しました。実モデルの品質・費用比較は未実施です。正式レビュー・merge・merge後mainのCI確認は親側に残ります。