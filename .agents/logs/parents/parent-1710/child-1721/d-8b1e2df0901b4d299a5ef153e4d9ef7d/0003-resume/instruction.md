#1721 の継続です。前回の報告で閲覧中の質問・全体再構成・画像要求を止めていたwriterの変更申告は、#1828（PR #1829）で廃止しmain へmerge済みです。writerは request・document・change_summary だけを返し、変更一覧はruntimeがcompile後の差分から計算します。evidence_changes は信頼側contextから組み立てます。

1. 追従: PR #1807 を最新の origin/main へrebaseし（#1807 固有のcommitだけを載せる）、force-with-leaseでpushしてください。skill・references・helperのうち、変更申告（raw差分とslot化の照合など）を案内している記述や、restructure-request.txt のような申告の注意書きは削除し、runtimeの正本と二重管理にしないでください。
2. 実機の追加上限を24回とします（直列、agy 1.4.11 固定、--model gemini-3.8-flash-high --effort high。writer・verifier・figure修復を含む）。優先順位:
   a. 三角測量（公開済みの初版から）で、閲覧中の質問・全体再構成・新規画像要求（image_generation_unavailable の理由付き未完了）・undo をブラウザの実UIで確認。各操作は初回の結果を記録し、診断付き再試行と区別する。
   b. カメラ配置比較と観測誤差の初版を、skillの手順（Codexの根拠照合→誤りがあればwriterへの明示的な修正指示で再生成→全検査）で公開まで進める。1題材あたり最大3試行。
3. Codexの根拠照合で不合格にした候補を、本文の手修正や判定の書換えで公開しない方針は維持してください。
4. 動画・スクリーンショット・評価表・親の完了条件との対応表を更新し、人間の評価欄は空けたままにします。受入を満たせばPRをreadyにし、満たさなければdraftのまま残る問題と、モデルの出力品質に起因するか、runtime・skillの設計に起因するかの区別を書いて返してください。
