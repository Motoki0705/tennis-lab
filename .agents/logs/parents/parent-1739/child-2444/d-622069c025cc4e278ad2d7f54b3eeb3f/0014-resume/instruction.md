# PR #2449: 正式レビュー（run parent-1739-ccae7e33131447219b2a379599c8ce9e、round 1、HEAD 907e4d66）の指摘への対応

次の2件のmajorを採用した。

## 1. 未実行の人物観測を、候補ゼロの正常結果として扱う
既存 recipe が `person_observations.enabled=false` で出す空の DINO・ViTPose 成果物を `read_saved_inputs()` に渡すと、検査を通ってしまう。`reconstruct()` は全フレームで候補ゼロを返す。`multiview_pose/inputs.py:215-242` は DINO の方式・範囲・マージ設定だけを検査し、DINO と ViTPose の有効設定を照合していない。このため下流は「観測を実行していない」と「実行した結果、人物がいなかった」を区別できない。
- 入力の境界で、DINO と ViTPose の両方の有効設定を確かめる。無効なら明示的なエラーで止める。
- 回帰テストで、無効設定を拒否することと、有効設定で人物が写っていない観測（空の観測）を正常に受理することを確かめる。

## 2. 固定版の手順では Meiji 未見評価を再現できない
README の `holdout-plan`・`holdout-score` が呼ぶ `_solver_config()` は、固定ソルバー `466ba35c` と現在の `src` 全体が一致することを要求する。PR head では main 取り込み後の128ファイルの差分で終了値1になり、評価入力を処理する前に止まる。一方、knowledge の `repro.commit` が指す `466ba35c` には、未見採点の入口がない。採点時の `590c2377` には入口があり、同じ検査が通る。
- 再現手順（`knowledge/runs/run-multiview-pose-link-recall-20261124/README.md` と関係するknowledgeノードの repro）に、実際に採点した版 `590c2377` と固定入力を明記する。その版で実行する手順を書く。
- 現行HEADで固定ソルバー検査が止まるのは意図どおりの保護か、手順の誤りかを整理して README に書く。意図した保護なら、止まる理由と、どの版で実行するかを明記する。黙って検査を緩めない。

## 共通
- 関連テスト、pre-commit、knowledge 検査（`--base-ref origin/main`）を通して push し、PR本文を更新する。knowledge の連番は98〜100のまま。
- ベンチマークとデータセットには触れない。GPU は使わない。

新HEAD（完全SHA）、各指摘への対処、テスト結果を報告し、CI待ちで終了する。
