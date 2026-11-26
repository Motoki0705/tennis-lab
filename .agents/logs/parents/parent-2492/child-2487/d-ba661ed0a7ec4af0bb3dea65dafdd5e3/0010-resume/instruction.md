# #2487 統合担当: helper の最終結果をまとめて #2499 に取り込む（親の判断）

前回の報告（`edeea50b4`、36条件・Pareto 図2枚・代表6 clip）を受け取った。helper 3人の作業がすべて終わったので、まとめて取り込む。

## 取り込むもの（各 helper の branch の最新 HEAD）
- edgecrafter `2d05a6de50103f4903e073a58fa5c0a71ae61645`:
  - PT16 の速度を、ホストスレッドを揃えて測り直した12条件（`edgecrafter-pt16-speed-matched.json`、PyTorch 4・OpenCV 1スレッド）。
  - ECDet 4重みの `trt16-fp32attn` の精度と速度（`edgecrafter-trt16-fp32attn-results.json`）。ECPose は書き換えを適用できない（`not_applicable`）。
  - TensorRT の調査は打ち切った。元の `trt16` と `trt16-fp32attn` は「変換の不具合、原因未特定」。
  - queue ログ26件を保持先へ移した。担当によると、`comparison-evidence.json` の EdgeCrafter 参照 SHA-256 3件の更新が要る。
- rfdetr `3e60008c243d803a08b457e2cfca63f625e792ac`: queue ログ11件の保持と、記録の参照の更新。
- rtdetrv4 `d2706513d58b342444240d7a136455133c8fff25`:
  - PT16 L/X × batch 1・2 を、PyTorch 4・OpenCV 1スレッドで測り直した。batch 1 の wall P50 は L 53.67・X 58.16 ms/画像。旧2スレッドの値は `initial_speed` に残っている。
  - `trt16-fp32attn` は全ケースで NaN。TensorRT は「変換の不具合、原因未特定」。

## やること
1. 3つの branch を取り込む。コンフリクトは、各 helper の記録を正として解く。
2. 比較表と Pareto 図を作り直す。
   - 速度は、ホストスレッドが揃った値だけを使う（DINO・RF-DETR-L・EdgeCrafter・RT-DETRv4 がすべて PyTorch 4・OpenCV 1スレッドになったか、記録で確かめる）。表にスレッド設定の列を足す。旧2スレッドの値は表から外し、記録に残す。
   - EdgeCrafter と RT-DETRv4 の TensorRT 条件（`trt16`・`trt16-fp32attn`）は、測定点として表と図に残す。Pareto の判定からは外し、「変換の不具合、原因未特定」と注記する。ECPose の `trt16-fp32attn` は「適用不可」と表に書く。
3. PR #2500 の正式レビューで出た指摘を、#2499 の集計にも先に当てる。
   - **精度を測った batch の明記**: 速度・VRAM を測った batch と、精度を測った batch を、表の別の列と図の注記で示す。未計測の batch の精度を、測ったように見せない。
   - **入力の検査**: 比較 CLI 自体で、各レポートを固定インデックスの期待 SHA-256 と照合し、不一致なら失敗させる。欠測（実行不能）を受理する前に、その条件の失敗を記録した終端ログがあることを確かめる。それぞれ回帰テストを足す。
   - **ログの保持**: あなたの記録（`comparison-evidence.json`、DINO＋ViTPose-H の一体速度など）が `.training_queue/logs/` を参照していれば、ログを自分の保持先へ SHA-256 付きでコピーし、参照と読込を揃える。queue のログは完了から7日で消える。元ログが消えていれば、親の複製 `.worktree-outputs/parent-2492-queue-log-archive/logs/`（`SHA256SUMS` 付き）から取ってよい。
4. PR 本文の結果・証拠・検証の節を、新しい HEAD に合わせて直す。採用判断に残る検証（実 BBox での ViTPose-H と下流品質、TensorRT の原因、ライセンス）を残課題として書く。既定モデルの切替は人間の判断事項。

## 終わり方
テストと pre-commit を通して push する。比較 CSV と図を再生成し、helper の記録の値と一致することを確かめる。最終回答には、比較表で何が変わったか（特に Pareto に残る条件）、push した HEAD を書き、CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"<新HEAD>","resume_on_success":false}}` を返す。
