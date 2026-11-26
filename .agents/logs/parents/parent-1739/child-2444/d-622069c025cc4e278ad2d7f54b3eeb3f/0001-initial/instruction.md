# #2444 新経路の3D骨格候補・観測割当: 設計調査（planning）

担当issue #2444 の本文が範囲と受入条件の正本。親 #1739 の本文（再設計の合意、未決事項）と、調査レポート #1737（`knowledge/reports/multiview-3d-player-tracking-survey.md`、branch `codex/multiview-3d-player-survey` にある場合はそこ）を読む。
このターンは**調査と設計案の提出まで**。本実装は、親がユーザーに確認してから行う。CPU での小さな試作・計測はしてよい。GPU は training queue 経由に限り、queue worker は起動しない。

## 調べて決める案を作ること
1. **入力契約**:
   - 全人物の DINO 検出 row と、全 row の ViTPose（`person_pose_features`）、校正済みカメラが、現行の成果物（schema・保存場所）でどう取れるか。
   - 新経路の入力から旧経路の track・選別・対応付けを除けるか。
   - meiji_4cam の全clip実行の成果物（`knowledge/runs/run-i2363-trt-player-video`）で実際に確認する。
2. **court side の決め方**（現行の本番は `court_side.evidence=player` で、旧経路の選別に依存する）。3案を比べる。
   - (a) ボールで決める。i2363 はボール検出なしで実行した。必要な追加計算を見積もる。
   - (b) video 単位で共有した判定を使う。
   - (c) 新経路の中で、half-turn の仮説ごとに人物の関節レイの整合を比べる。
   - 可能なら、meiji_4cam の数〜全clipで、(c) の判定と現行本番の判定の一致を CPU で試算する。
3. **3D骨格候補と割当の solver**:
   - #1737 の 4D Association Graph を参考にした幾何方式を初期候補とし、他の候補（frameごとのMILP・貪欲な三角測量のクラスタリングなど）と比べる。
   - 比べる観点: 計算量、二重使用の禁止、別人の関節の混入の検出、ダブルス（4人＋周辺人物）での曖昧さ。
   - 小さな試作で、Meiji 3cam の人の確定ラベル付き clip の frameごとの割当の pair F1 を測れるとよい。
4. **出力 schema 案**: D（#2445 の3D追跡）が使う、frameごとの候補・支持カメラ・再投影誤差・品質mask・棄却理由。
5. **テスト設計と評価計画**:
   - Meiji 3cam のラベルと、#2442 の疑似正解（作成中）を使う。
   - 使っていない視点への再投影で検査する。

## 返すもの（日本語の最終メッセージ）
- 各論点の推奨案と根拠。試作・計測をした場合は数値と再現コマンド。
- ユーザーに確認すべき判断。court side の案と solver を、選択肢の形で示す。
- 実装の分割案（必要なら）。
- 試作コードを残す場合は、自分の worktree の branch に commit・push する（PR は不要）。
