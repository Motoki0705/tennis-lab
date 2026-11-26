# #2442 疑似正解: 抽出・注釈基盤と試行（全体実行はしない）

担当issue #2442 の本文が範囲と受入条件の正本。親 #1739 の「全体方針・合意事項」（2026-11-24 の再設計）を読む。
このターンは、基盤の実装と**試行**まで。全clipの注釈は、ユーザーが予算を確認するまで実行しない。

## 入力
- 既存経路の全clip実行: `knowledge/runs/run-i2363-trt-player-video/README.md`。
  - 実体は `[local path]`。
  - このworktreeは前の親（#2278）の解除後に削除されうる。使う成果物（全人物の検出・追跡・選別・対応付け・カメラ・clipの元動画の識別）は、自分のworktreeの outputs へ複製し、sha256 を記録する。
  - 停止した clip は 020・058（選別）と、「近すぎて決められない」の9 clip。どこまで成果物があるかを確認する。
- ラベル形式: `src/tasks/player_association/evaluation/labels.py`（`player_association_labels_v1`）、採点は `src/tasks/person_tracking/evaluation.py` と `src/tasks/player_association/evaluation/metrics.py`。
- 人の確定ラベル: Meiji 3cam（`src/tasks/player_association/evaluation/dataset_labels.py`、`src/tasks/player_association/review/README.md` の現物の表）。
- 選別で落ちる原因の分析（#2440 本文）。疑義frameの規則の出発点に使う。

## 作業
1. **疑義frameの抽出**を、再利用できるモジュールとして実装し、テストを書く。
   - 規則と閾値は設定で明示する。
   - 層ごと（選別外れ・コート近く、1視点、未割当・再投影不一致、ID switch・断片、停止clip、無作為の確信frame）に frame を出す。
   - 同じ入力からは同じ集合を出す（seed を固定）。
2. **注釈の形式とingest**:
   - `player_association_labels_v1` と既存の採点コードで読めるようにする。
   - 注釈した frame の集合と出自（モデル名・effort・プロンプトhash・入力hash）を明示的に保存する。schema を拡張するなら版を上げ、旧版の読み替えはしない。
   - 形式違反・矛盾は拒否する（黙って補完しない）。
3. **gpt-6.1-terra への注釈依頼の仕組み**:
   - 依頼の単位は「1 clip の1区間」。同期した全カメラの frame と、番号付きの box を画像で渡す。
   - terra 用の作業指示書を作る。人物IDは clip 内でカメラ・時間をまたいで一貫させる（ダブルスは選手4人）。選手でない人・曖昧な box の扱いも書く。
   - 実行は `codex exec -m gpt-6.1-terra -c model_reasoning_effort="xhigh"`。同じ clip に2つの依頼を同時に走らせない。並列は最大4。
   - この worker の中から `codex exec` を起動できない場合（認証・sandbox）は、実行用 script と入力一式を用意して、その旨を宣言して終了する。親が起動する。
4. **試行**（親が予算確認の材料にする）:
   - (a) 精度の較正: Meiji 3cam の人の確定ラベルがある clip から、frame を数十〜百程度選んで terra に注釈させ、人のラベルとの一致率を測る（box 単位・人物ID単位）。
   - (b) meiji_4cam: clip_020・clip_032 と、問題のない clip 1本で、疑義frameを注釈させる。
   - 各試行で、1 frame あたりの時間と使用量を記録する。codex の rate limit 情報や token 数など、取れるものを記録する。
   - 全体（69 clip の疑義frame＋無作為層）の frame 数と、費用・時間の見積もりを作る。
5. **抜き取り確認の案**: 人が review ツールで確認する件数の案を、層別に出す。人の確認はこのターンでは行わない。

## 返すもの
- PR（draft可、本文は日本語）: 実装・テスト・`knowledge/runs/` の試行記録。
  - pytest の対象を通し、commit・push する。PR番号と完全な HEAD を宣言する。
  - GPU を使う処理があれば training queue 経由で投入する。queue worker は起動しない。
- 最終メッセージ（日本語）:
  - 層別の疑義frame数。
  - 試行の一致率。
  - 1 frame あたりの費用・時間と全体の見積もり。
  - 抜き取り件数の案。
  - ユーザーに確認すべき点。
