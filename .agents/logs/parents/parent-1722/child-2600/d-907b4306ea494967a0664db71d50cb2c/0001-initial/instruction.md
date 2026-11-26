# #2600 の実装・ポストトレーニング・アブレーション

担当Issue #2600 の本文を受入条件の正本とする。親 #1722。入力の multiview-v3 と SceneResult v3 は #2605（PR #2610、未マージ）が作る。

## 土台

- 未マージの #2610 の branch `codex/parent-1722-issue-2605-primary` に積む。起動時の `--from` に渡した commit を境界として PR 本文に記録し、PR の base は同 branch にする。#2610 のマージ後の載せ替えは親が指示する。
- #2610 はレビュー中で、v3 の読み込みAPIが変わる可能性がある。v3 の読み込みは `src/tennis_scene` の公開API（`SceneResult` とその検証・読み込み関数）だけを使い、#2610 の内部関数に依存しない。

## 固定した事実

- 事前学習済みモデル: `.worktree-outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt`（メインcheckoutからの相対パス）。SHA-256 `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0`、契約 `ball_refiner_3d.events.v3`。実体は `[local path]` にある。ハッシュが違う、または読めない場合は推測で代替せず停止して報告する。
- 入力: `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3`（起動時点で #2605 の担当が最終HEADでの再生成・照合を進めている）。学習を queue に投入する前に、PR #2610 本文の最新の実行証拠にある manifest の SHA-256 と、手元の manifest が一致することを確かめる。一致しなければ投入せず停止して報告する。manifest の分割（train 47・val 11・test 11、seed 42、ネット接触クリップで層化、#2542 の採点用6クリップ 027・028・040・043・056・064 は test）をそのまま使い、作り直さない。manifest の SHA-256 を全成果物に記録する。
- #2542 の独立ラベル: knowledge の `knowledge/nodes/tennis_scene/000096-run-i2542-meiji-ball-events.md` と run bundle を参照する。test の6クリップでの併記採点に使う。

## 人間が承認した学習・評価の扱い（2026-11-26「推奨でよい」）

- certain と probable は正例。uncertain はその前後を無視区間にする（正例にも負例にもしない）。`ball_event_coverage` が false の判定不能区間は loss から除外する。
- ポイント外のイベントも正例。評価は全イベントとポイント中だけの両方で報告する。
- Gaussian 幅は事前学習と同じ固定値。`frame_uncertainty` は使わない。
- toss・fence など3クラス以外は扱わない。
- 調整は val だけで行う。test による選択・閾値決定は禁止。

## 作業の区切り

1. 3方式（A Shared / B Per Player / C SAB）の入力融合、共通SMPL前処理、v3 を読むポストトレーニング入口、評価、テストを実装する。Issue 本文のテスト（初期出力一致・置換不変性・欠損mask・全選手欠損・勾配・旧経路回帰）を設計してから書く。SMPL の手首位置は正規の順運動学でコートに置き、単位・向き・マスクを実データ1クリップで検証して数値を残す。commit・push して draft PR を作る。
2. 主比較（trunk・head・ball embedding を凍結し、新規 player encoder と fusion だけを学習）を3方式×共通3 seedで、training queue（`TRAINING_QUEUE_DIR=[local path]`、worktree から `add`、queue worker は起動しない）に投入する。事前学習済み ball-only モデルの同じ test 推論も参照基準として投入する。全 job ID を `PARENT_RESULT` の `wait` で宣言して終了する。学習予算・optimizer・checkpoint 選択規則は全方式で同じにし、val で決めた値と根拠を PR に書く。
3. 結果を受け取ったターンで、Issue の指標（クラス別 P/R/F1、macro-F1、時刻MAE、TP/FP/FN、条件別、#2542 ラベルでの併記、パラメータ数・推論時間・メモリ、seed 間差）を集計し、代表例の可視化を作る。PR の実行証拠を最終HEADで更新し、knowledge 記録は `.agents/skills/parent-session/references/workers.md` の「ナレッジ記録PRの作成」 の手順で別PRにする。全層低学習率 fine-tuning を行うかは主比較の結果を見て判断し、行うなら3方式同条件の補助段階として分ける。

## 制約

- `data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events` はユーザーの成果物で確認中。読むだけにし、変更・移動・削除しない。
- 静かなフォールバックを作らない。SMPL・root・ball の欠測は mask で明示し、実観測と混同しない。
- GPU は training queue 経由だけで使う。CPU の長時間検証は `parent.local_jobs start` を使う。
- 既存の ball-only 経路と checkpoint の読み込みを壊さない。
- 受入条件の変更が必要になったら、自分で決めずに理由を書いて停止する。
