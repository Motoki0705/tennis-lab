# #1757: 3clipでの実GPUの回帰検証とbenchmark（親 #1734）

Issue #1757の本文を読んでください。ただし、2026-11-19に人間の指示で範囲が変わりました（親issueのnoteに原文があります）。次の変更が本文より優先します。

## 変わった範囲
- 対象はmeiji_4camのvideo_000の3clip（clip_000・clip_027・clip_037）だけ。69clipは対象外。
- benchmarkにはball推論も含める（ball_detection → ball_points → ball_triangulationを含む全段）。
- court_sideは、新しいプレーヤー証拠の経路（#2269、`court_side.evidence: player`、branch `codex/court-side-player`）で判定する。この経路のテストを兼ねる。ball証拠での判定 `[F,F,T,T]` が正しいことは確認済みなので、player証拠の結果がこれと一致するかを記録する。
- 期限: 2026-11-19 15:00（JST）までに、benchmarkを実際に動かせる状態にする。具体的には、計測スクリプトと条件の定義をpushし、最初の条件のGPU計測をtraining queueに投入して、走り始めたことを確かめる。

## 土台
- branch `codex/video-batch-benchmark`、worktree `.claude/worktrees/video-batch-benchmark`。#1752の一括CLI（`codex/video-batch-cli`、HEAD 8ab460050）の上に積んでいる。
- まず `origin/main` をmergeする（#2225と#2226がmainに入っている。#2226がまだmerge中なら `origin/codex/person-feature-artifacts` をmergeする）。
- #2269のPRが開いたら（14:10ごろの予定）、`origin/codex/court-side-player` をmergeして使う。それまでは、player経路を差し込む場所を空けて準備を進める。15:00に間に合わない場合は、ball証拠のcourt_sideで計測を始めない。その状況を報告する。

## 条件
- 比較する条件は次のとおり。
  1. 現行clip逐次: clipごとに標準の単一clip入口で実行する。
  2. モデル再利用（batch size 1）: #1752の一括CLIで3clipをまとめて実行する。
  3. モデル再利用＋batch化: #1755の実装を待つ。今は条件の定義だけを用意し、計測は#1755のmerge後に行う。
- #1756（CLIP embeddingの再利用）は、別の条件として、merge後に追加する。
- すべての条件で、入力・重み・精度・保存条件をそろえる。精度は、人間が本番用に決めたViTPose・DINOのfp16に合わせる。ダブルス（players_per_side=2）にする。court_kpはvideo_shared、court_sideはplayer証拠を使う。
- 計測する量: 総wall time、clips/hour、camera-frame/s、モデルの読み込み回数、peak VRAM/RAM、中間保存と最終公開の時間。初回のロードを含む条件とwarmの条件を分け、繰り返しのばらつきを示す。繰り返しの回数は時間の予算から決めて、理由を書く。
- 出力の比較: 実行の前に、dtype・単位ごとの許容誤差を決めてcommitする。連続値はその許容誤差で比べる。ID・採否・欠測mask・停止理由が変わったら、原因と下流への影響を調べる。ballの結果も比較に含める。
- 計測には#1748のrecorder（PR #2154、`codex/video-batch-baseline`）が使える。必要な部分は取り込むか参照する。#1748の過去の実測（knowledge 000046・000048）は、clipや条件が違うので比較の基準にしない。

## 実行
- GPUは必ず共有training queue経由で使う。worktreeでenqueueし、`TRAINING_QUEUE_DIR` はrepo rootの `.training_queue/` にする。
- CUDAのエラーやハングが起きたら、再試行せずに止めて報告する。原因は人間が調べる。
- 長時間の計測は、投入して最初の条件が安定して走り始めるところまで見届けてから、このターンを終えてよい。終える前に、queueのjob ID・ログの場所・終了の確かめ方を報告する。
- 15:00の時点の状況を、PR（Draft）の本文に書いておく。

最後に、PRのURL・HEAD・投入したjob・許容誤差のファイル・未完了の条件を簡潔に報告してください。
