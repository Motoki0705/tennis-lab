# #2445 3D状態による時間追跡と3D軌跡による選手選別（親 #1739、担当親 Joseph）

Issue #2445 の本文が仕様の正本。最初に `gh issue view 2445` と、親 #1739 の本文（特に「2026-11-24 午後の判断（ユーザー）」）を読む。

## 起点と依存

- このworktreeは #2444（PR #2449、branch `codex/parent-1739-issue-2444-primary`）のHEADから始まっている。#2444 の `src/tennis_scene/multiview_pose/` が出す frameごとの3D骨格候補（`frame_pose_candidates` v1: source_rows, positions_m, joint_valid, scores, ambiguous など）が入力。まずそのREADMEとコードを読む。
- #2440（PR #2450、branch `origin/codex/parent-1739-issue-2440-primary`）は別PR。既存経路の選別既定をダブルス幅（|x| ≤ 5.485 m、|y| ≤ 16.885 m、clipの25%滞在、人数は `players_per_side`）にした。その比較結果は `git show origin/codex/parent-1739-issue-2440-primary:knowledge/runs/run-i2440-player-selection/README.md`（同ディレクトリの comparison.json・loss.json・singles.json）で読める。このbranchへmergeしない（既存経路の規則の変更は #2440 の担当）。必要な定数は新モジュール側で明示的に持ち、#2440 と同じ値であることを記録で示す。
- #2442 の疑似正解は棄却、#2443 は取消。どちらも評価に使わない。

## 方針

- 追跡は**一定の時間窓での最適化**（ユーザー承認済み）。窓の長さ・重なり・確定の仕方は設計で決め、根拠（実データの統計など）を knowledge に残す。frameごとの候補と接戦（ambiguous・代替解のgap）を前後の文脈で解く。
- 出力では、実観測による再構成と遮蔽中の予測・補完を区別し、生成・消失・遮蔽・再出現・IDの曖昧さを明示する。静かなフォールバックは禁止。
- 選別は3D軌跡の滞在と運動から行い、シングルス・ダブルスは明示的な設定で扱う。根拠（滞在・領域・理由）を診断に残す。

## 評価

- 定量: Meiji 3cam の人手ラベルで、3D IDを各カメラへ投影した IDF1・switch・fragment と、選別の precision・recall を採点する。#2444 で予約した評価用clipの扱い（開発用と一度きりの holdout の区別）を守る。holdout の採点は1回だけ。
- 定性（meiji_4cam）: 後で #2446 が比較ページ（#2459）に段階2として加え、ユーザーが評価する。そのために、clip単位で読める出力（人物ID付き3D関節・選別結果・品質mask）の保存形式を README に書く。
- 停止した clip_020・clip_058 で選手4人を保てることを確認する。

## GPU とメモリ

- GPUの処理は training queue（`TRAINING_QUEUE_DIR=[local path]`）経由だけ。現在、GPUロックファイルのACLが壊れていてGPUジョブは開始できない。GPUが必要な段は、投入前に `[local path]` を開けるか確認し、開けなければ停止事項として報告する。ワーカーを直接起動しない。
- 本日、巨大JSONを `jq` で丸ごと読みWSLがメモリ枯渇で停止した。大きなJSON・npzは必要な部分だけをPythonで読む。並列数はメモリ約40 GB（他作業と共有）を考えて抑える。

## 終わり方

- テストを設計してから実装する。pre-commit・pytest を通す。
- PRは日本語、`main` 向けではなく **base を `codex/parent-1739-issue-2444-primary`** にしたdraft PRで出し、本文1行目は `Closes #2445`。knowledge の run 記録を同じPRに含める。
- 最後にPR番号と完全なHEADを宣言する。
