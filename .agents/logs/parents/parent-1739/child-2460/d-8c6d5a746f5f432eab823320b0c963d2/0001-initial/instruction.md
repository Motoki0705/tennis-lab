# #2460 i2363 全69 clipの選手をダブルス幅で選び直し、ボール入りsceneへ反映（親 #1739、担当親 Joseph）

Issue #2460 の本文が仕様の正本。最初に `gh issue view 2460` と親 #1739 の本文（「2026-11-24 午後の判断（ユーザー）」）を読む。

## 起点

- このworktreeの起点 `980a902e` は、#2440（PR #2450、選別既定をダブルス幅に変更）と #2458（PR #2462、外部ボールの取り込みとscene切替）の両HEADをmergeしたcommit。両方のコードが入っている。
- #2458 の担当は、`[local path]` の全69 clipの `scene.json` と `annotation.json` をボール入りの新版に切り替え済み。
  - 成功55 clipは既存のカメラ幾何を使う。
  - 失敗14 clipは、既存校正と共有コート向きによるカメラ幾何で、選手軸が空。
  - 切替手順・記録は PR #2462 の本文と `src/tennis_scene/pipeline/imports/external_ball_*.py` を読む。
- #2440 の比較記録は `knowledge/runs/run-i2440-player-selection/`。
  - 新既定での再選別がその比較のmaskと全276視点で一致し、停止は14件から11件に減る見込み。
  - その見込みを、実際の再構成で確かめる位置付け。

## やること

- 保存済みの検出・追跡・pose・外見特徴を再利用し、`player_selection` 以降を必要な範囲だけ再実行する。
  - 対象は、選手選別・カメラ間対応付け・三角測量・body_view_selection・HMR2特徴・GVHMR・body_placement・scene_assembly。
  - 再実行しない段の結果は、artifact ID で同一であることを示す。
- 成功55 clipは既存の `camera_alignment` を固定する。失敗14 clipは #2458 のカメラ幾何を使う。`camera_alignment` は再実行しない。
- 出力は #2458 と同じ方式にする。
  - 新しい `scene_assembly` ID とexportを追加し、`scene.json`・`annotation.json` を原子的に切り替える。
  - 切替前の参照（#2458 版）を記録する。
  - ボールは #2458 の結果をそのまま引き継ぐ。
- まず clip_000（成功）と clip_020（停止）でパイロットを行う。CPUの段から確認する。
- 選手が復元できない clip は、空の選手軸と停止理由を明示して残す。静かなフォールバックは禁止。
- knowledge の run 記録に、clip別の次の項目を残す。
  - 3D選手フレーム数（#2458版と新版）
  - 選手数
  - 停止clipと理由
  - 処理時間

## GPU とメモリ

- GPUの段（HMR2特徴・GVHMRなど）は training queue（`TRAINING_QUEUE_DIR=[local path]`）経由だけで実行する。workerを起動・serveしない。
- 現在、GPUロックファイル（`[local path]` と `.gate`・`.slot-0`・`.slot-1`）のACLが壊れていて、開けない。
  - GPUの段に入る前に、`exec 9<>[local path]` で開けるか確かめる。
  - 開けなければ、CPUの段とパイロットの準備まで進めて、停止事項として報告する。
  - ユーザーには親から修正コマンドを依頼済み。
- 本日、巨大JSONを `jq` で丸ごと読み、WSLがメモリ枯渇で停止した。
  - 大きなJSON・npzは、必要な部分だけをPythonで読む。
  - 並列数は、他作業と共有するメモリ約40 GBを考えて抑える。

## 終わり方

- テストを設計してから実装する。pre-commit・pytestを通す。
- draft PRを日本語で出す。
  - baseは `main` にする（2つのPRに積んでいるため）。
  - 本文には、積んだ境界commit `980a902e27f819db7def3bd969a4ff8b8ebdf850` と、依存PR #2450・#2462 を明記する。
  - 本文1行目は `Closes #2460`。
- 最後にPR番号と完全なHEADを宣言する。GPU待ちで止まる場合は、その時点の状態と再開条件を報告する。
