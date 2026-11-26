# PR #2461: #2469 の修正版を取り込み、比較側の互換表を組み直す（正式レビュー round 1 の採用3）

#2469 の修正版の HEAD は `6f69068f88635cb8220e524f0045e2665259eed6`（最新mainを取り込み済み、CI成功）。マージ順は #2469 → #2461。

1. `origin/codex/parent-1739-issue-2445-primary`（`6f69068f`）をこのbranchへ merge する（rebase はしない）。`origin/main` も取り込む。競合は `source_identity.py`・`inventory.py`・FastHMR と FP32 の互換テスト2ファイルなど。両方の意図を保つ形で解く。
2. 比較側の互換表 `pose_comparison_cache_compatibility.json` を、新しい基点（#2469 を含むmain相当）と合流後の実ファイルに結び直す。両PRの回帰テストを残し、実ソースで次のことを確かめる。
   - 完全一致時に、既存 recipe のキャッシュ識別が16条件で main（#2469 を含む想定）と一致する。
   - 対象ファイルを編集すると失効する。
   値をただ再生成して合わせない。
3. knowledge の連番は #2469 の最新（tennis_scene `.sequence` 104）に合わせる。このbranchでノードを足さない。
4. 関連テスト、pre-commit、knowledge 検査（`--base-ref origin/main`）を通して push し、PR本文を更新する。本文に「#2469 を取り込み済みで、#2469 のマージ後に差分が #2461 だけになる」ことを書く。
5. ベンチマークとデータセットには触れない。GPU は使わない。

新HEAD（完全SHA）、解いた競合、互換の確認結果、テスト結果を報告し、CI待ちで終了する。
