# #2491 再開: ユーザー判断「置き換えは見送り」を記録して #2491 を閉じる

## 何が決まったか

2026-11-26、ユーザーが判断した: 「FastHMRの速度優位性がないなら置き換えは見送りです。」

親が main の記録で条件を照合した（親issueの記録: https://github.com/Motoki0705/tennis-lab/issues/2494#issuecomment-7178978237 ）。

- 共通ハーネスの wall P50（1人・1フレーム）: FastHMR CameraHMR PT16 68.18 ms、現行 HMR2特徴＋GVHMR PT16 30.77 ms、現行 TRT16 25.55 ms。
- 系列 batch 1・2・4 のすべてで FastHMR PT16 が現行 PT16 より遅い（2.22・2.57・1.68倍）。
- FastHMR の TRT16 は書き出しで停止して実行不能。
- 速度の優位がある条件はないので、置き換えは見送り。

## 扱い（親の方針）

- 既定 recipe は GVHMR のまま。明示選択の `recipe=fasthmr` 経路は #2491 の受入条件（両経路を残す）どおり main に残す。コードは変えない。
- 24本の同期比較動画での目視評価は、この判断で不要になったので行わない。記録には「目視評価は未実施（速度条件で見送りが決定）」と書く。品質を評価したとは書かない。

## やること

1. 最新の origin/main から、新しい worktree `.claude/worktrees/parent-2492-issue-2491-decision`（branch `codex/parent-2492-issue-2491-decision`）を作る。メイン worktree と既存の worktree のファイルは変えない。
2. knowledge の FastHMR 記録に判断を反映する。少なくとも次の2つ。書き方は knowledge-control skill に従う。
   - `knowledge/nodes/tennis_scene/000095-run-fasthmr-camerahmr-pt16-speed.md` の「定性品質と採否は人間の目視確認待ち」の記述を、上の判断（日付・ユーザー発言・根拠の数値・目視未実施・経路は残す）に置き換える。
   - `knowledge/nodes/tennis_scene/000094-run-fasthmr-camerahmr-pt16-review.md` で目視所見や採否を「確認待ち」としている記述があれば、目視評価は未実施で、速度条件により見送りが決まったことに直す。
   - ほかに「人間の確認待ち」「採否未定」を書いた FastHMR の記録（knowledge の summary や registry を含む）があれば同じく直す。測定値・ハッシュ・再現手順は変えない。
3. knowledge の検証（ノード検証と再現パス検査）を通す。差分は knowledge の記録だけにする（記録専用PR）。
4. 日本語の PR を作る。本文の先頭行は `Closes #2491`。続けて、何を直したかを短く書き、`## 実行証拠` の見出しの直下に `json` のコードブロックを1つ置く（`head_sha` は検証した HEAD の完全 SHA、`command` は実際に実行した検証コマンド、`exit_code`、`result`）。
5. 最終回答に PR 番号と HEAD を書き、CI を待つので `PARENT_RESULT` は `waiting` / `ci` で1行返す。
