# 親からの続き（2026-11-24 午後、担当親 Joseph）

ユーザーが疑似正解ラベルについて「品質はあまりよくないので、棄却します」と判断した。人による抜き取り確認の結果ではなく、ユーザーが結果を見ての判断である。#2443 は取消、#2440 は疑似正解による採点を外す。

## このターンで行うこと
1. **PR #2448 を knowledge の記録だけにする。** `src/`・`tests/` の変更（`src/tasks/player_association/pseudo_annotation/`、レビュー画面の変更など）はすべて外し、`origin/main` を基点に作り直す。自分のブランチへの `git push --force-with-lease` は可。
2. 残す記録は `knowledge/runs/run-i2442-meiji-pseudo-association-pilot/` と `-full[local path]`。README に次を書く:
   - 方式（層別抽出・4視点合成画像・旧IDを伏せる・gpt-6.1-terra xhigh）、規模・時間・費用、Meiji 3cam での一致率。
   - **2026-11-24 にユーザーが品質不足として棄却した**こと、採点に使わないこと。抜き取り確認は未実施のまま終わったこと。
   - 生成物の場所（ローカル、`outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association/`）。コードはマージしていないので、再現手順は「このPRの元のブランチの履歴（コミットSHAを明記）」を参照する形にする。
   - 同じ方式をもう一度試すなら何を変えるべきかが分かる範囲で、分かっている限界（4cam で人数が増える、遠くの選手が小さい等）。推測は推測と書く。
3. PR #2448 の本文を日本語で書き直す。先頭は `Refs #2442`（`Closes` にしない。issue は親が not planned で閉じる）。knowledge検査と pre-commit を通し、PR番号と完全な HEAD を宣言して終了する。
