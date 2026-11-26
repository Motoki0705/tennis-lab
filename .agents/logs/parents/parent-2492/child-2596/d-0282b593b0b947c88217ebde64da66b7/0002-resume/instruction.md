# #2596 親レビュー: 3点を直してから push してよい

原稿5ページと既存7ページの導線を読み、表の数値を `conditions.csv`・`comparison.csv`・#2492 本文・#2238 で抜き取り照合しました。構成・出典・公開禁止事項は問題ありません。次の3点を直したうえで、**Wiki への commit・push を承認します**。

## 1. Top-down pose の表の振り分けが逆（必須）

`Top-Down-Pose-Model-Comparison.md` の「同期済みの速度・VRAMと固定枠精度」の主表（`pose_main`）に ViTPose-H **flipなし** の2行が入り、「ViTPose-H flipなしの補助比較」の表（`pose_noflip`）に **PMPose-H** の2行が入っています。

main の `knowledge/runs/run-meiji-topdown-pose-comparison/comparison.csv` の30条件は、`vitpose-h`（flipあり）・`rtmpose-m`・`rtmpose-x`・`pmpose-l`・`pmpose-h` の5モデル × PT16/TRT16 × batch 1/4/8 です。flipなしは補助条件です（本文の「主比較15条件と flipなし補助3条件」、Pareto の「各 batch の10条件のうち9条件が残り、PMPose-H PT16 は…」とも一致させる）。

- 主表を `comparison.csv` の10行（5モデル × 2 backend）にし、PMPose-H PT16・TRT16 を主表へ移す。
- 補助表を ViTPose-H flipなし PT16・TRT16 の2行にする。
- 移したあと、全行の値を出典（主表は `comparison.csv`、flipなしの速度は `run-person-model-speed-low-precision/cases.csv`、精度と PT32 差は `accuracy.csv`・`vitpose_accuracy.json`）ともう一度照合し、結果を evidence に追記する。
- 表の直後の説明文と、概要ページの表・候補表に、この取り違えに由来する記述がないかも確かめる。

## 2. 文体の不統一（3か所）

次の文がです・ます調の中で常体になっています。「可視化はローカルに保管しており、Wiki には載せていません。」のように直してください（動画の箇所は「動画は…」）。

- `Person-Detector-and-ECPose-Comparison.md`「小人物・遮蔽と保存した確認」の1項目め
- `Top-Down-Pose-Model-Comparison.md`「難条件と欠測」の後の段落
- `Body-Recovery-FastHMR-vs-GVHMR.md`「同期比較用の成果と目視の状態」の2段落め

## 3. push 直前に #2593 と #2491 の状態を確かめる

push の直前に `gh pr view 2593` と `gh issue view 2491` を確認する。

- #2593 がマージ済みで #2491 が閉じていれば、概要ページ「Issue・PR と詳細ページ」の「#2491 の速度による採否は確定し、記録専用 PR で反映する段階です」を、#2593 で記録して #2491 を閉じた旨に直す。FastHMR の判断の出典に main の `knowledge/nodes/tennis_scene/000095-run-fasthmr-camerahmr-pt16-speed.md` を足してよい（#2494 の記録へのリンクは残す）。
- まだなら、そのままでよい。

## push

直したら、禁止パターンの検索と内部リンク・main リンクの検査をもう一度通し、`git pull --rebase` してから commit・push する（commit メッセージは日本語、`Co-Authored-By` 等は付けない）。push 後に新規5ページの公開 URL が開けることを確かめ、commit hash と、3点それぞれの対応内容を最終回答に書いて `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。
