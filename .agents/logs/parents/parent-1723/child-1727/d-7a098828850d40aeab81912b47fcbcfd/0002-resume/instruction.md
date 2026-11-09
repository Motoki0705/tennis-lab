## 人間の確認結果（親が記録済み: #1727 のコメント）
2026-11-09、人間は動画を見て「実装としての問題はないと思います。ただし品質としての問題があります。それはブレが激しいということです。」と判断した。ブレ補正は別の子 #1776 で実装し、補正後のカメラで再描画した動画で改めて人間の受入を受ける。このPRのコード（Blender描画・package・公開処理）は #1776 の再描画の土台になるので、先にmergeできる状態にする。

## このターンでやること（再描画はしない）
1. PR #1759 のbranchは現在のmainと `src/utils/configuration/inventory.py` で衝突している（mainで追加された `src.tasks.ball_detection.annotation.video.agent.scripts.client` の項目と、このPRの `src.tennis_scene.visualization.blender.shots.player_view` の項目）。mainを取り込み、両方の項目を残して解消する。関連テストとpre-commitを通し、pushしてHEADのCIを成功させる。
2. PR本文の `Closes #1727` を `Refs #1727` に変える。#1727 の人間受入（ブレ補正後の再描画の確認）はまだなので、merge でissueが閉じないようにするため。本文には、人間の判断（実装は問題なし、ブレの品質指摘は #1776 で対応）を1〜2行で追記する。draftは解除してよい。
3. 終わったらHEAD・CIの結果を短く報告する。動画の再描画・Colabの使用はしない。

最終報告とPR本文にはローカルの絶対パスを書かない。
