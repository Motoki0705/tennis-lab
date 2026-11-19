# #2151 の続き（親からの指示）

## 人間の判断
2026-11-16、学習時validationとの一致は「現状の証拠で受け入れる」と人間が判断した（証拠用Issue #2150 に記録済み）。PR本文と受入記録の「履歴一致：未達」を、未達の数値はそのまま残したうえで「人間が現状の証拠で受入れ（2026-11-16）」と書き換える。許容差や数値を事後に変えない。

## 配置切替後の環境
- data/outputs/ckpt はD:の実ディレクトリへ移った。このworktreeのoutputsは新しい空のpersistent outputsを指す。
- 11/15までに作った比較動画・計測物は、E:側の旧outputsに残っている（旧main outputs。worktree内で `readlink` しても今は辿れない。必要ならrepo rootの `.agents/ops/storage_tier/README.md` に従って場所を確認する）。PRとknowledgeで動画の場所を「旧outputs（E:）」と正しく書き、必要な動画は新しいoutputsへコピーしてSHAを再照合する。
- ckptのscene用artifactは、新しい `ckpt/ball_detection/` に存在するか確かめ、無ければ停止して報告する（勝手に別の重みで代用しない）。

## やること
1. origin/main（#2153・#2156・#2148がmerge済み）へrebaseし、衝突を解消する。
2. knowledgeの連番を付け直す。`knowledge/nodes/tennis_scene/000042` は #2153 が使った。このPRは **000043** を使う（#1748は000044以降を使う）。run bundle名・リンク・summaryも合わせる。
3. pre-commit・関連pytest・knowledge validatorを通し、pushしてCIの成功を確認する。
4. コート検出の失敗（court_region_unavailable）への対処は、人間が各clipの結果を見て判断する。このPRでは対処しない。3D比較は今ある条件付き結果のままでよい。
5. 終わったらPRをready for reviewにして、最終文で変更点・CI・残件を短く報告する。merge・issue closeはしない。
