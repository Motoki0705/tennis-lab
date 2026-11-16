# #1748 の続き（親からの指示）

## 比較入力の変更（親判断。証拠用Issue #2150 に記録済み）
C1の固定入力を、meiji_4camのclip_000〜002（Court停止）から **video_000/clip_035・clip_037** の2clip（全4camera）に替える。
この2clipは、2026-11-16の全件可視化で、現行の採用条件のまま4cameraとも採用された。
- 全件可視化の証拠: メインcheckoutの共有outputs `outputs/tennis_scene/evaluate/meiji4cam-court-review-20261116/summary.json`
- 入力manifestとSHA-256を固定し直す。
- 元datasetには書かず、outputs配下のコピーを処理する点は従来どおり。
- clip_000〜002の停止例の実測は、失敗例の記録として残してよい。ただし基準値としては扱わない。
- 途中の段（court_sideなど）で止まったら、閾値・camera・入力を変えて回避しない。停止理由を報告して止まる。

## 配置切替後の環境
- data/outputs/ckpt はD:へ移った。このworktreeのoutputsは、新しい空のpersistent outputsを指す。
- 11/15までの実測物は、E:側の旧outputsにだけ残っている。必要なものは新しいoutputsへコピーし、SHAを照合する。
- datasetがD:に無い場合は、`storage_tier hydrate` で取得する（repo rootの `.agents/ops/storage_tier/README.md` を参照）。

## やること
1. origin/mainへrebaseし、衝突を解消する。#2156・#2153・#2155・#2148がmerge済み。
2. knowledgeの連番を付け直す。000042は#2153、000043は#2155が使用済み。このPRは **000044以降** を使う。
3. 新しい入力で、現行の全段（Court・DINO・Ball（既定のi935）・ViTPose・CLIP・HMR2・GVHMRなど）について、cold／完成prefix再利用／全cache hitの基準値を測る。測定は共有training queue経由（`--resource all`、`--provider codex`、`--issue 1748`）。必要な項目は #1748 のAcceptance checklistに従う。
4. pre-commit・関連pytest・knowledge validatorを通し、pushしてCIの成功を確認する。PR本文を更新し、受入条件を満たしたらready for reviewにする。
5. 最終文で、結果・CI・残件を短く報告する。merge・issue closeはしない。
