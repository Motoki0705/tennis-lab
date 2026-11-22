## 担当範囲
#1748（C1）を完了させる。計測recorderの実装（C3のruntimeがそのまま使える形）と、現行clip逐次実行の基準値の実測・保存まで。

## 2026-11-15の親判断（#1734 合意revision 2）
- 比較入力は `data/tennis_multiview/processed/meiji_4cam/dataset/`（video_000、4camera、59.94fps、1920x1080）から選ぶ。現在clip_studioが全69clipを書き出し中（clip番号の若い順に完了）。基準値には、書き出しが完了したclipから、長さの異なる3clip程度（例: clip_000〜clip_002）を選ぶ。選んだclipの全ファイルが書き出し済みで以後変わらないことを、計測前に `dataset.json` の登録とファイルのSHA-256で固定する。書き出し中のclipは使わない。
- 書き出しプロセス（`clip_studio`、port 8765）と同期スクリプトを止めたり、dataset配下へ書き込んだりしない。出力はoutputs配下へ置く。
- ボール検出は現行既定（i935）のまま測る。#1734の3条件は同じボールモデルで比べる（X3D/CSN＋DPTは別子#2151）。

## 資源と制約
- GPU実行は必ず共有training queue（`.agents/skills/training-queue/SKILL.md`、`TRAINING_QUEUE_DIR=[local path]`）経由。WSL2・16GB GPUでVRAMがあふれるとホストが固まるので、peak VRAMを事前に見積もる。
- worktreeの `data`・`ckpt`・`outputs` はmainの実体へのsymlink、`.venv` はmainのvenv。`third_party` のsubmoduleは未初期化の可能性があるので、必要なら確認して対処し、その方法を報告する。
- 繰返し数・cache条件・実行順は計測前に決めてissue/knowledgeへ書く。初回ロード込みとwarmを分ける。

## 報告
PR URL、固定したcommit・入力のhash、基準値の表（総wall time、stage別、モデルload/unload回数と時間、peak VRAM/RAM）、knowledgeの記録先。
