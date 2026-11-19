## 担当範囲
#2151 を完了させる。本文の具体的な変更内容とAcceptance checklistのとおり。

## 補足
- 2026-11-15に人間がこのモデルを選んだ。候補比較は済んでいるので、別モデルへの変更や再比較はしない。
- 重みはDrive（tennis-drive skill、`--rclone-config $HOME/.config/tennis-lab/rclone-drive.conf`）から `outputs/` 配下へ取得し、完了記録のSHA-256と照合する。
- encoderは学習commit `13dff57d4`（origin/codex/ball-capacity-resume-20261108）と固定source `db0485048` の実装が正本。PR #1897（open、#1859の別親が担当）に同じencoderの移植がある。#1897は書き換えず、merge状況を確認して本文の方針で決める。判断と理由を報告する。
- GPU実行は共有training queue（`TRAINING_QUEUE_DIR=[local path]`）経由。WSL2・16GB GPU。720p・T32の推論でVRAMを見積もってから実行する。
- 実動画確認には `data/tennis_multiview/processed/meiji_4cam/dataset/` の書き出し完了済みclipを読み取り専用で使う（書き出し中。未完了clipは使わない。datasetへ書き込まない）。worktreeの `data`・`ckpt`・`outputs` はmainの実体へのsymlink。`third_party` のsubmoduleは未初期化の可能性がある。
- 比較動画は人間の品質確認用。PR本文に置き場所を書く。

## 報告
PR URL、一致確認の結果、決めた存在閾値とFPR/FNR、meiji_4camでの比較表と動画の場所、推論時間とpeak VRAM。
