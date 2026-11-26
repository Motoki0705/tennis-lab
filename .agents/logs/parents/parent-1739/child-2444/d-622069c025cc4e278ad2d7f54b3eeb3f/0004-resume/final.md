Meiji 3camの開発4 clipで、micro pair F1は **0.8050 → 0.8515**、誤リンクは **1,045件**でした。試作のF1 0.8485以上・誤リンク1,102件以下の両方を満たしました。肩・腰で人物対応を採点し、局所的な関節の異常はマスクするよう変更しました。

- 試作から失った正解リンクは2,458件、新たに拾ったリンクは980件でした。失った内訳は骨長1,280件、MILP競合1,142件、残差33件、骨長と残差の両方3件です。2,448件は少なくとも一方が遠側でした。 [clip・役割・近遠別の集計と再現手順](https://github.com/Motoki0705/tennis-lab/blob/5abec5a715d485ebd13c53112a9d764ba3eef08d/knowledge/runs/run-multiview-pose-link-recall-20261124/README.md)
- Meiji 4camは全69 clip・46,228フレームで候補を生成しました。割当率は86.02% → 85.90%、4 CPUプロセスの再計測は329.00秒 → 367.68秒でした。今回の2回の計測で全69 clipの成果物digestが一致しました。
- 未使用視点で採点できた関節は75.68% → 80.40%に増え、再投影誤差のP90は27.81 px → 43.64 pxに悪化しました。関連pytest 101件、ruff、mypy、pre-commit、knowledge検査は通過しました。

予約未見の採点は0回です。3 clipとも追跡前のViTPoseが必要で、`video_001/clip_003` はcam0の校正とボール由来の向きも欠けています。このclipの入力補完可否を親に判断してもらい、固定設定で未見を一括採点します。#2442の疑似正解によるダブルス採点も残っています。

同じブランチ `codex/parent-1739-issue-2444-primary` に `5abec5a715d485ebd13c53112a9d764ba3eef08d` をpushし、PR本文を更新しました。新HEADのCIは実行中です。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449) · [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45245218907)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"5abec5a715d485ebd13c53112a9d764ba3eef08d","resume_on_success":true,"next_step":"予約未見clipの追跡前ViTPoseを準備し、video_001/clip_003のcam0校正とボール由来の向きの補完可否を親と確認した上で、固定設定で一括採点を一度だけ行う。"}}