#2444 のフレーム別3D骨格復元を実装し、[PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449) を作成しました。ボール由来の動画共有判定を固定し、全身の関節残差と骨長を検査した候補を MILP で排他的に割り当てます。D に渡す `frame_pose_candidates` v1 に、元検出行・支持視点・品質mask・棄却理由を保存します。 [入出力契約](https://github.com/Motoki0705/tennis-lab/blob/f58984e7b41d4559d976b365a44559403d83e791/src/tennis_scene/multiview_pose/README.md)

- **Meiji 3cam** の micro pair F1 は試作の **0.8485 → 0.8050**。誤リンクは1,102件から906件に減り、正解リンクの取りこぼしは4,500件から5,978件に増えました。clip別のF1は `000/000` が0.7415、`000/007` が0.8935、`001/001` が0.7760、`002/013` が0.8745でした。
- **未使用視点への再投影**は、1台ずつ外して再実行し、対象関節の **75.68%** を採点できました。誤差は中央値 **9.59 px**、P90 **27.81 px** でした。
- **meiji_4cam** は全69 clip・46,228元フレームで候補を保存し、全検出行の **86.02%** を割り当てました。4 CPUプロセスで入力照合から保存・読戻しまで **329.00秒**。再実行で全69 clipの配列SHAが一致しました。 [評価条件・詳細・再現コマンド](https://github.com/Motoki0705/tennis-lab/blob/f58984e7b41d4559d976b365a44559403d83e791/knowledge/runs/run-multiview-pose-meiji-20261124/README.md)

ローカルでは pytest **98件**、ruff、mypy、pre-commit、knowledge検査が通過しました。#2442 の確定済み疑似正解でのダブルス採点が残るため、#2444 は開いたままです。

ブランチは `codex/parent-1739-issue-2444-primary`、HEADは `f58984e7b41d4559d976b365a44559403d83e791` です。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45227001842) は起動済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"f58984e7b41d4559d976b365a44559403d83e791","resume_on_success":false}}