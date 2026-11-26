選手枠への断片統合を [draft PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469) にpushしました。ブランチは `codex/parent-1739-issue-2445-primary`、HEADは `b5f97c893a26c4961e7daf93bb997db13cf9be23` です。CIは実行中です。

- **Meiji 3cam開発用4本**では、選別後のIDF1が76.09%→84.50%、precisionが99.94%→99.95%、recallが61.43%→73.18%でした。000/000の遠側は領域内実測が229→593フレームとなり、選別に入りました。
- **ID切替と欠測**では、新規switchは0件、選別前の全IDでは741→737件、選別後は0件でした。選別後のfragmentは190→387件に増えました。採点に入る実測区間が増え、その間の欠測が残っています。
- **meiji_4cam全69本**で各サイド2 IDを選別しました。4人の実測は前後とも43,726フレームで、020・058はそれぞれ651/651、765/765フレームでした。追加連結は0件です。統合版は `outputs/i2445/integrated/meiji4/` にあり、#2446の読込コードで `multiview_pose_tracks` v1として開けることを確認しました。 [計測と再現手順](https://github.com/Motoki0705/tennis-lab/blob/b5f97c893a26c4961e7daf93bb997db13cf9be23/knowledge/runs/run-i2445-player-slot-integration/README.md)

pytest 55件、pre-commit、knowledge検査は通過しました。予約未見3本は #2444 の既存ViTPoseジョブの完了待ちで、採点は0回です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"b5f97c893a26c4961e7daf93bb997db13cf9be23","resume_on_success":true,"next_step":"#2444の既存ViTPoseジョブの完了記録を照合し、全9カメラが揃ったら固定設定で予約未見3本を一度だけ採点してPR #2469に記録する。未完了なら既存ジョブの完了待ちを引き渡す。"}}