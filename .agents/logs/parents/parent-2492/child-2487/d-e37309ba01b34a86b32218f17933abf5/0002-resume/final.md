RF-DETR-Lの3条件を固定6,844画像・26,276選手BBoxで採点しました。生出力を全件再採点し、保存結果との一致を確認しました。

| backend | Recall@0.50 | Recall@0.75 | PT32との差・座標誤差P95 |
|---|---:|---:|---:|
| PT32 | 96.81% | 42.06% | 基準 |
| PT16 | 96.82% | 42.03% | 0.039 px |
| TRT16 | 96.81% | 42.09% | 0.404 px |

PT16・TRT16ともPT32との出力照合基準を満たしました。疑似BBoxの各辺余白を5%から15%に広げると、PT32のRecall@0.75は13.94%から56.87%に変わりました。

実行HEADは `64fde00dca465a7c2bb27881805f971d0bbaaee4`、記録コミットは `88869d882fc6679eabf9ab183ed1d9f0f0e5d08d` です。branchは `codex/parent-2492-issue-2487-rfdetr`。生予測・採点・engineの実体パスとSHA-256を保存し、knowledge検査を通過しました。[実行記録とSHA-256](https://github.com/Motoki0705/tennis-lab/blob/88869d882fc6679eabf9ab183ed1d9f0f0e5d08d/knowledge/runs/run-meiji-rfdetr-large-accuracy/results.json)・[#2487への記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7130929188)

残りは、#2496 の固定速度manifest公開後の速度・VRAM計測と、RF-DETR-2XLの3条件です。2XLの再開には、ユーザーによるPML 1.0への同意、ローカル評価を許すRoboflowアカウント・プランの確認、公式 `rf-detr-xxlarge.pth` の取得先が必要です。[PML 1.0](https://github.com/roboflow/rf-detr_plus/blob/31474876e519dacef3ba320bc8977f1483d69b34/LICENSE)・[Roboflowアカウント](https://app.roboflow.com/)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}