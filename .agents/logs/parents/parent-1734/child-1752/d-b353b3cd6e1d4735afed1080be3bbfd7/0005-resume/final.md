`origin/main` を統合し、[PR #2227](https://github.com/Motoki0705/tennis-lab/pull/2227)を更新しました。baseはmain、Draftを維持しています。

- HEAD: `8ab460050dc7dd1c3dcdf28f3433ef6f1732438b`
- 衝突は、mainの精度設定・DINO pin・共有court side検証と、一括CLIの失敗分離・公開時のlock照合を両立して解消しました。
- CPUテスト **1,551件成功**。pre-commit成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44348542631)はPythonを含む全ジョブ成功、[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44348542890)も成功しました。
- 本番コマンドは変更不要です。CPUで全69clip・2,622 nodeの単一計画、両モデルのfp16伝達、共有sideの検証を確認しました。
- GPU・本番投入は行っていません。既報のBall補助evidence差は未解決です。validatorは指定・試行・完了とも0回です。