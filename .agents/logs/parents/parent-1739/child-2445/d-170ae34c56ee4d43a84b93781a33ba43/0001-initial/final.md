#2445の実装をdraft PRで提出しました。ブランチは `codex/parent-1739-issue-2445-primary`、baseは #2444 のブランチです。HEADは `7b2c2dfd2345b1a1bd4e3473a75d6c5ff2bfdf44` で、CIが起動しています。 [PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469)

1.2秒の時間窓を0.5秒ずつ進め、3D骨格を人物IDにつなぎます。実測と補間の関節マスク、生成・消失・遮蔽・再出現、IDの曖昧さを保存します。上流v1が代替割当について保存するのはスコア差とマスクなので、時間窓では採用済み骨格の連結と分離を最適化します。

- **meiji_4cam** は全69クリップで各サイド2 IDを選別しました。020は651/651、058は765/765フレームで4人の実測骨格が揃いました。#2446向けのclip storeは `outputs/i2445/meiji4/` にあります。
- **Meiji 3cam開発用4本** の選別後は、射影IDF1 76.09%、switch 0、fragment 190、選別precision 99.94%、recall 61.43%でした。選別前はswitch 741で、`video_000/clip_000` の遠側は断片化により選別から外れました。
- **ローカル検証** はpytest 1,960件、pre-commit、knowledge検査が通過しました。 [評価結果と再現手順](https://github.com/Motoki0705/tennis-lab/blob/7b2c2dfd2345b1a1bd4e3473a75d6c5ff2bfdf44/knowledge/runs/run-i2445-multiview-tracking/README.md)

予約未見3本は共有GPUロックの権限エラーでViTPoseが揃わず、採点は0回です。#2444のViTPose完了後に、固定済みの設定で一度だけ採点できます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2469,"head":"7b2c2dfd2345b1a1bd4e3473a75d6c5ff2bfdf44","resume_on_success":false}}