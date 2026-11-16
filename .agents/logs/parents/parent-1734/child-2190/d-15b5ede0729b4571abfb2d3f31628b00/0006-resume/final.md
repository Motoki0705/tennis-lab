**受入目標は未達です。** [PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)を更新し、ready for reviewにしました。

`codex/court-acceptance-postprocess` / `fa7bb8ebe`

- RGB込みは誤採用0件ですが、正しい推定の採用率は **5/221＝2.3%**。cam1〜3は採用0件です。
- Court valの114件喪失は回復しました。一方、syntheticでは(a)が58件喪失・20件獲得、RGB込みは採用0/886で、非劣化も未達です。
- 関連テスト253件成功、CUDA 1件skip。最終HEADの[通常CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43928345589)・Knowledge CIとも成功。独立validatorは指定なし、0回です。

詳細は[knowledge 73](https://github.com/Motoki0705/tennis-lab/blob/fa7bb8ebe/knowledge/nodes/court_detection/000073-run-court-fixed-support-rgb-20261117.md)、画像は[全69clipの一覧]([local path])にあります。閾値は変更していません。支持集合候補と初期塗線対応の再設計方針を親へ返します。merge・closeは未実施です。

新CUDA確認は共有lockの権限で起動前に失敗しました。[training-queue skill](https://github.com/Motoki0705/tennis-lab/blob/fa7bb8ebe/.agents/skills/training-queue/SKILL.md#L245)の「never repaired or bypassed」に従い迂回せず、復旧後の確認を残しています。