[Draft PR #2222](https://github.com/Motoki0705/tennis-lab/pull/2222) に実装・検証結果を保存しました。

- branch: `codex/court-video-shared`
- HEAD: `5ec8e30e3`
- 最終HEADのPython・knowledge CI、pre-commit成功。
- 既定clipの3,545画像・8,530領域は旧処理と完全一致。

**共通Hは4cameraとも参照精度目標を達成しましたが、受入は未達です。** drift検査不能が169/276件あり、4cameraそろって使用できるclipは0/69です。閾値は維持しています。

[全件画像・原寸crop](https://github.com/Motoki0705/tennis-lab/blob/5ec8e30e3119304227f43b1854cb493ec4cd5e0d/outputs/tennis_scene/evaluate/court-video-shared-acceptance-20261117/review/index.html)と[結果・原因分析](https://github.com/Motoki0705/tennis-lab/blob/5ec8e30e3119304227f43b1854cb493ec4cd5e0d/knowledge/nodes/court_detection/000077-run-court-video-shared-acceptance-20261117.md)を残しました。人間の確認と#1752の一括CLI接続は未実施です。

検査不能多発時は親へ返す指示に従い、**共有した線対応とclip内の複数frameを使うdrift診断の再設計**への継続判断をお願いします。merge・closeは行っていません。validatorは未指定・実施0回です。