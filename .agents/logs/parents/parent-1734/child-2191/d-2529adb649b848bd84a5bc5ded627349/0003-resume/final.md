再設計・再評価を [Draft PR #2222](https://github.com/Motoki0705/tennis-lab/pull/2222) に反映しました。**受入は未達です。**

- branch: `codex/court-video-shared`
- HEAD: `3f03ea70e`
- [CI・Python判定](https://github.com/Motoki0705/tennis-lab/actions/runs/44061474772)、Knowledge CI、pre-commit成功。追加22ケース。validatorは未指定・0回。
- 共通Hと内容hashは不変。独立人工評価は24/24一致、検査不能・移動の安全誤判定とも0。

|camera|使用可（warning含む）|warning|失敗・検査不能|
|---|---:|---:|---:|
|cam0|21|0|48|
|cam1|61|4|8|
|cam2|39|0|30|
|cam3|0|0|69|

**4camera成立は0/69clip。** 失敗は観測線と共通変位の矛盾です。事後分析でも53frameは、実際の動きと測定限界を分離できません。閾値変更や失敗の採用への変更はしていません。

[run 78の詳細](https://github.com/Motoki0705/tennis-lab/blob/3f03ea70e91c5b0f7e07c58fbce53bc3da1a7bda/knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md) と [全件レビューHTML](https://github.com/Motoki0705/tennis-lab/blob/3f03ea70e91c5b0f7e07c58fbce53bc3da1a7bda/outputs/tennis_scene/evaluate/court-video-drift-lines-20261117/review/index.html) に、1,656枚のoverlay・6,624組の原寸crop・frame別診断を残しました。

次案は線候補の保持・対応改善です。追加実装の判断と人間の品質確認を親へ返します。#1752接続・merge・closeは未実施です。