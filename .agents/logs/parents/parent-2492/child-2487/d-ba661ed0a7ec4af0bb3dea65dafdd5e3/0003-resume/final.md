[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に `recipe=ecpose` と候補3系列のadapterを統合しました。共通recipe実装を再利用するため、draftのbaseを #2501 に変更しました。ブランチは `codex/parent-2492-issue-2487-primary`、最新commitは `7a5564570` です。[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45403468174)

ECPose-Lの公式重みで `clip_003/cam0` の連続3フレームをCPU推論し、各フレームで2人分のBBoxと対応するCOCO17を確認しました。既存のv2形式での保存・再読込、設定が異なる成果の拒否、既定経路のFP32 cache再利用をローカルテストで確認しました。pre-commitとknowledge検査も通過しました。

DINO系とRF-DETR-Lの計5条件を、最新版の採点器で各6,844画像について再採点し、全指標と層別集計が既報と一致しました。RF-DETR-2XLの3条件は、ユーザーのライセンス判断に従い対象外として記録済みです。

#2496の固定速度入力と速度・VRAM基準、残りのGPU精度、Pareto frontierと横並び可視化、#1739の新3D経路での品質比較が残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"7a55645709ce8e551e3222d6771bff950089f342","resume_on_success":true,"next_step":"#2496の固定速度入力と現行速度基準、補助担当のGPU成果を照合して比較表・Pareto frontier・代表clip可視化を更新する。未確認のECPose実機backendと#1739新経路の検証条件を確認する。"}}