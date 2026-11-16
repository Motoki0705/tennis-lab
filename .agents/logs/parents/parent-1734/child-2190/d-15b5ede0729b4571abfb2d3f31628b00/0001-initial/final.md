planningを完了しました。実装・閾値は変更していません。[詳細計画](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/knowledge/nodes/court_detection/000069-run-court-acceptance-diagnosis-20261116.md)と[#2190の報告](https://github.com/Motoki0705/tennis-lab/issues/2190#issuecomment-6909676404)に記録しました。

1. **診断**：不採用124件はinlier不足のみ79、画像外のみ15、両方30。全276件を目視し、不採用の61件は概ね整合、13件は明瞭なずれ、50件は保留と仮分類しました。採用152件は144件が概ね整合、cam3の8件は保留です。参照未確認なので正誤の確定値ではありません。

2. **原因**：全14点の画像内投影を要求するgateが部分観測を拒否します。一方、cam3の代表例では端に集まった点は最終fitから既に除外されていました。少数の点とcrop内LINEからの不安定な外挿も改善対象です。[代表画像](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/representative_examples.jpg)

3. **参照**：各cameraの手指定点から草案を作り、冒頭・中盤・終盤の12画像に重ねました。全件「未確認」で、評価・調整には未使用です。確認用画像：[cam0](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/reference_draft/cam0_review.jpg)、[cam1](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/reference_draft/cam1_review.jpg)、[cam2](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/reference_draft/cam2_review.jpg)、[cam3](https://github.com/Motoki0705/tennis-lab/blob/b46b97869d18b259db8bb092c513f2da91ba6494/outputs/tennis_scene/evaluate/court-acceptance-planning-20261116/reference_draft/cam3_review.jpg)。

4. **改善案**：可視線の一致、独立した支持点・線の分布、Hの悪条件性、外挿の安定性で採否を決めます。境界・重複点の扱い、候補順位、線ごとの診断も見直します。数値の緩和だけでは進めません。

5. **評価案**：参照確認後、旧処理／採否だけ変更／後処理も変更の3条件で、camera別混同行列・px誤差・clip間ばらつきを比較します。meiji_3cam、tennis_clip、Court val/testでも回帰確認します。Court評価資産の復元とsplit来歴確認が残っています。

6. **検証**：現行10crop＋full-frame 1件をqueue経由で記録し、10cropのstatus・inlier数は旧summaryと一致。既存58テスト成功、knowledge検証はエラー・警告0。pre-commitは対象外でskip、validatorは未指定につき0回です。

branch：`codex/court-acceptance-postprocess`、HEAD：`b46b97869`。調査記録のみコミット済みで、push・PR・CIは未実施です。

親へ返す判断点は、**参照草案の確認・修正、正誤と回帰のpx許容・目標率、実装方針**です。今回のphase指示に従い、実装開始の具体指示を待ちます。