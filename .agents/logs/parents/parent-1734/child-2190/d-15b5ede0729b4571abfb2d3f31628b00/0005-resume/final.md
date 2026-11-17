承認記録と分析・改善計画を[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)へ反映しました。

- branch: `codex/court-acceptance-postprocess`
- commit: `f3eb0a469`
- 今回の実装変更・GPU使用・queue追加・新しいモデルraw取得はありません。

主な結果です。

- 参照の承認flagを更新。H・座標・評価値は不変です。
- Court valの114件喪失は、支持点選択の非収束が原因。旧選択に戻すCPU対照では114件とも回復しました。
- 正解候補がある184画像の未救済は、gate拒否106件・順位負け12件でした。
- 中断前のCPU RGB試行では、32画像の正しい候補が9→29件。ただし3件の誤りと評価の依存性が残ります。

**推奨は、支持点選択の回帰修復を先に行い、独立評価を組み込んだRGB精密化へ進む案です。** 詳細・画像・テスト設計・GPU再開後のjob計画は[knowledge 72](https://github.com/Motoki0705/tennis-lab/blob/f3eb0a469c62e19199f7b5b341b5a9e0e08f0398/knowledge/nodes/court_detection/000072-run-court-approved-error-analysis-20261116.md)へ記録しました。

syntheticの12件は、元runの中間距離場・solver履歴が未保存のため厳密な帰属が未確定です。親には実装方針と独立評価用注釈の準備を判断事項として返します。GPU再開は別途指示が必要です。

分析テスト3件、knowledge検証、pre-commit、対象HEADの[CPU CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43890052668)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43890052653)は成功。受入目標は未達のまま、merge・issue closeはしていません。