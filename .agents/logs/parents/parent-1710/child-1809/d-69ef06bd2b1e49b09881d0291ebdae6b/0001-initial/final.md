[PR #1811](https://github.com/Motoki0705/tennis-lab/pull/1811) をreadyにしました。

- branch: `codex/explain-html-generation-robustness`
- HEAD: `760a0b466`
- 関連102テスト、Chromium検証、pre-commit、最終HEADの全CI成功。追加テストは上限内の12ケースです。
- 指定Gemini/highを直列8呼出しで検証し、3題材の初版＋三角測量の全体再構成が、すべて公開・静的exportに成功しました。生成前停止2試行は別記しています。

時間・実効設定・成果物は[実機記録](https://github.com/Motoki0705/tennis-lab/blob/760a0b46617232acddc7f79b6f14d3edfdc827b5/outputs/explain-html/robustness/execution-summary.md)に保存しました。

残りは親の統合レビュー・mergeです。事実性の改善は#1810の範囲です。