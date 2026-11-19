5本ともmain `f627e01ac`をmerge commitで反映し、競合なし・最終HEADのCI成功を確認しました。PRのbaseは維持しています。

| PR | 新HEAD | 成功したCI |
|---|---|---|
| [#1892](https://github.com/Motoki0705/tennis-lab/pull/1892) | `f9b9645807b0` | [Python](https://github.com/Motoki0705/tennis-lab/actions/runs/44456957110) / [Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44457189570) |
| [#1896](https://github.com/Motoki0705/tennis-lab/pull/1896) | `58f8d0b3fa9a` | [Python](https://github.com/Motoki0705/tennis-lab/actions/runs/44456964551) / [Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44456964449) |
| [#1897](https://github.com/Motoki0705/tennis-lab/pull/1897) | `ae4b57e7cdbd` | [Python](https://github.com/Motoki0705/tennis-lab/actions/runs/44456970537) / [Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44456970534) |
| [#1950](https://github.com/Motoki0705/tennis-lab/pull/1950) | `df7bfe82f8cf` | [Python](https://github.com/Motoki0705/tennis-lab/actions/runs/44456976657) / [Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44457189389) |
| [#1955](https://github.com/Motoki0705/tennis-lab/pull/1955) | `9647408ef52c` | [Python](https://github.com/Motoki0705/tennis-lab/actions/runs/44458117962) / [Knowledge](https://github.com/Motoki0705/tennis-lab/actions/runs/44458118005) |

X3D/CSNは共通encoder・factoryを使い、state名と演算順を維持しました。mainの正規化・設定拒否テストと、既存の受容野・chunkテストを統合しています。

統合前後を別プロセスのCPU FP32で比較し、scene実checkpoint＋720p/T32実入力、DPT 10k、query dim512・6層、C1〜C3/T1/T2でstrict load・最大差0を確認しました。**T1/T2は公式CPU参照演算での検証です。**

ball_detection unit/integration、scene ball/X3D、MSDA関連は**1809成功・9 skip、408.14秒**。skipはCUDA専用6件と資産不足3件です。ruff・mypy・knowledge検証も成功しました。[検証記録・skip一覧](https://github.com/Motoki0705/tennis-lab/blob/9647408ef52ce4b519a6043a4b386be0be56dabf/knowledge/runs/run-ball-arch-desktop-20261113/integration-cpu.json)

C3未完了・途中結果をPRとknowledgeへ明記しました。固定tag `ball-arch-c3-exec-7d7a2e4fa`を保持し、実行commitが#1955の祖先であることを確認済みです。実行worktree・E:出力は変更していません。

親レビューでは、テスト統合、T1/T2のCPU参照限定、knowledgeノード95→96の再採番を確認してください。review・mergeへ引き継ぎます。