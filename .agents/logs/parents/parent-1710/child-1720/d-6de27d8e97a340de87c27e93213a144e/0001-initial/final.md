[PR #1800](https://github.com/Motoki0705/tennis-lab/pull/1800) を作成・push済みです。**実装・検証は完了し、上流更新に伴うrebase待ちです。**

- branch: `codex/explain-html-ui`
- HEAD: `4f96bfac410e665762119483478b367de6525253`
- [最終CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42590119772): attempt 2でPython・Lint・CPU全10本が成功。既存注釈テスト2件の失敗は、失敗ジョブだけの再実行で成功しました。
- ローカル回帰194件成功、読み位置の強化検証25.93秒、契約・HTTP検証7件成功。pre-commit・ruff・mypyも成功です。

実service＋偽editorの4シナリオで確認しました。

| 受入項目 | 検証・実測 |
|---|---|
| 全体／選択指示、処理中の閲覧 | reading：6.98秒。旧版保持、本文への回答追加、UTF-16選択 |
| 移動・履歴・undo・復旧 | recovery：13.80秒、removed：1.71秒。重複生成防止、cancel/retry、競合、削除時の案内 |
| 隔離・狭幅・キーボード | security：1.80秒。不正message/script拒否、390px幅、Enter/Tab操作 |

iframeは `sandbox="allow-scripts"` のopaque originです。本文→UIはready・selection・閲覧stateだけをwindow・origin・文書版・nonceで照合します。tokenはUIメモリだけに保持し、本文から指示を送信できません。

Windows Edge 157でlocalhost／127.0.0.1両方の代表操作が成功しました。console error・試験用Edge残存は0です。[変更後の画面](https://github.com/Motoki0705/tennis-lab/blob/4f96bfac410e665762119483478b367de6525253/outputs/explain-html/windows-edge/02-change-navigation.png)、[結果JSON](https://github.com/Motoki0705/tennis-lab/blob/4f96bfac410e665762119483478b367de6525253/outputs/explain-html/windows-edge/result.json)、[7.32秒の動画](https://github.com/Motoki0705/tennis-lab/blob/4f96bfac410e665762119483478b367de6525253/outputs/explain-html/chromium/reading-workflow.webm)を保存しました。

選択通知・状態復元のため、serviceの閲覧view分岐10行と共通ファイル名列挙への追加1行を実装しました。JSON要求・応答、store・agy・visualsは維持しています。

上流更新でPRは現在競合しています。自身の変更は元base `8817935` から3コミット・16ファイルですが、現在のPR比較は33ファイル・10コミットです。指示どおりrebaseは未実施です。親による載せ替え・統合レビューと、#1721の実Gemini／実visual validator／managed renderer統合確認が残ります。