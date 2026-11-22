[PR #1787](https://github.com/Motoki0705/tennis-lab/pull/1787) をmainベースへ載せ替え、force-pushしました。

- HEAD: `a4b617aeca5e773c869a6f696de8cd8a7406139e`
- 本件の2コミット・15ファイルのみ。README 3言語版の競合を解消し、契約3ファイルはmainと一致しています。
- ローカル **170件成功[local path]。pre-commit・追加mypyも成功。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42563944096) は **Python・Lint・CPU全10ジョブ成功**。

PR本文に「下流への影響」を追加しました。編集実測271.07秒に対し、生成全体の既定timeout 600秒には328.93秒の余裕があるため維持します。非同期の処理中表示・旧版保持・取消と、描画／保存時間の別途考慮を明記しました。

追加の実機呼出しはありません。worktreeはclean、PRは競合なしです。残りは親の統合レビュー・mergeです。