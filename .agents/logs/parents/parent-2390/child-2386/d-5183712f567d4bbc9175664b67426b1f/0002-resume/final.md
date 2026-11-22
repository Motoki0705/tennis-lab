PR #2407 を `main` の `631bee5aa` に載せ替え、本文を更新しました。新HEADは `a874310ea` です。差分は #2386 の1コミット・27ファイルで、マージ競合はありません。 [PR #2407](https://github.com/Motoki0705/tennis-lab/pull/2407)

容量保護の再試行と記録専用マージを含め、316件の通過を確認しました。並列実行で失敗した通知記録テスト1件は、同じHEADでの単独再実行で通過しました。文書リンク、pre-commitのruff・mypy、差分チェックも通過しています。

新HEADのCI起動を確認しました。 [CI実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45032696377)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2407,"head":"a874310ea03510a627f4ee81b281026813e92ddd","resume_on_success":false}}