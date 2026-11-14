## 結果

- 123本・63,540frameをv3へ移行し、`meiji-active-ball-v3-20261113`へ公開。reader、全41clipのscene import、同一receiptへの再送を検証済みです。
- 対応表・保留一覧・再現手順を[実行記録](https://github.com/Motoki0705/tennis-lab/blob/b8e997748afb6ff48f07cd5300daa24fa81b6870/knowledge/runs/run-ball-v3-migration-20261113/README.md)に保存しました。
- 旧版の内容照合は成功。旧画像archive513本の比較基準は復旧時で、移設前SHAは未保存です。
- 新規28テストと変更対象pre-commitは成功。関連171件の再実行は155件成功でI/O中断。全体mypyには対象外292件のエラーが残ります。validator起動は0回です。

## branch・commit・PR・CI

`codex/ball-annotation-v3-migration`  
`b8e997748afb6ff48f07cd5300daa24fa81b6870`

[PR #1996](https://github.com/Motoki0705/tennis-lab/pull/1996)はレビュー可能です。`Closes #1714`を設定しました。最新HEADの[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/43574605400)・[Knowledge library](https://github.com/Motoki0705/tennis-lab/actions/runs/43574605341)とも成功しています。

## issueへの報告案

なし

## 親への判断依頼

- 保留48本の後続対応。対象相違39本、処理完了の根拠不足15本、重複6本です。
- 標準factoryが移設後のsymlinkを拒否する点の対応。今回はEの物理root指定で検証しました。merge後mainでの再読込は未確認です。