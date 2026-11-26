掲示板E2Eのテスト用Herdrで、共有JSONの読み書きをファイルロックで保護しました。書き込み途中に読み込むと旧版では `JSONDecodeError` が発生し、修正後は両方の呼び出しが成功して記録も2件残ることを確認しました。

CIと同じPlaywright・Chromiumで掲示板・Herdrの既存8件が成功しました。Clip Studioのpytest 254件、Node回帰10件、pre-commitも通過しました。

`eaeb3a41a` を `codex/parent-2601-issue-2601-primary` にpushし、PR本文に検証結果を追記しました。新HEADのCIは実行中です。[PR #2612](https://github.com/Motoki0705/tennis-lab/pull/2612)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2612,"head":"eaeb3a41a0deb93f4efae5cc5a046891bfc96083","resume_on_success":false}}