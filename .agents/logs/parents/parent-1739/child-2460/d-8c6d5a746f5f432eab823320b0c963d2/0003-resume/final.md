[PR #2471](https://github.com/Motoki0705/tennis-lab/pull/2471) のFP32キャッシュ互換テスト2件を修正しました。#2458 のCLI監査登録後の `inventory.py` のSHA-256を互換manifestに固定し、未知のハッシュを拒否する検査を追加しました。

関連pytest 32件とpre-commitが通過しました。新HEADでパイロット2件を読み戻し、記録済みのscene ID・有効フレーム数と、共有dataset全69件の #2458 版参照が一致しました。[依存PR #2462](https://github.com/Motoki0705/tennis-lab/pull/2462) でも同じ2件の失敗を確認しました。

PR本文の実行証拠を更新し、HEAD `1396f63feb57bece36e4acb73e351fbcf12aff9f` をpushしました。CI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2471,"head":"1396f63feb57bece36e4acb73e351fbcf12aff9f","resume_on_success":false}}