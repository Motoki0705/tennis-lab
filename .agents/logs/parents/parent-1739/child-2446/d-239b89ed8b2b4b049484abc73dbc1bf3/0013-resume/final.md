[#2470](https://github.com/Motoki0705/tennis-lab/pull/2470) のCIで失敗した `storage_tier` テストを修正し、pushしました。新HEADは `06ac66ab58ef771c2ec30e422ba02e4bc24e00bc` です。

子プロセスの2回目の `open` が一時パスの削除と競合していました。両方の `open` に同じinodeを指すハードリンクを使い、リースによる拒否、監査記録、解除後の書込みを確認するようにしました。

関連テスト11件が2 workerで成功し、pre-commitも通過しました。PR本文に結果を記録し、新HEADのCI待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"06ac66ab58ef771c2ec30e422ba02e4bc24e00bc","resume_on_success":false}}