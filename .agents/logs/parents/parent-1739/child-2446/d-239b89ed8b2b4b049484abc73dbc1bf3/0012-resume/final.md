[#2470](https://github.com/Motoki0705/tennis-lab/pull/2470) に #2462 をマージしてpushし、PR本文にレビュー指摘1〜5の対応を記録しました。新HEADは `f5566a4be572a8dc80a3421ddd306d2a6f8f82e9` です。

- `scene_index.py` とFP32回帰テストの競合を解消しました。knowledgeは外部ボールrunのIDと採番105を保持しました。
- #2450 → #2462 → #2470 の統合順で、既存recipeのキャッシュ識別が全16条件で一致し、対象20ファイルの各編集で互換条件が失効することを確認しました。
- 共用の切替処理で、片方のポインタだけが更新された状態では4つの読込入口が復旧手順を示し、公開の再実行と復元で整合した世代に戻ることを確認しました。

CPUテストは1,938件成功、警告1件でした。pre-commitとknowledge検査758ノードも通過しました。前回誤記したジョブIDは `20261126T145128Z_d85353459e97_i2446-review-r1-123-check` に訂正し、PRに記録しました。新HEADの [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45849564964) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"f5566a4be572a8dc80a3421ddd306d2a6f8f82e9","resume_on_success":false}}