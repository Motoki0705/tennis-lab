ダブルスの枠を確定する前に断片の連結可能性を調べるよう修正しました。40フレームの回帰テストでは、2選手が各1枠に入り、片方が不在ならその側の欠員が1になることを確認しました。追加2ケースは修正前に失敗し、修正後は関連pytest 41件とpre-commitが通過しました。

修正コミットは `d31076eb929d51b8980780ff73bcb4969fbfaaac` です。開発4本と4cam 69本のCPU回帰を登録しました。結果を照合した後、[PR #2469](https://github.com/Motoki0705/tennis-lab/pull/2469) に反映します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T114610Z_7989777621e9_i2445-slot-review"]}}