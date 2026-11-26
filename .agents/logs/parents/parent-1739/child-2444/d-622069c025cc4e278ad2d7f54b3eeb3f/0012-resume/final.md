FastHMRのテストは、#2444追加後の `source_identity.py` のハッシュをmainの値と直接比較していました。#2444の互換変換を適用したうえで、ECPose・FastHMRの完全一致条件と単眼PLCSの識別を検証する形に修正しました。

関連テスト2件は5.00秒で成功し、pre-commitとknowledge検査も通過しました。修正後のローカルHEADは `907e4d66b3f78a3db2d15a4da5733929ae5e60b1` です。全件を再実行中で、完了結果を確認してからpushとPR本文の更新へ進みます。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T073938Z_83daa37a1583_i2444-merge-cpu-r2"]}}