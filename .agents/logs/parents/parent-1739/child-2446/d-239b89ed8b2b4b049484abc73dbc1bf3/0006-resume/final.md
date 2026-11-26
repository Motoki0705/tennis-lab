`main` の `5e088c3d4ebfeb58bac8e66fff2a18b0067f01ee` を取り込み、ナレッジ採番、ソース識別、設定登録、FP32回帰テストの4ファイルの競合を解消しました。ナレッジ6件はrun IDを維持して98〜103に採番しました。

統合後のソース識別は、キャッシュの16条件すべてで取り込み元の `main` と一致しました。履歴のFP32 fixtureは固定し、対象ファイルを編集した場合は識別が変わることも確認しました。ECPose・FastHMRの進捗記録を新経路のGVHMRに渡すと明示的に停止します。

重点テスト37件とknowledge検査は成功しました。全対象のCPUテストとpre-commitをローカルジョブへ渡しました。結果を受けてコミットとpush、本文更新、Draft解除を行います。 [PR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T062258Z_6948dbcef7e6_i2446-main-merge-check"]}}