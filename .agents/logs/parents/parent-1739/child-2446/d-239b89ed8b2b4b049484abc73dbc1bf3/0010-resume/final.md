[#2470](https://github.com/Motoki0705/tennis-lab/pull/2470) に `main` の `826c0de759f17e388cac778e13aea978a73470be` を取り込み、pushとPR本文の更新を完了しました。新HEADは `0411c84b060162448c2f7587ee3d73cca8e18d5c` です。knowledgeの重複・採番、設定登録、キャッシュ互換、比較ページと回帰テストの競合を解消しました。

- 指摘4は、比較ページと完了表に `--legacy-generation before-multiview` を追加しました。一時データセットで新経路を採用した後も、両CLIが保存された同じ旧sceneを読めることを確認しました。
- 指摘5は、保存済みバッチから身体復元に必要な設定を読み取るようにしました。イベント設定のない #2363 形式で `prepare` と `run` が3D入力の読込みまで到達しました。出力は SceneResult v2 です。v3化にはイベント検出段の追加と、公開・読戻しの検証が必要です。

CPUテストは1,875件が成功しました。共有Python 3.11.15で `os.pidfd_open` がなく失敗した既存の注釈テスト1件も、Python 3.11.17の独立環境で再実行し成功しました。pre-commit、knowledge検査757ノード、`main` とのキャッシュ識別照合16条件も通過しました。

指摘1〜3は次の継続指示の対象です。新HEADの [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45835131237) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"0411c84b060162448c2f7587ee3d73cca8e18d5c","resume_on_success":false}}