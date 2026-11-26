[#2470](https://github.com/Motoki0705/tennis-lab/pull/2470) の作業ツリーに `main` の `a0b327a74f3335489ae9c34a7766e2d250135716` を取り込みました。knowledge の重複と採番、設定登録、キャッシュ互換層、比較ページと回帰テストの競合を解消しました。標準経路のソース識別は全16条件で `main` と一致しました。

- 指摘4は、比較ページと完了表の両方に `--legacy-generation before-multiview` を追加しました。一時データセットで採用切替を行い、両方が公開記録に保存された旧sceneを読めることを確認しました。
- 指摘5は、保存済みバッチから新経路に必要な設定だけを読む契約に変更しました。#2363形式の fixture で `prepare` と `run` が3D入力の読込段階まで到達しました。追加3ケースは8.40秒で成功しました。出力は SceneResult v2 です。v3化には、イベント検出の明示設定と実行段の追加、公開・読戻し時の依存関係の検証が必要です。

関連CPUテスト、pre-commit、knowledge検査をローカルジョブに登録しました。結果確認後にコミット、push、PR本文の更新へ進みます。現在のPR HEADは `b62a0f3ddca17145394f87992a3b03012051466f` です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T134357Z_67fb419f89fa_i2446-review-r1-45-check"]}}