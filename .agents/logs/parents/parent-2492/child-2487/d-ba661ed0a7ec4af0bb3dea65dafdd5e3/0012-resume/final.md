[PR #2499](https://github.com/Motoki0705/tennis-lab/pull/2499) の正式レビュー指摘3件を修正し、`codex/parent-2492-issue-2487-primary` に `c5664bc288a2c8f79a0a34ab950cdd954c3e4ecc` をpushしました。PR本文の `Closes #2575` を保持し、実行証拠を更新しました。

- `c1410f014` — YAMLの作業状態をロードごとに分離しました。`test_load_network_isolates_release_yaml_across_loads` は2件通過し、公式重みでも同一CPUプロセスでL→X→LとX→L→Xの厳密ロードに成功しました。
- `54dc31c00` — 公開と完了マーカー再利用の両方で、BBoxとCOCO17の出所および実行設定を照合します。`test_dataset_publication_binds_ecpose_to_the_scene_backend_and_release` が通過し、別backend・別モデル・別余白・別producerの拒否を確認しました。
- `c5664bc28` — 比較表44行とPareto図2枚を再生成し、保存済みPT32比較10条件を追記しました。RF-DETR-L TRT16は旧基準で合格、共通基準ではIoU P05が0.9620で不成立です。現行計時の「PT32未計測」とPareto点を保持しました。`test_rfdetr_saved_pt32_comparison_preserves_measured_frontier_point` は2件通過しました。

標準recipeの公開・再利用を含む関連102件、比較68件、境界検査18件が通過しました。pre-commitとknowledge検査も通過し、18条件の計測値と両図のPareto点はレビュー時の結果と一致しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"c5664bc288a2c8f79a0a34ab950cdd954c3e4ecc","resume_on_success":false}}