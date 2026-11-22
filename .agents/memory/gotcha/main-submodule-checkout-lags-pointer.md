---
title: NHTのsubmodule checkoutが古いと学習・描画で古いコードを使うことがある
summary: NHTのgitlinkと実checkoutのずれは実行版に影響する。submodule内の未追跡設定を整理する前にsceneの参照を確かめる。
applies_to: all
source: third_party/nht の M 表示の調査（2026-10-16）
created: 2026-10-16
last_verified: 2026-11-20
evidence: git submodule status --recursive third_party/nht の行頭+はgitlinkとのずれを表す。setup-nhtはpublic toolをeditableで導入し、専用trainerをsubmodule内に作る。NHT設定の読込は参照ファイルの存在を要求する。
citations:
- path: .agents/ops/git_sync/README.md
  symbol: '#更新の契約'
- path: src/synthetic_data_generation/README.md
  symbol: '#set-up-nht-with-spin'
- path: src/synthetic_data_generation/configuration.py
  symbol: def _nht_pipeline_config
---

2026-10-16はメインcheckoutのpull後も `third_party/nht` の実checkoutに2回分のgitlink更新が反映されていなかった。公開コマンドのeditable installや専用trainerがこのsubmoduleを使っていれば、メインのコードを更新してもNHTの修正は実行されない。`git submodule status --recursive third_party/nht` の行頭 `+` を見て、gitlinkのずれとsubmodule内の実変更を区別する。

submodule内の未追跡ファイルが `resolved-config.yaml` の `nht.pipeline_config_path` から参照されている場合、そのファイルを消すとsceneの再実行が止まる。整理前に `data/synthetic_data_generation/scenes` の保存設定から該当ファイルの参照を調べる。
