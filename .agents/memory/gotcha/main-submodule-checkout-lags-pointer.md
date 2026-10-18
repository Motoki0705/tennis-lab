---
title: "PRで submodule の pointer が進んでも、メインtreeの submodule の checkout は古いまま残る"
applies_to: all
source: "third_party/nht の M 表示の調査（2026-10-16）"
created: 2026-10-16
last_verified: 2026-10-16
evidence: "メインtreeで `git submodule status --recursive third_party/nht` を実行し、行頭が `+` なら記録された pointer と checkout がずれている"
---

`deps(third-party): bump third_party/nht` のようなPRがmergeされても、メインtreeで `git pull` するだけでは submodule は更新されない。メインtreeの `third_party/nht`（と入れ子の `gsplat`）は古いcommitのまま残り、`git status` に ` M third_party/nht` が出続ける。2026-10-16 には2回分のbumpが反映されていなかった。

`nht-render` と NHT trainer は、メインtreeの `third_party/nht` から editable install されている。そのため、このずれは「mainにあるはずのNHTの修正が実行時に使われない」という静かな食い違いになる。

submodule 内の未追跡ファイルにも、scene の `resolved-config.yaml` から参照されているものがある（例: `nht.pipeline_config_path` が指す派生config）。消すと、その scene の再実行が fail-closed で止まる。

**使い方:** submodule の `M` を見たら、まず `git submodule status --recursive` で `+`（pointer とのずれ）か、未追跡・変更ファイルかを分ける。ずれだけなら、メインtreeで `git submodule update --checkout --recursive third_party/nht` を実行する。未追跡ファイルは、`grep -rl <ファイル名> data/synthetic_data_generation/scenes/*/resolved-config.yaml` で参照を確かめるまで消さない。必要なものは submodule の `.git/modules/.../info/exclude` に理由付きで書いて残す。
