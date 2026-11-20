---
title: "PRで submodule の pointer が進んでも、メインtreeの submodule の checkout は古いまま残る"
summary: "メインcheckoutの定期同期後もNHTのsubmoduleは古いcommitを指すことがある。checkoutのずれと作業中の変更を区別する。"
applies_to: all
source: "third_party/nht の M 表示の調査（2026-10-16）"
created: 2026-10-16
last_verified: 2026-11-20
evidence: "メインtreeで `git submodule status --recursive third_party/nht` を実行し、行頭が `+` なら記録された pointer と checkout がずれている"
citations:
  - path: ".agents/ops/git_sync/README.md"
    symbol: "#更新の契約"
  - path: ".spin/cmds.py"
    symbol: "NHT_ROOT_RELATIVE"
---

`deps(third-party): bump third_party/nht` のようなPRで固定commitが進んでも、メインcheckoutの定期更新はsubmoduleを同期しない。メインtreeの `third_party/nht`（と入れ子の `gsplat`）は古いcommitのまま残り、`git status` に ` M third_party/nht` が出ることがある。2026-10-16には手動pull後も2回分のbumpが反映されていなかった。

`nht-render` と NHT trainer は、メインtreeの `third_party/nht` から editable install されている。そのため、このずれは「mainにあるはずのNHTの修正が実行時に使われない」という静かな食い違いになる。

submodule 内の未追跡ファイルにも、scene の `resolved-config.yaml` から参照されているものがある（例: `nht.pipeline_config_path` が指す派生config）。消すと、その scene の再実行が fail-closed で止まる。

**使い方:** submodule の `M` を見たら、`git submodule status --recursive` で `+`（pointer とのずれ）を確認し、submodule内の変更有無も調べる。ずれだけなら、同期の要否を判断する材料として報告する。未追跡ファイルを整理する前に、`rg -l <ファイル名> data/synthetic_data_generation/scenes -g resolved-config.yaml` で設定からの参照を確かめる。
