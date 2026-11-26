# #2446: main を取り込み、#2470 を統合PRとしてマージ可能にする

親の方針（#1739 コメント https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7184189944 ）に従う。

## 背景
- main は分岐点から約435コミット進み、ECPose（#2499）・FastHMR（#2501・#2593）・単眼PLCS（#2536）の recipe、ボールイベント規則（#2569）などが入った。
- `git merge-tree origin/main origin/codex/parent-1739-issue-2446-primary` で、次の4ファイルが競合する: `knowledge/nodes/tennis_scene/.sequence`、`src/tennis_scene/pipeline/source_identity.py`、`src/utils/configuration/inventory.py`、`tests/integration/tennis_scene/test_fp32_cache_compatibility.py`。
- main 側の `definition.py`・`recipes.py`・`dataset_publication.py`・`body_artifacts.py`・`components/body_placement.py`・各 `*_recipe_cache_compatibility.json` も変わっている。テキスト上で競合しなくても、意味的にぶつかっていないかを確かめる。
- このbranchは #2444（PR #2449）・#2445（PR #2469）・#2459（PR #2461）をすべて含む。親は #2470 だけを正式レビューしてマージし、ほかの3PRは閉じる。

## やること
1. `origin/main` をこのbranchへ merge する（rebase はしない）。競合は、両方の意図を保つ形で解く。
   - `multiview_scene` recipe と、main の ECPose・FastHMR・単眼PLCS の recipe が、どれも明示的に選べる別経路として共存するようにする。既定経路は変えない。暗黙の fallback は作らない。
   - キャッシュ互換テストと fixture（code_sha256 など）は、値をただ再生成して合わせない。どのソース変更で hash が変わったのかを特定し、互換条件の意味が保たれることを確かめる。そのうえで、統合後のソース版に対する定義を更新する。
2. 関連テストを実行する。少なくとも `tests/integration/tennis_scene`、`tests/unit/tennis_scene`、multiview_pose・multiview_tracking・multiview_scene と、main 側で追加された ECPose・FastHMR・単眼PLCS のテスト。pre-commit と knowledge 検査（`kg_validate.py --check-summary --base-ref origin/main`）も通す。
3. push して、PR #2470 を更新する。
   - 本文の先頭を `Closes #2444` `Closes #2445` `Closes #2459` `Closes #2446` の4行にする。#2449・#2469・#2461 の内容をこのPRに統合したことと、統合したHEAD（各PRの現HEAD）を書く。
   - main を取り込んだこと、解いた競合と意味的な確認の要点を書く。
   - Draft を解除する。
4. 次のことはしない。
   - ベンチマーク（`data/tennis_multiview/processed/meiji_4cam/benchmarks/` 配下）と `i2363-all-v1/inputs/dataset` には触れない。`multiview_scene publish` の実行は、マージ後に親が別途指示する。
   - GPU は使わない。
   - #2493 の branch（`codex/parent-1739-issue-2446-evidence`）は変更しない。
5. 補足: #2605（#1722、未マージ）が SceneResult v3 と `event_detection` コンポーネントを `STANDARD_COMPONENTS` に加える予定で、`definition.py`・`schema.py` 付近で重なる可能性がある。先に main へ入った場合は、それも取り込んで解く。

## 完了の宣言
PR #2470 の新HEAD（完全SHA）と、解いた競合・意味的な確認・テスト結果の要点を報告し、CI待ちで終了する。
