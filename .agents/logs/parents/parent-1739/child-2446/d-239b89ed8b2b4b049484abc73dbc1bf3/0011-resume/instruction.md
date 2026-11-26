# PR #2470: #2462 の取り込みと、正式レビュー round 1 の指摘1〜3への対応（その2）

前のターンの HEAD は `0411c84b`（main `826c0de7` を取り込み済み、指摘4・5は対処済み）。マージ順は #2450 → #2462 → #2470。#2462 の新HEADは `497195ff38ac8bbe49baefac64cffddd64515ba7`（main 取り込み済み、CI成功）。共通の一貫した切替の部品を持ち、knowledge は tennis_scene `000105`（`.sequence` 105）を使う。

1. `origin/codex/parent-1739-issue-2458-primary`（`497195ff`）をこのbranchへ merge する（rebase はしない）。`origin/main` が進んでいれば、それも取り込む。
2. 指摘1（knowledge 採番の衝突）: #2462 取り込み後の連番に合わせる。このPRでノードを足すなら 106 以降にする（`.sequence` も合わせる）。run ID は維持する。`kg_validate.py --check-summary --base-ref origin/main` を通す。
3. 指摘2（指定順の統合でキャッシュ互換条件が成立しない）: #2462 の外部ボールCLIの登録を保持したまま `src/utils/configuration/inventory.py` を統合する。統合後の実ファイルに合わせて、#2470 の互換表（マニフェストの照合値）を結び直す。次の2点を、#2462 を含むmain相当の実ソースで確かめる。
   - 完全一致時に、既存 recipe のキャッシュ識別が16条件で一致する。
   - 対象ファイルを編集すると失効する。
   `test_multiview_integration_preserves_main_source_identity` は、指定順で合成した状態で通るようにする。値をただ再生成して合わせない。
4. 指摘3（参照を片方だけ切り替えた状態が下流に見える）: `multiview_scene publish/restore` を、#2462 の共通の一貫切替の部品に乗り換える。独自の `_set_pointers` などによる、ファイルごとの切替は削除する。テストで次のことを確かめる。
   - 2つのポインタの更新の間で止まった状態を作る。
   - 通常の読込入口（SLCS の読込・`scene_index` の参照解決・比較CLI・完了表）が、一貫した世代を読むか、明示的なエラーと復旧手順を出す。
   - `publish` の再実行と `restore` で回復する。
   旧参照の記録（指摘4の `--legacy-generation before-multiview` が読む先）が、この部品と矛盾しないことも確かめる。
5. 関連テスト、pre-commit、knowledge 検査を通して push し、PR本文を更新する。本文に、指摘1〜5の対処の要点を書く。
6. `i2363-all-v1/inputs/dataset` とベンチマークには書き込まない。`outputs/i2446-ballrev1` を消さない。GPU は使わない。

新HEAD（完全SHA）、解いた競合、指摘1〜3の対処、テスト結果を報告し、CI待ちで終了する。
