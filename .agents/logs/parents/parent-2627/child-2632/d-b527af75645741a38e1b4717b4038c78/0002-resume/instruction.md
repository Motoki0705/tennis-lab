PR #2640 の正式レビューで major 1件。採用する。

指摘: 共有の `.git/info/exclude` の `[local path]` により、`paper/tennis_scene_architecture/` の新規ファイル（figures・tables・evidence など）が通常の `git add` から漏れる。#2632 で指定した `git add -f` の扱いが README と制作スキルに書かれていない。

修正: `paper/tennis_scene_architecture/README.md` と `.agents/skills/paper-writing/SKILL.md`（記述は一方を正本にし、他方はリンク）に、追跡する新規ファイルをパス指定で `git add -f` すること、ステージ済み一覧（`git diff --cached --name-only`）で必要な成果物が揃い `build/` の中間生成物が含まれないことを確認する手順を書く。可能なら、追跡対象の図表・表・evidenceの漏れを検出する簡単な確認（例: TeXが参照するファイルがすべて `git ls-files` にあるかを確かめるスクリプトをビルド手順に組み込む）を加え、未追跡の参照ファイルで失敗することを確かめる。

同じPR・branchで修正してpushし、新HEADのCIを `PARENT_RESULT` の wait で宣言して終了する。
