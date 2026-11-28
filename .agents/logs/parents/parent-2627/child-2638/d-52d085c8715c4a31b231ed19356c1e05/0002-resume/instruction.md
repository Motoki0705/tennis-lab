PR #2720 の正式レビュー（round 1 / `parent-2627-069237179bbe4e5d884fe678b7a96ca6`）で以下の **major 指摘 3件** が入りました。3件すべてを採用し、実装・正本コードと突き合わせた上で修正してください。

## 修正対象（全3件）

### 1. 第4章の身体配置式に選手ごとのスケールが抜けている
- 箇所: `paper/tennis_scene_architecture/chapters/reconstruction.tex`（45行目付近）
- 確認元: `src/tennis_scene/motion_alignment/temporal.py:150`・`:299`、`src/tennis_scene/motion_alignment/mesh_placement.py:113`
- 指摘内容: 現行式は局所関節に並進と yaw だけを適用し、スケールを点検項目として書いている。実装では選手ごとに対応する骨長比の中央値（スケール $s$ 等）を求め、そのスケールを局所関節に掛けてから位置と yaw 補正を最適化し、保存する SMPL root も推定した腰中心にスケール適用済みのオフセットを加えた値になっている。
- 修正方針: 実装コードを確認し、残差式・変数定義・SMPL root 配置の説明を実装と完全に一致させること。必要なら `evidence/claims.md` / `evidence/sources.tsv` に参照ソースを追加する。

### 2. 第6章が AFLink による断片連結にも pose を使うと誤記している
- 箇所: `paper/tennis_scene_architecture/chapters/player_tracking.tex`（7行目付近）
- 確認元: `knowledge/nodes/person_tracking/000012-run-i964-tracker-hybrids-r10-20260930.md:42`、`src/tasks/person_tracking/strongsort.py:185`、`src/tasks/person_tracking/strongsort_offline.py:100`
- 指摘内容: `000012` の比較で pose コストを加えたのは「オンラインの外観・運動照合および残余 IoU 照合」であり、オフラインの AFLink（断片連結）には box と実観測マスクが渡され、フレーム番号と box 位置で連結判定を行う。
- 修正方針: raw IDF1 が `0.930 -> 0.955` に改善した比較の介入範囲が正確に伝わるよう、オンライン対応（外観・運動・残余 IoU）に pose を加える一方、オフライン AFLink による断片連結は box と時刻・実観測マスクを使うことを明確に書き分けること。

### 3. 付録に SceneResult のマスク・理由コード一覧と図表の再生成入口が欠けている
- 箇所: `paper/tennis_scene_architecture/appendix/reproducibility.tex`（27行目以降）
- 確認元: `paper/tennis_scene_architecture/design.md:74`、`src/tennis_scene/schema.py`、`src/tennis_scene/README.md`、`paper/tennis_scene_architecture/scripts/`
- 指摘内容: `design.md:74` は SceneResult v2 のマスク・理由コード一覧と図表の再生成入口を付録に収録すると定めているが、現行付録から保存マスクの識別子、非ゼロ理由コードの意味、および表・図の再生成・照合コマンドを参照できない。
- 修正方針:
  1. `src/tennis_scene/schema.py` と `src/tennis_scene/README.md` の実装正本から、SceneResult v2 の保存マスク識別子一覧および各理由コード配列の非ゼロ値の意味を表で収録すること（数値・コード値・名称はすべて実装から確認し、`evidence/claims.md` / `sources.tsv` と整合させる）。
  2. 図表・根拠の再生成・照合入口（`scripts/make_tables.py`, `scripts/prepare_figures.py`, `scripts/make_diagrams.sh`, `scripts/check_evidence.py`, `build.sh --check-tracked`、および元図の再現手順の正本 `evidence/figures.md`・`evidence/claims.md`）を付録へ明記すること。

## 検証・提出手順
1. 修正後、必要に応じて `make_tables.py` / `check_evidence.py` / `prepare_figures.py` および `ruff` / `mypy`（変更したスクリプトがあれば）を実行する。
2. 新規・変更ファイルは `git add -f` で追跡登録し、`build/` 生成物はコミットしない。
3. クリーンなコミット済み HEAD で `bash paper/tennis_scene_architecture/build.sh --check-tracked` を実行し、終了コード 0（警告・Overfull・文字欠落・未解決参照・未追跡入力 0 件）を確認する。
4. `codex/parent-2627-issue-2638-draft` へ push し、PR #2720 本文の `## 実行証拠` を新 HEAD SHA・ページ数・照合結果で更新する。
5. PR CI の完了は待たず、修正内容と新 HEAD SHA・検証結果を簡潔に報告して `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終了する。
