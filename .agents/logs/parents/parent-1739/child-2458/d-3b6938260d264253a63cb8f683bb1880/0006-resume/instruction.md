# PR #2462: main の取り込みと knowledge の採番変更

正式レビュー round 1 の「参照を片方だけ切り替えた状態が下流に見える」は、前回のターンで共通の一貫した切替（HEAD b4b859d2）にして対処済み。main が #2449・#2469・#2461・#2610 で進み、このPRは main と競合している（`knowledge/nodes/tennis_scene/.sequence`、`src/tennis_scene/pipeline/storage/scene_export.py`、`tests/integration/tennis_scene/test_fasthmr_recipe.py`）。

1. `origin/main` をこのbranchへ merge する（rebase はしない）。
   - `scene_export.py`: main の #2610 で SceneResult v3（export が `event_detection` を参照し、その依存先に `scene_assembly` がある）が入った。外部ボールの公開と一貫した切替が、v2 と v3 の両方の scene で正しく働くようにする。どちらの版かを推測で決めない。既存の系譜検証に沿って読む。
   - v3 の保存済み scene（例: `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3` か main の fixture）で、切替・読込・復旧のテストを足す。ベンチマークは読み取りだけにし、書き込まない。
2. knowledge: このPRのノード `000098-run-meiji-external-ball-20261124.md` を `000105-run-meiji-external-ball-20261124.md` へ移し、`.sequence` を 105 にする。ファイル内の番号・参照も更新する。run ID は変えない。
3. 共通の一貫切替の部品は、後で #2470 の `multiview_scene publish/restore` が使う。公開の入口（関数名・引数・エラー）が main のマージ後も安定していることを、README に1か所だけ書く。
4. キャッシュ互換（FP32 など）の照合値は、main の実ソースに結び直す。完全一致時に一致し、編集時に失効することを確かめる。値をただ再生成して合わせない。
5. 関連テスト、pre-commit、`kg_validate.py --check-summary --base-ref origin/main` を通して push し、PR本文を更新する。
6. `i2363-all-v1/inputs/dataset` とベンチマークには書き込まない。GPU は使わない。

新HEAD（完全SHA）、解いた競合、v3 への対応、テスト結果を報告し、CI待ちで終了する。
