# #2458: main を取り込み、PR #2462 のCI失敗を直してマージ可能にする

親の方針（#1739 コメント https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7184189944 ）に従い、#2450 → #2462 の順にマージする。

1. `origin/main` をこのbranchへ merge する（rebase はしない）。現在 `knowledge/nodes/tennis_scene/.sequence` が競合している。
2. HEAD `c839934f` のCIは次の2件で失敗している。
   - `CPU tests (3)`: `tests/integration/tennis_scene/test_fp32_cache_compatibility.py` の `test_current_fp32_cache_is_adopted_in_execute_and_load_modes` と `test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources` が、`code_sha256` の不一致（`31f72e2b…` と `6022ecaa…`）で失敗。
   - `Python` ジョブ: `digest-mismatch: error`。
   どのソース変更で hash が変わったのかを特定してから直す。fixture の値をただ再生成して合わせない。互換条件の意味が保たれることを確かめる。
3. 関連テスト、pre-commit、knowledge 検査を通す。
4. push して PR #2462 の本文を現状に合わせる。
5. `i2363-all-v1/inputs/dataset` とベンチマークには触れない。GPU は使わない。

PR の新HEAD（完全SHA）、CI失敗の原因と修正、テスト結果を報告し、CI待ちで終了する。
