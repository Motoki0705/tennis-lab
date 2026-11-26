# #2672 main CI の修正（親 #1722）

担当Issue #2672 の本文が受入条件の正本。main の CI が赤なので、最小の変更で急いで直す。

- 親は origin/main `1e47e5d3` で、失敗するテスト `tests/integration/tennis_scene/test_fp32_cache_compatibility.py::test_multiview_integration_preserves_main_source_identity` を再現した。`src/tennis_scene/pipeline/multiview_scene_cache_compatibility.json` の `utils/configuration/inventory.py` の `after` を `95b4cd05267ed635be8af4d581eb9d38cf7efc69a0cbb300fce72fccca8ba0e4` に変えると通ることも確かめた。
- 互換表は #1739 の親（#2470）の領域。今回の変更は `after` の1項目に限る。ほかの項目・テスト・仕組みは変えない。
- 検証: 同ファイルと `tests/integration/tennis_scene/test_event_detection_cache.py` を修正後の HEAD で実行する。pre-commit を通す。
- commit・push して PR を作る（本文は日本語、`Closes #2672`）。PR 本文の実行証拠は最終 HEAD で書き、CI を `PARENT_RESULT` の wait で宣言して終了する。
