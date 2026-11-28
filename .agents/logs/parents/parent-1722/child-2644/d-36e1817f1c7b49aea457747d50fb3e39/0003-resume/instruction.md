# PR #2657 正式レビュー（round 1）の major 修正

対象HEAD `5c89ec41be1599185196f4ca32b435353dc3e3a4`。Sol の正式レビューで major 1件が出た。親は採用する。

## Direct annotation の公開再開が、ソース変更後も古いイベント成果物を受け入れる

`direct_annotation.resume=publication` は実行計画を通らず、`saved_scene()` に `code_commit` を除いたイベント設定を渡す（`orchestrator.py` L258–300、`direct_annotation/__init__.py` L119–166 の `saved_scene`）。`saved_scene()` は、保存時の設定・採用参照・アーカイブのチェックサムを照合するが、保存された `event_detection` の実装識別を現在の `self.code_identity` と比べていない。そのため、たとえば保存後に `ball_events/rules.py` を変えても、設定と保存済みファイルが同じなら、旧イベント付きの SceneResult を読み込んで公開に進んでしまう。以前は設定の `code_commit` がこの役割を（過剰に）果たしていた。

直し方:
- 再開時に、正規化済みの実装識別（実行計画と同じ `compatible_source_digests` 由来のもの）を照合する。
- commit だけが変わった場合は、元の来歴（計算時の `code_commit`）を保持したまま再開できる。
- イベントのソースが変わった場合は、公開前に明示的に拒否する。
- 既存の公開再開テストで、上の2つの場合を検証する。
- 旧 `scene-completion.json`（設定に `code_commit` を含む形式）を持つ利用者向けの再開手順を README に明記する。

あわせて、実行計画を通らずに保存済みの成果物を受け入れる経路がほかにないか（publication・load・resume）を確かめ、PR 本文に短く書く。

修正 commit と回帰テストを push し、PR 本文の実行証拠を最終 HEAD で更新して、CI を `PARENT_RESULT` の wait で宣言して終了する。
