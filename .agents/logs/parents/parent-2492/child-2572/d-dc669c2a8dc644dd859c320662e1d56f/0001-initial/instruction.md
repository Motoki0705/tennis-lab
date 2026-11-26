# 子 #2572: main の CI を戻す（FastHMR の単体テストの import）

あなたは親 #2492 の子 #2572 の担当。issue 本文の受入条件が正。急ぎの修正なので、小さく直してすぐに PR を出す。

## 状況
- PR #2501 のマージ後、main `466e92d3408894bd83545981df9d130c37107144` の CI が失敗した（run 45668593936、`CPU tests (3)`）。
- 失敗は `tests/e2e/development/test_architecture_boundaries.py::test_static_src_import_modules_exist`。`tests/unit/submodules/models/fasthmr/test_upstream.py:12` の `from src.submodules.models.fasthmr import upstream` が「unresolved repository imports」になる。`src/submodules/models/fasthmr/` には `__init__.py` がない。

## やること
1. 専用の worktree と branch で、最新の main から作業する。
2. テストの import を、検査が解決できる形に直す（例: `import src.submodules.models.fasthmr.upstream as upstream`）。テストの中身は変えない。同じ形の import がほかにないか、検査をローカルで実行して確かめる。
3. 対象の2つのテスト（境界検査と `test_upstream.py`）と pre-commit を通す。
4. PR #2501 の CI でこの検査が選ばれなかった理由（CI のテスト選択の仕組み）を調べ、PR 本文に短く書く。仕組みを直す変更はこの PR に入れない。
5. PR を main に向けて作る。本文は日本語で、先頭に `Closes #2572`、`Refs #2492, #2558` を書く。draft にしない。

## 終わり方
最終回答に、直した内容、テスト結果、PR 番号と HEAD を書き、CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<HEAD>","resume_on_success":false}}` を返す。
