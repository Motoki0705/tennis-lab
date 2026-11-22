---
title: spin setup-nht は、実行したtreeの third_party/nht に全体共通の nht-render を結び付ける
summary: spin setup-nhtは実行したcheckoutのNHTに共通コマンドを結び付ける。作業用worktreeでは専用の実行パスで試す。
applies_to: all
source: nht-render の ModuleNotFoundError 調査（2026-10-16）
created: 2026-10-16
last_verified: 2026-11-22
evidence: setup_nhtのeditable installは.spin/cmds.pyのREPO_ROOTをcwdとして行われる。導入済みtoolのeditable finderのMAPPINGで参照先を確認できる。
citations:
- path: .spin/cmds.py
  symbol: def setup_nht
- path: src/synthetic_data_generation/rendering/nht/contracts.py
  symbol: def _validate_executable
  note: 専用ラッパーのパスに課される契約
---

同じ `uv tool` 環境を使うworktreeで `spin setup-nht` を実行すると、共用の `nht-render` / `nht-reconstruct` がそのworktreeのNHTへ付け替わる。2026-10-16には削除済みのworktreeを参照し、`ModuleNotFoundError: No module named 'nht_pipeline'` で失敗した。

worktreeを試す際に共用toolを更新せず、`nht-render` という実行可能なラッパーを用意できる。ラッパーからtoolのPythonを使い、試すNHTソースを `PYTHONPATH` に指定して `-m nht_pipeline.render` を実行する。ラッパーの絶対パスを `NHTRenderCommandRequest.executable` または `reveal.nht.executable` に指定する。変更した依存やCUDA拡張がある場合は、試す環境の導入版も照合する。
