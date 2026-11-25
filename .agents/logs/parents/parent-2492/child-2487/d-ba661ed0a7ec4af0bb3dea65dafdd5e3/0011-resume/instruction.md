# #2487 統合担当: PR #2499 を main に向け直し、main を取り込む

前回の報告（`9b5556898`、44行の比較表、Pareto 図2枚）を受け取った。内容を承認する。

## 何が起きたか
- PR #2501（FastHMR の経路、子 #2558）は、正式レビューの major 3件を直したうえで main にマージされた（`466e92d34`）。修正は commit `9c48c47a7` にある。
  - FastHMR の load は、component が生成した成果だけを受け付ける（`ComponentNode.component_origin_only`）。
  - FastHMR の固定重みを、計画と publication で identity を組む時点で照合する。
  - VAE を CPU に復元してから指定デバイスへ移す。
- 親が PR #2499 の base を `codex/parent-2492-issue-2491-primary` から `main` に付け替えた。今は main とコンフリクトしている。
- マージ後の main で、FastHMR の単体テストの import が境界検査に引っかかった。子 #2572 の PR #2573 で直していて、正式レビュー中。

## やること
1. `git fetch origin main` をして main を取り込み、コンフリクトを解く。FastHMR 側（`definition.py`・`execution_plan.py`・`body_artifacts.py`・`dataset_publication.py`・`standard_recipe_cache_compatibility.json` など）は main の内容を正とする。ECPose 経路の変更がこれらに触れていれば、main の新しい仕組み（`component_origin_only`、計画時の資産照合）に合わせる。ECPose 経路にも同じ問題（別精度の成果のインポートを受け付ける、推論対象が空のとき資産の hash を照合しない、重みを設定外のデバイスに復元する）がないか確かめ、あれば同じ方針で直して回帰テストを足す。
2. 標準 FP32 キャッシュの互換 manifest を更新する必要があれば、既定 `standard` の結果が変わらないことを回帰テストで確かめてから更新する。
3. 境界検査（`tests/e2e/development/test_architecture_boundaries.py`）は通常の PR の CI では繰り延べられ、main への push でしか走らない。ローカルで実行して通ることを確かめる。`__init__.py` のない package から `from src.x.y import module` の形で import していないかも見る。
4. 関連テスト、pre-commit、knowledge 検査を通して push する。PR #2573 が先に main に入ったら、それも取り込む。PR 本文の検証の節を新しい HEAD に合わせて直す。

## 終わり方
最終回答には、解いたコンフリクト、ECPose 経路で直したこと（あれば）、push した HEAD を書き、新しい HEAD の CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"<新HEAD>","resume_on_success":false}}` を返す。CI が通れば、親が正式レビューに回す。
