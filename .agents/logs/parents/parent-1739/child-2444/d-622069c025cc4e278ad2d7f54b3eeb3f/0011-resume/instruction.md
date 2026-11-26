# #2444: main を取り込み、PR #2449 をマージ可能にする

親の方針（#1739 コメント https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7184189944 ）では、新経路を #2470 にまとめてマージする予定だった。しかしマージ管理が、子issueの先行依存（#2446 は #2445・#2459 に依存）を、各子のPRがmainへマージされることで判定する。そのため、積み上げどおり #2449 → #2469・#2461 → #2470 の順にマージする。

1. `origin/main` をこのbranchへ merge する（rebase はしない）。競合は `knowledge/nodes/tennis_scene/.sequence`、`src/tennis_scene/pipeline/source_identity.py`、`src/utils/configuration/inventory.py`、`tests/integration/tennis_scene/test_fp32_cache_compatibility.py` の4件。
   - 同じ競合は、#2446 のbranchが merge commit（`b62a0f3d` の祖先のmain取り込み）でもう解いている。その解き方を参考にしてよい。ただしこのbranchにあるのは #2444 の変更だけなので、#2445・#2459・#2446 の内容を持ち込まない。
   - main 側の ECPose・FastHMR・単眼PLCS の recipe と共存させる。既定経路は変えない。キャッシュ互換の fixture は、値をただ再生成して合わせない。どのソース変更で hash が変わったかを特定して直す。
   - 正式レビューでは「後からマージされたPRの `inventory.py` の変更でマニフェストの照合値がずれる」「knowledge の連番が別PRと重複する」が指摘された。この2点に気を付ける。
2. 関連テスト（`tests/unit/tennis_scene`・`tests/integration/tennis_scene`・multiview_pose 関連）、pre-commit、knowledge 検査（`kg_validate.py --check-summary --base-ref origin/main`）を通す。
3. push して PR #2449 の本文を現状に合わせる。本文の閉鎖指定は `Closes #2444` のままにする。
4. ベンチマークとデータセットには触れない。GPU は使わない。

新HEAD（完全SHA）と、解いた競合、テスト結果を報告し、CI待ちで終了する。
