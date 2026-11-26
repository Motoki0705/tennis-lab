# #2459: 更新した #2444 を取り込み、PR #2461 をmainへマージ可能にする

マージ管理は、子issueの先行依存をPRがmainへマージされたかで判定する。そのため、新経路は #2449 → #2469・#2461 → #2470 の順にマージする（#1739 の方針 https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7184189944 を修正）。

1. #2449（#2444）は main にマージされた（merge commit `fcb97df3a64e4445efee3d943e419dd5bd928201`、最終HEAD `9d6630ba`）。`origin/main` をこのbranchへ merge する（rebase はしない）。#2449 の最終修正（`multiview_pose/inputs.py` の enabled 検査、未見評価の再現手順）も入る。
2. knowledge の連番: #2449 は98〜100、#2469 は101〜103 を使う。このbranchで tennis_scene のノードを新たに足す場合は、親に相談する（`.sequence` は #2444 の100のまま）。
3. キャッシュ互換（`source_identity.py`・`inventory.py`・`test_fp32_cache_compatibility.py`）は、#2444 の取り込み後のソースで、完全一致時は互換になり、編集時は失効することを確かめる。値をただ再生成して合わせない。
4. 関連テスト、pre-commit、knowledge 検査（`--base-ref origin/main`）を通して push する。
5. PR の base は親が main に付け替えた。PR本文は `Closes #2459` のままで、現状に合わせて更新し、Draft を解除する。
6. ベンチマークとデータセットには触れない。GPU は使わない。

新HEAD（完全SHA）とテスト結果を報告し、CI待ちで終了する。
