# PR #2450: main の取り込み

#2449・#2469・#2461・#2610（SceneResult v3・ball event）などのマージで main が進み、PR #2450 は main と競合している（`tests/benchmarks/person_unseen_freeze.py`）。正式レビュー round 1（run parent-1739-1b51b8c3078245e4ab34994f4271b2cf）で、#2450 単体には停止事項がなかった。

1. `origin/main` をこのbranchへ merge する（rebase はしない）。競合は両方の意図を保つ形で解く。#2469 で入った3D軌跡の選別（`multiview_tracking/selection.py` のダブルス幅）と、このPRの `LinkingConfig` の既定値（ダブルス幅・滞在25%）が、矛盾せずに共存することを確かめる。
2. knowledge を足す場合は、main の tennis_scene `.sequence`（104）より後の番号を使う。#2462 が 105 を使う予定なので、106 以降にする。足さないなら触らない。
3. 関連テスト、pre-commit、`kg_validate.py --check-summary --base-ref origin/main` を通して push し、PR本文の「main の取り込み」と確認HEADを更新する。
4. ベンチマークとデータセットには触れない。GPU は使わない。比較の再実行はしない。

新HEAD（完全SHA）、解いた競合、テスト結果を報告し、CI待ちで終了する。
