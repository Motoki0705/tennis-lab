# #2440: main を取り込み、PR #2450 をマージ可能にする

親の方針（#1739 コメント https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7184189944 ）に従い、#2450 → #2462 の順にマージする。

1. `origin/main` をこのbranchへ merge する（rebase はしない）。現在 `knowledge/summary.md` が競合している。両方の記述を保つ形で解く。
   - main 側で選手選別まわり（`person_tracking`・`player_association`・`tennis_scene/pipeline`）が変わっていないか確かめる。意味的な衝突があれば解いて報告する。
2. 関連テスト、pre-commit、knowledge 検査（`kg_validate.py --check-summary --base-ref origin/main`）を通す。
3. push して PR #2450 の本文を現状に合わせ、Draft を解除する。
4. ベンチマークとデータセットには触れない。GPU は使わない。

PR の新HEAD（完全SHA）とテスト結果を報告し、CI待ちで終了する。
