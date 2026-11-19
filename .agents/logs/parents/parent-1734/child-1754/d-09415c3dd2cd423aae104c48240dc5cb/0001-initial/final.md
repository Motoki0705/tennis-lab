実装とCPU・CI検証を完了しました。実GPU受入は未達のため、[PR #2225](https://github.com/Motoki0705/tennis-lab/pull/2225)はdraftです。

- branch: `codex/hmr2-feature-artifacts`
- commit: `61eb1ca8545ca0cc9b53f1f9918707845068461a`
- HMR2特徴保存node、GVHMRの保存特徴入力、schema v2、README・galleryを更新。
- CPU **1,170件成功**。pre-commit、[必須Pythonを含むCI](https://github.com/Motoki0705/tennis-lab/actions/runs/44086221438)、Knowledge CIも成功。途中で検出した取得案内・gallery登録の退行は修正済みです。

GPU確認は固定`clip_035`の人物対応が`player_association_ambiguous_players`で停止し、身体段へ未到達です。比較可能な固定入力または保存済みBodyViewSelectionの選定を親へ返します。証拠はPR・担当issue・knowledge 000044へ保存済みです。

validator指定／試行／完了は0／0／0。merge・issue closeは未実施です。