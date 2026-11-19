PR #2149 の統合レビュー（run parent-1893-3888740872ec4f539b32a698c24135cc、HEAD ee23fdad、request_changes）で major 1件が出た。妥当と判断した。修正してほしい。正式レビューは1回だけなので、修正後は親が差分・回帰テスト・CIを確認して resolve で解消を記録する。

### pr-2149-1 .agents/ops/review/parent_review_policy.py:809（major）
静的推論（テストによる再現は未実施）。修正head Fをresolveした後、baseも変更したファイルをPRで衝突解消してhead Cにすると、resolution_scopeはそのファイルを確認対象にします。しかしverify_rebased_snapshot(F,C)は「baseでも変更されたパス」というだけでTrueになり、gateはfixed_inheritedとして現在snapshot・確認範囲・diff digestの照合を省略します。そのため、衝突解消で修正を取り消してもFのresolutionで指摘解消済みとして通過し得ます。旧継承経路にも新しい確認範囲を適用し、PR差分が変わった場合は親の再確認を要求してください。追加テストはrenameを併発させて継承を拒否しており、この抜け道を検出できません。renameなしの衝突解消と、resolve後に修正を戻すケースでgateの拒否を検証する必要があります。

## 親の指示
- resolve 後の head 変化（rebase・merge・衝突解消）を gate が継承扱いにする経路（verify_rebased_snapshot / fixed_inherited）にも、新しい確認範囲の定義を適用する。resolve 時の fixed head から現在 head までに、PR自身の差分が変わったファイル（base とPRの両方が変えたファイルの衝突解消を含む）があれば、継承を認めず親の再確認（新しい resolve）を要求する。base 側だけの変更でPR自身の差分が変わらない場合だけ継承を認める。
- 回帰テスト: renameなしの衝突解消で修正を戻すケース、resolve 後に修正を戻すケース、base だけが進んでPR差分が変わらないケース（継承される）を、gate の判定で検証する。修正を外すと前2つが失敗すること。
- 他に、確認範囲の照合を省略している経路（complete-evidence の継承など）がないか調べ、結果をPR本文に書く。
- 関連テストと pre-commit を通し、通常の push（force pushしない）、PR本文の更新、対象HEADのCI確認まで進める。pytest の並列は `-n 4` 以下。GitHub API は共有上限が逼迫しているので CI の確認は必要最小限にする。実PRへの resolve/gate、systemd・git-sync 操作はしない。作業時間の目安は最大3時間。
