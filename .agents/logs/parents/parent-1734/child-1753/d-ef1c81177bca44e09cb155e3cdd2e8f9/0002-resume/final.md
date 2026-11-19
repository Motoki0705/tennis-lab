担当PR [#2226](https://github.com/Motoki0705/tennis-lab/pull/2226) を更新しました。指示内の#2225は別案件です。

- branch: `codex/person-feature-artifacts`
- HEAD: `698129d6ef6dbfc955fc17ee07d3c0a838d99fe2`
- mainのModelSpec・精度設定・失敗伝播・court side保護を維持して衝突を解消。外観特徴v2・追跡v7へ接続しました。
- CPU **1,583件成功**、追加統合テスト5件・pre-commit成功。
- [CI全14ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/44345777299)（Python含む）と[Knowledge CI全2ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/44345777570)成功。

PR本文更新済み、Ready・merge可能、worktree cleanです。GPUは使用せず、統合後の実GPU推論は未確認。validatorは指定／試行／完了0回です。統合レビュー・mergeは親へ返します。