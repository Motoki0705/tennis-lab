[PR #2225](https://github.com/Motoki0705/tennis-lab/pull/2225) のmajor 3件を修正しました。

- 混在キャッシュでも全対象clipのHMR2完了・解放後にGVHMRを開始。
- 公開再開・marker発行時にbody_parameters v2と保存特徴の依存を検証。
- PoseレビューとBlenderの配置export・rankingをv2対応。

branch: `codex/hmr2-feature-artifacts`  
HEAD: `a4fd96078186d19c68c4ded2b1e6cdaa28ec2945`

CPU **1,503件成功**、pre-commit成功。[CI全14ジョブ（Python含む）](https://github.com/Motoki0705/tennis-lab/actions/runs/44358682416)・[Knowledge全2ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/44358682504)も成功しました。

PR本文に各修正の根拠と読込先の確認表を記録済みです。Ready・競合なし・worktree clean。GPU使用・再レビューは0回です。merge・closeは親へ引き継ぎます。