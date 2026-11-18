HMR2特徴の成果物分離を実装し、[PR #2225](https://github.com/Motoki0705/tennis-lab/pull/2225)をReady for reviewにしました。競合なしです。

branch: `codex/hmr2-feature-artifacts`  
commit: `a4b2a5226ffa`

- GPU：clip_037・4人物で分離前後の最大差0。有効配置1,156件が一致し、各モデルのload/unloadは各1回。
- CPU：1,216 passed / 50.18秒。pre-commit、最終HEADのCI・Knowledgeも全成功。
- 入力・許容差・recorder・原ログはknowledge記録53とPRへ保存済み。

CUDA異常の原因と全69clipは未確認です。追加GPU実行はありません。validatorは0/0/0。merge・closeは親へ引き継ぎます。