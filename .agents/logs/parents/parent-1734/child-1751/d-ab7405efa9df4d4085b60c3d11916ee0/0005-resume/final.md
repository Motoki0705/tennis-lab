[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182) を **Ready for review** にしました。

- branch: `codex/scene-inference-runtime`[local path]`b05c056fc`
- knowledge 000052とPRに、親判断によるBall条件（uv ≤0.5px、3D ≤1mm、mask完全一致）を明記。実測は最大0.135px／0.362mmで合格です。元条件での不合格記録も保持しました。
- CPU追加確認6件成功。[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44204187669)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44204187796)とも全成功。
- 追加GPU実行は0回。差の原因はBF16のcompile/kernel丸め差が候補で、未特定です。

統合レビュー・merge・closeを親へ返します。