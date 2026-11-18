# FP32キャッシュ互換テスト

`main_fp32_cache.tar.gz` は、共有court sideを取り込んだmain `dcd6534594f73cca12a45c2d9afaa8ac09d48bc7` の
`TennisSceneOrchestrator` と `ClipStore` が保存した3camera・24frameのfixture。
既存の `test_auto_pipeline.py` の数値と仮の重みを使い、モデル推論を固定の出力に置き換えている。
他のディレクトリで開けるよう、入力動画・重みのidentityに相対パスを使う。
`baseline.json` は同じ実行の公開identityとソースファイルのSHA-256を保持する。

再生成する場合は、このテストを含むworktreeから実行する。

```bash
worktree=$(pwd)
baseline=$(mktemp -d)
scratch=$(mktemp -d)
git archive dcd6534594f73cca12a45c2d9afaa8ac09d48bc7 src tests/unit/tennis_scene/pipeline/test_auto_pipeline.py |
    tar -x -C "$baseline"
cd "$baseline"
PYTHONPATH="$baseline" "$worktree/.venv/bin/python" \
    "$worktree/tests/integration/tennis_scene/test_fp32_cache_compatibility.py" \
    "$scratch" "$worktree/tests/integration/tennis_scene/fixtures/main_fp32_cache.tar.gz"
```
