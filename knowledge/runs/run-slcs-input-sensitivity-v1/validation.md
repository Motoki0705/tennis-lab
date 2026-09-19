# 検証記録

2026-09-19、専用worktree、CPU・CUDA_VISIBLE_DEVICES空・OMP/MKL各2thread。

- 実行: `probe.py --output-dir knowledge/runs/run-slcs-input-sensitivity-v1/result`、exit 0、status completed。
- `python -B -m pytest -o addopts='' -q knowledge/runs/run-slcs-input-sensitivity-v1/test_probe.py`: 最終版で3 passed (7.52s)。通常の並列worker起動を避けた小規模CPUテスト。
- `ruff check knowledge/runs/run-slcs-input-sensitivity-v1`: passed。
- `ruff format --check knowledge/runs/run-slcs-input-sensitivity-v1`: passed。
- `mypy --follow-imports=silent knowledge/runs/run-slcs-input-sensitivity-v1/probe.py knowledge/runs/run-slcs-input-sensitivity-v1/test_probe.py`: Success, 2 source files。初回の型エラー4件は型注釈・型検査・非None検査・保存APIの明示引数で修正した。
- `kg_curves.py run-slcs-input-sensitivity-v1`: exit 0、0 written / 1 skipped。追加学習なしのため対応TensorBoardなし。
- `kg_validate.py`: exit 0、286 nodes / 0 errors / 126 warnings（issue未指定）。
- 実行時ソースsnapshot SHA-256 `295dab0828b1bebdfb2e9231eb978f0f3fa1d592344e2280c2102ad4405e3848` は結果JSONに一致。
- full出力は両domainで親probeのeval出力と完全一致、モデルstate不変。
- 実入力hashで変更keyを確認: broadcast ball_reverseはball_uvのみ（ball_visは反転しても同値）、Meijiはball_uv/ball_vis、DINOは両domainともdino_tokensのみ。教師・時刻slot・paddingは同値。
- source hashの検査失敗はなかった。validator起動なし。
