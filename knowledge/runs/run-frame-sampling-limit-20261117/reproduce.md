CPU replayの再現手順（#1748）

元の計算コードはmain `486aa1ab09be857909d0fcf8c2b616c75a53f5b0`。
このPRの変更はYAMLの上限とテスト・記録だけで、計算Pythonの内容は
`source-parity.json`に固定した。Python/NumPy/SciPy/Torchとthread数は`results.json`。

保存済みclip_035のCourt校正・ball points・poseを読み取り専用で利用する。
入力97ファイルの場所・サイズ・SHA-256・artifact identityは`input-artifacts.json`。
元datasetへは書かず、GPUは使用しない。別checkoutで再現する場合は保存済みoutputsを
同じmain相対位置へ取得し、manifestの全SHAを照合してから実行する。
動画や新しいclipを作成する手順ではない。

repo rootから実行する。scriptsは保存した実行時の原文で、出力先は
`outputs/tennis_scene/evaluate/frame-sampling-limit-20261117`。
再実行はこの記録のGit管理下JSONを上書きするので、専用checkoutで実行する。

```bash
mkdir -p outputs/tennis_scene/evaluate/frame-sampling-limit-20261117
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python knowledge/runs/run-frame-sampling-limit-20261117/benchmark_limit.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python knowledge/runs/run-frame-sampling-limit-20261117/verify_finite.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  knowledge/runs/run-frame-sampling-limit-20261117/check_identity.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -n 0 \
  tests/unit/tennis_scene/test_configuration.py \
  tests/unit/tennis_scene/test_pipeline_configuration.py \
  tests/unit/tennis_scene/pipeline/test_automatic_geometry.py \
  -q --durations=0 --durations-min=0
```

`benchmark_limit.py`は保存済み実観測のsetup replayと追加warmup 1回の後、
5条件を上限2048/4096ごとに各5回測る。配列hash計算は時間窓外。
RSSは2ms間隔とprocess lifetime maxRSSであり、個別段の厳密な割当量ではない。
未実施の人物対応を仮定したcarrier-index IDと、実校正から投影したdenseな2人の
制御入力は計算量・数値確認のfixtureで、実際の人物対応・3D精度の証拠ではない。

`check_identity.py`は同一Python・同一資産の実standard_definitionを作り、
1024/2048のsettingsと依存閉包・publication settingsを比較する。
dataset.jsonのSHAと69clipの長さも固定する。モデルをロード・実行しない。
元の計測ではconfiguration変更前に実行したため、結果のsource_commitは上記main。
