# 再現と入力来歴

実装commitは`5fa344519`。幾何評価は同梱の`legacy_geometry/`（`b46b97869`）と
`current_geometry/`を使う。後者と実装の数値処理が同一であることは
`snapshot-verification.json`に記録した。GPUモデル出力は一度取得して全条件で共用した。
`implementation.patch`は同期後のbase `8d05a5513`からの変更をcontext 0で保存したレビュー用diffで、
同梱raw fixtureのbinary本体を含まない。実装・fixtureの再現には上記のGit commitを使う。
`protocol.json`の参照SHA・許容差・目標を確認し、`references.json`の点を変更しない。

元実行の正確なGPUコマンド・CWD・provider/session/issue/resourceは3つの`.job`に保存した。
これらは記録であり、queueを迂回して直接実行しない。`queue-jobs.json`に完了状態とログ位置がある。
CPUの並列数はsceneの旧新各8、valの旧新各4→6、synthetic testの旧新各2、品質集計各2。
valは他の評価完了後にCPUを再配分し、atomicに保存済みの画像は再計算せず続行した。
`cpu-resume.json`と各`complete.json`を参照する。後者のsecondsは再開後の経過時間なので、
最初からの総実行時間と取り違えない。

資産は正規のstorage-tier経由で復元した。

```bash
PYTHONPATH=.agents/ops PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- .venv/bin/python -m storage_tier --repo /home/kamimura/projects/tennis-lab hydrate data/court_detection/tennis_court_detector-v1
```

実画像は元splitのtrain 6629 / val 2211（既知除外ID適用後）、testは未定義。
SyntheticCourt V3はB00〜B03の元split train 6594 / val 935 / test 886のうち明示的testを取得した。
データdescriptor・index・RGB・checkpoint等のhashはoutputsのmanifestと`input-identity.json`に保持した。
既定checkpointを学習したrunの来歴が欠けるため、synthetic testとの学習集合重複は未確定。
施設分離の実画像testや独立汎化性能を主張する記録ではない。

既存のraw配列は次にある。削除・上書きせず再集計できる。

```text
outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116/
  raw/           # scene 448画像、5,433領域
  val_raw/       # 元実画像val 2,211画像、full-frameのみ
  test_raw/      # 明示的synthetic test 886画像、full-frameのみ
  legacy/ current/ val_legacy/ val_current/ test_legacy/ test_current/
```

新規取得するときは、専用worktreeで同梱scriptとsource snapshotを別のoutputsディレクトリにコピーする。
`cache_raw.py`等は同じrepoの読み取り専用data/ckpt/third_partyの配置を前提とする。
下記は元実行を再現するコマンド例。sessionは再実行する自分の実session IDを指定する。

```bash
COURT_REPLAY=outputs/tennis_scene/evaluate/court-acceptance-replay
COURT_BUNDLE=knowledge/runs/run-court-acceptance-implementation-20261116
mkdir -p "$COURT_REPLAY"
cp "$COURT_BUNDLE"/*.py "$COURT_BUNDLE/protocol.json" "$COURT_BUNDLE/references.json" "$COURT_REPLAY/"
cp -R "$COURT_BUNDLE/legacy_geometry" "$COURT_BUNDLE/current_geometry" "$COURT_REPLAY/"
export TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue
COURT_QUEUE=.agents/skills/training-queue/scripts/training_queue.sh
bash "$COURT_QUEUE" add "PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u $COURT_REPLAY/cache_raw.py --output $COURT_REPLAY/raw" --name court-acceptance-replay-scenes --provider codex --session "$COURT_REPLAY_SESSION" --issue 2190 --resource half
bash "$COURT_QUEUE" add "PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u $COURT_REPLAY/cache_validation.py --output $COURT_REPLAY/val_raw" --name court-acceptance-replay-val --provider codex --session "$COURT_REPLAY_SESSION" --issue 2190 --resource half
bash "$COURT_QUEUE" add "PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u $COURT_REPLAY/cache_synthetic_test.py --output $COURT_REPLAY/test_raw" --name court-acceptance-replay-test --provider codex --session "$COURT_REPLAY_SESSION" --issue 2190 --resource half
bash "$COURT_QUEUE" start
```

GPU取得完了後、CPUで後処理する。下記scene用の`raw/legacy/current`を
`val_raw/val_legacy/val_current`、`test_raw/test_legacy/test_current`へ置き換えれば同じ評価になる。

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/evaluate_geometry.py" --raw "$COURT_REPLAY/raw" --output "$COURT_REPLAY/legacy" --mode legacy --workers 6
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/evaluate_geometry_current.py" --raw "$COURT_REPLAY/raw" --output "$COURT_REPLAY/current" --mode current --workers 6
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/summarize_final.py" --raw "$COURT_REPLAY/raw" --legacy "$COURT_REPLAY/legacy" --current "$COURT_REPLAY/current" --output "$COURT_REPLAY/scene-evaluation" --workers 2
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/summarize_final.py" --raw "$COURT_REPLAY/raw" --legacy "$COURT_REPLAY/legacy" --current "$COURT_REPLAY/current" --output "$COURT_REPLAY/meiji4-evaluation" --workers 2 --max-cases 276
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/produce_scene_evidence.py"
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/paired_regression.py" --raw "$COURT_REPLAY/val_raw" --legacy "$COURT_REPLAY/val_legacy" --current "$COURT_REPLAY/val_current" --output "$COURT_REPLAY/val-regression"
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python "$COURT_REPLAY/paired_regression.py" --raw "$COURT_REPLAY/test_raw" --legacy "$COURT_REPLAY/test_legacy" --current "$COURT_REPLAY/test_current" --output "$COURT_REPLAY/test-regression"
```

元Court APIの幾何成立率と新しいscene gateの採否は別に集計する。失敗を誤差ゼロにしない。
原画像pixelの誤差、両条件とも成立した同じ画像・同じ可視GTでの誤差、失った／得た推定を並記する。
raw KP指標は既存review metricと同じ可視GT点の距離だが、学習時のモデル座標metricとは単位が異なる。
SEG・semantic LINE・poseの評価を再実行したとは主張しない。モデルhead自体は変更していない。
