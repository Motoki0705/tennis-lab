# 重いテストの準備と評価

[English](heavy-test-cost.md) · [简体中文](heavy-test-cost.zh-CN.md)

親 [#1390](https://github.com/Motoki0705/tennis-lab/issues/1390) の子
[#1522](https://github.com/Motoki0705/tennis-lab/issues/1522) で、既存のmerge gate・
coordinates training全ケースを維持し、seedの独立性テストを1件追加する。
本番コード・既定値・CPU対象の選択・分割用の時間profileは変更しない。

## 変更と維持する契約

merge fixtureは、各pytest workerでvariantごとに実gitとparent/review CLIを通して
レビュー済みscenarioを作る。session内のcacheにはprivateなsnapshotと、最初のテストから
切り離したPython metadataを保持する。以降のケースにはfilesystemをコピーし、GitHub・
registry・review記録、gitのobject・ref・index・worktreeを独立させる。絶対private pathを
移し替え、実際の `git worktree repair` で移設先を確認する。binary state内にprivate pathが
あれば明示的に失敗する。cache本体をテストの書き込み対象には渡さない。
各gate操作は従来どおり実merge CLI subprocessとfake GitHub CLIを起動する。
head/base/diffの再収集とmergeの呼び出しはmock・cache化しない。

学習smokeは実BLCS生成・4ラリー・4カメラを維持し、シミュレーションを256物理step
（出力64 frame）で打ち切り、最短32 frameとする。seedは992に固定し、validation/testの複数飛行区間を明示的に検証する。短窓は32 frame・長窓の上限は64 frame。
既存の幅16・1層のモデル、batch 2、roundtrip 4更新、resume 5更新、Flow sampling 3 step、
2更新ごとのvalidation、最後の端数epochを維持する。CPU trainer threadは1に明示する。
保存時の物理診断には実際の非線形fitを使い、このテストのevaluator境界でだけ
初期値1通り・segment単独0反復・joint 1反復の設定を注入する。
予測・loss・勾配・optimizer・checkpoint・plot・metricの出力は置き換えない。

| 旧ケース・相互作用 | 維持先・検出する不具合・固有の相互作用 |
|---|---|
| 再投影と共有生成 | 同じ再投影テストとroundtrip内のsplit検査。実生成・上書き拒否、4レコードのtrain/val/testラリー独立性、3D/events/timeの完全一致、カメラ間で変わる可視2D再投影 |
| 回帰roundtrip | 同じケース。normal fit、4更新時の非空AdamW state、有限な重み付きloss、監視指標によるbest/last checkpointの再読込、TensorBoard events、seed付き評価の完全一致と推論file契約 |
| GAN・GAN-only roundtrip | 両ケースを維持。generator係数0/0.5/1/1、discriminator 3更新、再構成係数1/0.5/0/0、各項とtotalの一致、最終GAN-only目的関数 |
| Flow roundtrip | 同じケース。実Flow loss/backward/optimizer、3 stepのseed付きsampling、best/last再読込と予測・イベント確率の完全再現 |
| 物理head roundtrip | 同じケース。4種類の正の物理lossと重み付き和、積分軌道配列、場・surface・segment出力と推論schema、実fitの有限な残差とsmoke用設定の保存 |
| 回帰・Flow・GANのresume | 同じ3ケース。2更新後に中断、native checkpointから5更新・epoch 2まで復元、最後の1更新epoch、dropout、連続実行と再開の全重み・Torch/Flow RNG・学習曲線の完全一致 |
| 物理fitの数値的収束 | 既存 `test_physics_evaluation.py` と `test_ball_fitting.py` を変更せず維持。本番の多始点protocol、物理軌道の正確なfit、非物理残差、複数segment/surface、イベント、不正入力を検証する。数値精度はこれら、lifecycleの配線はsmokeが担う |
| 人間の保留、review欠落/中断/未確認、projection/head/base/diffの整合、draft/fork/conflict/不明入力 | 既存gateケースと全CI状態の組合せを維持。private copyだけを変更し、実merge CLIを呼び、個別の拒否理由とagent/merge非起動を検証 |
| 独立・依存PR、部分判定、base前進、古い/失敗/応答消失merge、修正、head競合、新規PR | 同じ3 PR variantとscenario本文。実git履歴の変更とCLIの再収集を維持し、相互作用のあるケースは統合しない |
| 単独issue、main以外の既定branch、共有path、完了済み研究依存、不正auto-merge、対象なし | 同じ専用variant・policy・CLIケース。既存の最小scenario規模を維持 |
| `test_execution_basis.py` / `test_judgement_execution.py` | 3 PR/shared variantと明示的な `make_gate_env` 呼び出しを含む下流scenarioを変更しない |
| 追加した独立性テスト | コピー先でcommit・worktree追加・GitHub状態変更・review証拠削除を行う。コピーの実CLIは拒否し、元のdry runは成功する。元のref/state/resultとseedの全bytesは不変、git object inodeは別、最初のケースによるmetadata変更もtemplateへ伝播しない |

## 同じローカル環境での計測

過去の [Actions run 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) の残る上位ケースは下表。merge gateのsetup/call合計は1129.879/713.591秒。学習roundtripは95.1–96.1秒、物理head付きは190.7秒、resumeは0.95–1.08秒だった。優先順位の根拠であり、ローカル前後比較の分母には使わない。

| Actions case | Setup s | Call s |
|---|---:|---:|
| `test_all_targets_merge_in_one_run_with_verified_base_advances[True-independent]` | 11.159 | 44.416 |
| `test_all_targets_merge_in_one_run_with_verified_base_advances[False-independent]` | 11.290 | 43.712 |
| `test_new_child_pr_does_not_invalidate_reviewed_prs[multiple]` | 11.188 | 36.792 |

| ファイル | 件数 | setup秒 | call秒 | 合計秒 | 削減率 |
| --- | --- | --- | --- | --- | --- |
| `test_merge_gate.py` | 162 → 163 | 1804.797 → 175.366 | 1047.166 → 942.883 | 2852.110 → 1118.353 | 60.8% |
| `test_coordinates_training.py` | 9 → 9 | 15.622 → 0.785 | 1326.628 → 25.534 | 1342.258 → 26.324 | 98.0% |

ローカルの完全な1組。変更前171 passed、変更後172 passed。command wall time: 2257.648 → 595.396秒。旧ケースの削除なし。

学習ファイル9件のcall時間はすべて **6.372秒以下**。

値はworker間で重なるpytest経過秒の合計であり、CPU時間やCI全体の短縮予測ではない。
旧inventoryをすべて維持し、変更後合計には追加した独立性テストも含める。
2 workers・`--dist=worksteal`・native thread 1・CUDA非表示で実行する。
fit時間は評価・学習時間に内包されるため、phase時間へ加算しない。

| 操作 | 変更前 | 変更後 |
| --- | --- | --- |
| test_merge_gate: `fixture_parent_env` | 157 | 158 |
| test_merge_gate: `fixture_gate_env` | 156 | 157 |
| test_merge_gate: `reviewed_gate_preparations` | 156 | 13 |
| test_merge_gate: `seed_restore` | 0 | 145 |
| test_merge_gate: `git` | 2734 | 1396 |
| test_merge_gate: `git_commit` | 601 | 219 |
| test_merge_gate: `worktree_add` | 268 | 28 |
| test_merge_gate: `parent_cli` | 1400 | 142 |
| test_merge_gate: `review_cli` | 168 | 25 |
| test_merge_gate: `gate_cli` | 169 | 171 |
| test_coordinates_training: `model_init` | 14 | 14 |
| test_coordinates_training: `optimizer_step` | 64 | 64 |
| test_coordinates_training: `torch_save` | 56 | 56 |
| test_coordinates_training: `evaluate` | 62 | 62 |
| test_coordinates_training: `physics_fit` | 24 | 24 |
| fake GitHub CLI | 64554 | 29446 |

実際の非線形fit: 1291.517 → 13.374秒。`git` はpytest helperの呼び出し回数で、CLI subprocess内のgitはこのカウンターの対象外。`model_init` は学習Lightning moduleの構築回数で、checkpoint/推論用modelの生成回数は含まない。

コピーした過去のfake GitHub呼び出しログ行は実行回数から差し引く。
基礎の `parent_env` は各ケースで独立した書き込み可能環境を初期化し続ける。
共有するのはレビュー済み準備だけで、モデル初期化・optimizer更新・保存時評価・
checkpoint I/Oは実際に実行する。

## 不具合注入と検証

| Fault | 変更前 | 変更後 |
| --- | --- | --- |
| `human_hold` | 3 AssertionError | 3 AssertionError |
| `projection_integrity` | 2 AssertionError | 2 AssertionError |
| `resume_rng` | 3 AssertionError | 3 AssertionError |
| `reconstruction_schedule` | 1 AssertionError | 1 AssertionError |

関連commandは **187件pass**。未変更の実行根拠・judgement scenario **71 + 12件**、
ball refiner unit **97件**、共有の数値fit **5件**、割当・新規ケース確認 **2件**を含む。
本番の多始点protocolと収束検証を維持する。timeoutした2 scenario、head競合、
修正有無のbase前進を `--dist=loadfile -n 2` で再現した別commandも **5件pass**し、
cacheからのコピーを通ることを確認した。

```bash
.venv/bin/python -m pytest -n 2 --dist=worksteal -q --no-cov \
  tests/e2e/agents_ops/test_execution_basis.py \
  tests/e2e/agents_ops/test_judgement_execution.py \
  tests/unit/tasks/ball_refiner_3d \
  tests/unit/utils/physics/test_ball_fitting.py \
  tests/e2e/development/test_ci_shards.py::test_assignment_uses_cost_and_preserves_files_including_new_tests \
  tests/e2e/development/test_ci_shards.py::test_only_explicit_function_fixture_files_split_by_measured_case_cost
.venv/bin/python -m spin lint
.venv/bin/python -m mypy --follow-imports=skip \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py \
  tests/support/agents_ops/directory_seed.py
```

lintと変更Python 3ファイルのmypyはpass。不具合注入は前後とも4 commandすべてexit 1、
同じ **3 / 2 / 3 / 1件のAssertionError**で検出し、setup/collection errorなし。
順に、人間の保留の無視、projection整合確認の無視、resume時のRNGリセット、
再構成係数を1へ固定する不具合を検証した。


全CPU収集・割当は **12783件が一意**、4分割 **3389 / 3246 / 3186 / 2962件**、欠落・重複 **0/0**。新規ケースにも既存の未計測case推定値で割当がある。実収集でfixture scopeも検査した。全suiteの実行成功を表すものではない。最初の収集はjsonschema/psutil/requests不足で6 errorとなり、lock記載の版を別の収集用overlayに導入した。前後比較用の環境は変更していない。

一時pytest pluginでメモリ上に不具合を注入し、本番sourceは編集しない。
gate側は現在のpolicyを変更して実CLI subprocess内でcompileする。
学習側は実Lightning実装を変更する。不具合ごとにpytestを新規起動してseedを再作成するため、
準備済みstateが変更された実装を隠さない。正確なplugin・command・各ケースの結果・phase・
初期化/操作回数・失敗した試行は [機械可読の証拠](heavy-test-evidence.json) に記録する。

## 再現と制限

各sourceを固定し、同じ環境とevidence JSON内のobserverで
[計測runner](README.ja.md) を実行する。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m scripts.ci_metrics run --workers 2 --output /absolute/new/observation -- \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py
```

操作回数は `observer_source` を一時 `PYTHONPATH` 上の `heavy_observer.py` に取り出し、
`PYTEST_PLUGINS=heavy_observer` と新しい出力先の `HEAVY_OBSERVER_DIR` を指定して取得する。
不具合注入は `mutation_plugin_source`、`PYTEST_PLUGINS=heavy_mutation`、`HEAVY_MUTATION`、
`HEAVY_MUTATION_DIR` とevidence記載のselectorを使う。command終了までsourceを変えない。

共有ローカルWSLホストでn=1（AMD Ryzen 5 7430U with Radeon Graphics、Python 3.12、Torch 2.15.0+cpu）。cache条件は不明。共有venvにspin/pytest-covがなかったため、成功した前後計測では同じ一時tool overlayを使った。正確な版とhost負荷はevidenceに記録する。Actionsとは環境が異なるので過去runを前後比較の分母にしない。旧trainerはTorch intraop threadを2へ上書きするが、新smokeは1を明示し、native libraryの環境変数による上限は同じ。spin/pytest-cov不足の初期試行と、強いhost競合下で中断した37 passed・5 failed・129 incompleteの試行も残す。別のcProfile試行は本番の長い物理fit内で中断しており、完了した計測には含めない。

最初の軽量化後の全件観測は、並行負荷下で170 passed・CLIの90秒timeout 2件だった。この時点のfit予算はsegment 1回/joint 1回で、成果物を保存した。成功した全件再実行でもgate実装は同じで、最終smokeのfitはsegment warmup 0回/joint 1回。不具合注入、cache再利用の5ケース再現、数値回帰テストも記録に残す。レビュー済み準備回数はfixture回数からseed復元を引き、独立性テストの追加復元1回を戻して算出する。

全CPUの成功baselineや4 shardのActions比較ではない。seedは各shard/workerで再構築される。
親は統合後に分割用の時間profileを更新し、Actionsを実測する必要がある。
この変更ではworkflow・profileを編集しない。テスト・失敗検出・slow markerを除外しない。
