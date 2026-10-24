# CPUコストbaselineとテスト契約

[English](baseline.md) · [简体中文](baseline.zh-CN.md)

#1401の計測記録で、#1402〜#1405の判断に使います。**現行sourceの全CPU計測は未完了です。**
CI目標の達成や短縮効果を実証したとは記録しません。正確なsource・対象集合は機械可読の
環境metadata、inventory、JUnit、phase eventを正本とし、失敗・cancelを成功へ置き換えません。

## Actionsの参考観測

[ci_attempts.json](../../tests/support/development/ci_attempts.json) は実際の再実行のfixtureです。
attemptごとのREST時刻・状態・run/head識別子と出典endpointを保持しています。

| Run | Attempt | 結果 | Apt | Test | 全jobのrunner時間 | 観測したPython完了時間 |
|---|---:|---|---:|---:|---:|---:|
| [39767861734](https://github.com/Motoki0705/tennis-lab/actions/runs/39767861734) | 1 | Cancelled | 1480秒 | Skip（0秒） | 1518秒 | 初回作成から1581秒 |
| 同じrun | 2 | Success | 25秒 | 2966秒 | 3027秒 | 再実行開始の代理値から3030秒 |

両attemptは合計**4545 runner秒**を消費しました。成功した再実行だけを数えると1518秒を
捨ててしまいます。正確な再実行要求時刻とcache条件は残っておらず、再実行の起点には
APIの開始時刻を代理値と明記して用います。成功標本はn=1、中央値・nearest-rank p95は
3030秒、p95は参考値です。改善前後の比較ではありません。

親が保存した直近mainの23 push runは、異なるrevisionで成功1、失敗1、cancel21です。
保存済みの直近成功 [40047681315](https://github.com/Motoki0705/tennis-lab/actions/runs/40047681315)
はTest 3338秒、Python runner時間3396秒です。保存済みのmain cancel
[40063537455](https://github.com/Motoki0705/tennis-lab/actions/runs/40063537455) はTest中に
中断しています。23件すべてのjob/step情報は揃っていません。この混合標本から全attempt
runner累計、warm-cache中央値、信頼できるp95は主張しません。

#1401記載の過去ローカル部分実行は14分28秒、3786 passed / 1 skipped、終了値2で中断
しています。merge gateのtestcase時間合計641.41秒は候補発見の手掛かりであり、完全な
baselineでも、現行sourceやActions runnerの所要時間の推定でもありません。

## 候補の契約と採否基準

以下のfile/functionで候補を固定します。計測summaryには件数とsetup/call/teardown寄与を
記録します。fixtureはsetupの内数です。test内でのモデル/CLI構築はcallに含まれます。
worker間の寄与時間は重なり得るため、そのままjob実時間の削減幅ではありません。

| 候補・後続 | 現行テスト | 重複契約・想定コスト | 残す境界 | 採否と不確実性 |
|---|---|---|---|---|
| merge gate準備 / #1402 | `tests/e2e/agents_ops/test_merge_gate.py`（`gate_env`）、`test_parent_session.py`（`parent_env`・`make_review_env`） | caseごとの使い捨てgit履歴、3PR worktree、review/CLI準備。過去部分実行の合計641.41秒 | 人間の指示と1回review消費、最新head/base/diffのcoverage、独立/依存merge順、共有基盤承認、stale/merge失敗の回復、dry-runとauto-merge権限。CIは観測のみ | 現行計測確認後、fixture再利用・scenario最小化を優先。case専用の書換可能状態と実git/CLI境界は残し、速さだけで可変session状態を共有しない。 |
| dense mask直積 / #1403 | `test_block.py::test_dense_boundary_rejects_noncanonical_full_masks`（30組） | 異常5分類×MHA/GQA×module/direct/update | bool dtype、厳密rank/shape、batch/query broadcast拒否、device不一致。各public入口のvalidator接続、方式固有経路、mask/state排他 | 共通validatorの異常分類＋代表入口接続への整理候補。限定的な不具合注入の根拠を必要とし、無条件の30→5は採用しない。 |
| direct/module一致 / #1403 | `test_block.py::test_direct_forward_matches_module_call_for_valid_arguments`（3）、`...values_gradients_and_state`（3） | 出力一致の反復 | 厳密出力、全入力/parameter勾配、state。MHA/GQA、CSWAのall-validと部分invalid分岐 | CSWAの固有分岐を対応づけてから統合。assertが似ているだけでは包含とみなさない。 |
| RGB predictor直積 / #1403 | `tests/unit/tasks/ball_detection/model_io/test_predict.py`（24組） | channel2設定×実model2種類×RGB異常6分類。forward patch前のモデル割当 | frame/channel/layout/dtype/spatial/empty拒否、DPT/DeepQueryとchannels2/5の接続、forward未到達、実型dispatch | 全異常を共通validatorへ、実入口の接続を代表例へ整理する候補。`isinstance`分岐を消すgeneric Mockは不可。 |
| SLCS smoke / #1404 | `tests/integration/tasks/slcs/test_training_smoke.py::test_dataset_model_loss_backward_smoke`（4組） | frames6/18×context off/on。残window学習と4mode評価の反復 | padded/short/full window、adapter→model→loss/backward、context勾配接続、optimizer更新、checkpoint再読込、full/no_rgb/detector_gap/rgb_onlyの保存target/mask/ID/frame | 寄与が有意なら契約を分け反復stepを最小化。unitの勾配検査だけでintegrationのcontext接続を代替しない。 |
| 通常fit/callback / #1404 | `...test_real_rgb_profile_normal_fit_saves_monitored_checkpoint` | 本番trainer/logger/callback初期化 | 実監視metric、通常fit、checkpoint callback、実TensorBoard event | 実際の通常fit smokeを維持。`fast_dev_run`や手動optimizerだけでは代替しない。 |
| MDD module構造 / #1404 | `tests/unit/tasks/ball_detection/models/mdd/test_cnn.py` の本番config構造検査 | Conv2d/Conv3d数とSwiGLU幅のための全weight割当 | 実際に構成したmodule treeと幅/数の不変条件 | 有意なコストがあり初期化が対応する場合だけ割当削減を検討。config値だけのassertへ変えず、非対応meta初期化は棄却。 |
| repo静的走査 / 後続候補 | `tests/e2e/development/test_architecture_boundaries.py` | file探索・AST・参照解析の反復 | 新規file、削除module、動的/CLI入口、transitive forward helper、負例によるpolicy検査 | 検証を維持。計測が根拠を示す場合に不変なworker内走査/index共有を検討し、件数だけで削除しない。 |

`test_block.py` の完全pathは `tests/unit/utils/models/components/test_block.py` です。
表は契約の候補であり削除承認一覧ではありません。overlay/installer/DSHの未計測直積は
見送ります。#1401ではテスト、marker、本番model、validatorを変更しません。

## 固定する比較手順

1. job構成比較では既定CPU command、source、全inventory、依存版、runner OS/image/CPU/RAM、
   worker/thread、cache、競合負荷を固定し、cancel・準備失敗を含む全attemptを比較します。
2. テスト統合でinventoryが変わる場合は、維持先の契約と代表的な不具合検出に加え、件数、
   fixture/CLI/model初期化回数、phase時間を比較します。層の移動を件数削減と呼びません。
3. n、成功の中央値/nearest-rank p95、失敗/cancel件数、全job runner累計を併記します。
   小標本p95は参考値、cacheや準備/後処理欠測があればwarm-cacheとは主張しません。
4. 固定したsourceで [README.ja.md](README.ja.md) の全CPU経路を実行します。部分計測は
   優先順位用であり、全体の同等性や50%・約15分の目標達成を認定しません。

全件完了、ローカル競合、計測overhead、現行sourceの候補別寄与は、#1401の受入前に
この記録へ追加する必要があります。
