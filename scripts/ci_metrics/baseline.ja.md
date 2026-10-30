# CPUコストbaselineとテスト契約

[English](baseline.md) · [简体中文](baseline.zh-CN.md)

#1401の計測記録とrunner構成の追試#1523をまとめ、#1402〜#1405の判断に使います。**現行sourceの全CPU観測は完了しましたが、失敗があり成功baselineではありません。**
CI目標の達成や短縮効果を実証したとは記録しません。正確なsource・対象集合は機械可読の
環境metadata、inventory、JUnit、phase eventを正本とし、失敗・cancelを成功へ置き換えません。

## 完了したローカル観測

全CPU対象 **12,764件（12,551成功、88 skip、125失敗〔setup error 8件を含む〕）** を
**5,790.692秒**で実行しました。inventory・phase・JUnitに欠測はありません。
これは**失敗した観測**であり、成功baselineではありません。pytestの117 failedと8 errorsを
計測summaryではともにfailedへ集計します。

WSL2、Python 3.11.15、Ryzen 7 9700Xの可視論理CPU 12、RAM 23.47 GiB、2 worker、
native thread各1、CUDA不可視です。他のローカル検証も動き、cache条件は不明です。
収集完了は78.673秒、最終報告・終了は25.273秒。個別process最大RSSは1,531,464 KiBで、
同時process treeのpeakではありません。wall clockに−183.694秒の補正が発生しましたが、
経過時間は単調時計で測定しています。

| File | 件数 | setup秒 | call秒 |
|---|---:|---:|---:|
| merge gate | 161 | 1630.339 | 766.868 |
| board messages | 539 | 88.313 | 877.462 |
| parent session | 97 | 194.544 | 380.229 |
| ball refiner coordinate training | 9 | 4.012 | 559.781 |
| annotation screening | 86 | 168.074 | 254.512 |

merge fixture準備と実測に基づく分割を優先します。dense/RGB契約は別の対応計測で初期化削減を
確認し、SLCS/MDD smokeは寄与が小さいため維持しました。結果の詳細は子の文書を参照します。
既存API/Fakeの失敗は #1504・#1509・#1511 で修復します。screening 1件は負荷下で15秒制限を
超え、単独再実行では成功しました。不安定性の観測として残し、元の失敗を消しません。

同じ12テストを交互に測った3組の中央値は通常 **5.992秒**、計測付き **6.243秒**（+0.251秒）です。
起動の寄与が大きく他作業も動いた部分測定で、全suiteのoverhead率へ外挿しません。
全件実行のevent書込みは4.269秒で、overheadの下限のみです。成功した全件実行の中央値・p95は
未測定です。失敗n=1と5,790.692秒を集計から除外しません。[機械証拠](baseline-evidence.json)に
出典・環境・対象識別・phase/fixture・overhead 3組を保存します。同sourceのActions比較は統合後に行います。


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
| merge gate準備 / #1402 | `tests/e2e/agents_ops/test_merge_gate.py`（`gate_env`）、`test_parent_session.py`（`parent_env`・`make_review_env`） | caseごとの使い捨てgit履歴、3PR worktree、review/CLI準備。過去部分実行の合計641.41秒 | 人間の指示と1回review消費、最新head/base/diffのcoverage、独立/依存merge順、通常レビュー、stale/merge失敗の回復、dry-runとauto-merge権限。CIは観測のみ | 現行計測確認後、fixture再利用・scenario最小化を優先。case専用の書換可能状態と実git/CLI境界は残し、速さだけで可変session状態を共有しない。 |
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

全件観測と計測上の制約は上記に記録しました。成功した同sourceのActions比較は統合後に記録します。

jobログへの転送追加後、同じ12テストの交互測定を新たに3組行い、通常中央値 **4.213秒**、計測付き **4.467秒**（+0.254秒）でした。6実行とも成功し対象は同一です。機械証拠の `overhead_live_log` に別の出典として記録します。起動の寄与が大きい部分測定で、旧測定との差を版間の短縮とは扱いません。

## runner構成の比較

初期4構成のbranch比較は同じsource・更新profile・12,795件のCPU inventoryで行いました。新しいCI回帰13件を含みます。10 shard追試ではCI回帰を3件追加（計12,798件）し、別sourceとして機械可読の証拠に記録します。プロダクト側テストは同じです。各構成n=1です。旧4 × 2参照は古いprofile・12,782件であり、同一sourceの対照実験ではありません。全体時間はActionsの `created_at`（queue込み）から `Python` の集約・後処理完了までです。runner時間はlint・集約を含む全jobの和です。失敗を含め全attemptを[機械可読の証拠](scheduling-evidence.json)に残します。

| 構成 / run | Python完了 分 | shard最短–最長 分（差 秒） | 全job runner 分 | test cache hit | 最大個別RSS GiB | 失敗件数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [4 × 2 / 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) | 26.22 | 14.42–25.80 (683) | 82.60 | 4/4 | 1.70 | 1 |
| [4 × 4 / 40992996490](https://github.com/Motoki0705/tennis-lab/actions/runs/40992996490) | 20.53 | 12.62–20.13 (451) | 66.83 | 4/4 | 1.65 | 2 |
| [6 × 4 / 40995509440](https://github.com/Motoki0705/tennis-lab/actions/runs/40995509440) | 16.22 | 11.73–15.77 (242) | 87.98 | 6/6 | 1.64 | 2 |
| [6 × 2 / 40997143981](https://github.com/Motoki0705/tennis-lab/actions/runs/40997143981) | 18.30 | 9.93–17.88 (477) | 90.33 | 6/6 | 1.60 | 1 |
| [8 × 2 / 40999125159](https://github.com/Motoki0705/tennis-lab/actions/runs/40999125159) | 15.02 | 10.20–14.62 (265) | 104.20 | 8/8 | 1.51 | 1 |

profile再生成とfixture境界の正本は[分割文書](sharding.ja.md#実測profileの再生成)です。個別process RSSはテスト子processを含まず、同時process treeのピークではありません。実測runnerはいずれも4 vCPU・約16 GB RAMですがCPU世代は異なります。完全なshard観測では全workerの終了記録が揃い、OOMやworker消失は観測されませんでした。native libraryの環境変数は1を維持していますが、既存テストの一部はPyTorch intraopを2/4へ変え、後続へ状態を残します。4 workerでは環境変数だけでCPU競合を防げません。テスト本体の修正は親へ引き継ぎます。

全runに既知のREADME.zh-CN失敗が残ります。4 workerの両観測には `test_timeout_reaps_group_when_output_is_open_or_closed[True]` の追加失敗もあります。timeout cleanup直後のassertが実行中のchildを観測したもので、無視も成功への変換もしていません。失敗観測から成功warm-cacheの中央値やp95は確定できません。source/profile/inventoryの監査は維持し、全件実行した失敗は `observations_complete=true`・`success=false` と区別します。

任意のローカル全件collect-onlyは競合負荷下で中止し、cancelを性能や成功の証拠へ含めていません。全件実行の証拠はActionsのmanifestとcoverage成果物です。親がfixture最適化をmergeした後はprofileを再生成し、選定構成を再計測してください。hardware差、標本数、thread状態の漏れが比較の制限です。
