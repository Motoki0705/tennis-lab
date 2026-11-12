# Court全6ペア・共通4sceneの最終比較

#1586 / 親#1581の記録。追加の学習・推論・評価は行わず、完了済み9評価をCPUで集計した。
結果と制約の正本は[group node](../../nodes/court_detection/000067-group-court-scene-pair-s42.md)。
実行と復元・停止の来歴は[execution run](../../nodes/court_detection/000056-run-court-scene-pair-execution-s42.md)。

## 入力と再現

raw評価JSONはDrive root相対の
`outputs/court_detection/evaluate/scene-pair-5500-s384-colab/<condition>-s42/evaluation.json`。
conditionは`scene-1, scene-2, scene-3, scene-4, P02, P03, P12, P13, P23`。
約574KB/件なのでGitへ複製しない。必要な9ファイルだけtennis-drive CLIの`download --verify`で取得し、
外部directoryの`<condition>.json`へ保存する。元のDrive markerの内容hashは[sources.json](sources.json)に固定した。
P01はscene-2の別名で、一度だけ集計する。

```bash
export TENNIS_REPO="$PWD"
export PAIR_EVALUATIONS=/absolute/path/outside/repo/evaluations
export PAIR_COMPARISON_OUTPUT=/absolute/path/outside/repo/comparison
PYTHONDONTWRITEBYTECODE=1 bash knowledge/runs/run-court-scene-pair-comparison-s42/repro.sh
```

`compare.py`はraw bytes、checkpoint/33160更新、config/data digest、共通manifest、BS/seed、
IDの順序・一意性・件数、scene別count・macroと有限指標を検査してから出力する。
Python3.11で作られた元macroは加算順も明示して完全一致を確認する。
出力の単純平均は`math.fsum`、独立検証の`crosscheck.py`は`compare.py`をimportせず
`Fraction`の厳密なbinary値から計算する。raw指標、群contrast、scene平均、全CSV/JSONセルが完全一致した。

sourceの小さい抽出を再作成する場合は、Driveの閉じたcampaign状態
`outputs/court_detection/campaign/scene-pair-5500-s384-colab`を外部directoryへ取得してから実行する。
この記録では親が内容照合済みで取得した85ファイルを読み取り専用で再利用した。

```bash
.venv/bin/python knowledge/runs/run-court-scene-pair-comparison-s42/capture_sources.py \
  --campaign "$PAIR_CAMPAIGN" --evaluations "$PAIR_EVALUATIONS" \
  --output "$PAIR_COMPARISON_OUTPUT/sources.json"
```

元のinputs identity・完了印・評価hash・train/eval checkpoint/dataの一致と、旧4条件の実画像5指標・
IDの完全一致を要求する。元manifest/過去のnode/runは編集しない。
原Colabが行ったcheckpoint payloadの検証と、このCPU作業のreport再照合は区別する。

## 出力

- [comparison.json](comparison.json): 9条件実画像、6ペア、δ・加法性、B00追加表、5指標の9×4行列と単純平均。
- [real_pairs.csv](real_pairs.csv)、[contrasts.csv](contrasts.csv)、[b00_addition.csv](b00_addition.csv)。
- [synthetic_long.csv](synthetic_long.csv): 36セルの全5指標・件数・学習区分。
- 指標別の`synthetic-*.csv`: 9×4値と非加重4scene平均。
- [synthetic-kp-median.png](synthetic-kp-median.png): KP中央値の対数色scale、実数値とS/G/N区分。図の数値は表示用に丸める。
- [crosscheck.json](crosscheck.json): rawからの独立再計算とCSV/JSON全セルの照合結果。
- [curve-sources.json](curve-sources.json): 選択取得したTensorBoard8ファイルと採用更新・除外末尾の記録。

S＝学習済みscene、G＝未学習scene・同群あり、N＝未学習scene・同群なし。
B00/B01＝オムニ、B02/B03＝クレー。群分けは親issueの合意、sceneの所属は元manifest来歴で確認した。
4sceneの平均は同じ重みであり、2,439画像のpooled値ではない。
実画像のKP中央値contrastは条件中央値の差で、sample単位のpaired推論ではない。

## 採用された学習履歴

`export_curves.py`は新5ペアのTensorBoard event8個を読み、source/restartの保存境界で
旧attemptの未採用末尾を除く。P03/P13/P23は19,896/6,632/29,844更新で切断し、
TensorBoardの0始まりstepを完了optimizer更新（step+1）へ変換する。
旧tailの実最終更新を推定せず、元restartのnullを維持する。
既存4条件の曲線は過去のrunに保持する。

イベントは[curve-sources.json](curve-sources.json)に示した8 pathだけを`download --verify`で取得する。
新しいcanonical directoryに採用scalarを作り、knowledge共通の`kg_curves.py`で各runの図を生成する。
イベントをGitへ保存しない。既存canonical directoryへの混合を拒否する。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  knowledge/runs/run-court-scene-pair-comparison-s42/export_curves.py \
  --events "$PAIR_EVENT_ROOT" --canonical-root "$PWD/.cache/court-scene-pair-fit-curves"
```

fit validationは合成500＋実画像2211のaggregate。最終domain別評価や共通2439評価と混同しない。
曲線の切断は連続実行とのsampler一致を保証するものではない。

## テスト設計と実行

数値の期待値は非対称の手計算（δ=24、加法性=4、非加重scene平均=6）を使う。
checkpoint、step、manifest、config/data、ID順序/重複、scene順序/件数、macro、非有限値、batchの不一致を拒否する。
既存のcampaignテストは原validatorのID順序と非加重平均のguardを再利用する。
別テストで再開時の未採用tail除外と不正な履歴を検査する。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -n 0 \
  knowledge/runs/run-court-scene-pair-comparison-s42/test_compare.py \
  knowledge/runs/run-court-scene-pair-comparison-s42/test_curves.py \
  knowledge/runs/run-court-scene-pair-colab-s42/test_campaign.py::test_evaluation_adapter_checks_pinned_order_and_unweighted_mean \
  --durations=0 --durations-min=0
```

集計scriptと証跡を新しいDrive documentation subtreeへ内容照合付きで保存する。
Gitが記録の正本で、rawモデル・入力・予測・eventや既存の科学出力は移動・削除しない。
