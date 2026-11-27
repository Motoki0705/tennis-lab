# ボール検出の単フレーム・時系列学習比較

Issue #2697。X3D/CSN＋DPTを `T1_rgb`、`T8_rgb_mdd`、`T32_rgb_mdd` で
初期化から個別に学習する。seed 42で同じ5チャネルの初期stateを生成し、T=1のstemへ
先頭RGB 3チャネルを転写する。対応する全重みの同一性はSHA-256で照合する。外部の学習済み
重みは使用しない。T=1の時刻軸は1、バッチの各要素は別フレームで、GroupNormもフレームごとに動く。
T=8/32のMDDは各子窓から作る。T=1と時系列条件の比較には、時間長とMDDの有無の両方が含まれる。

同じ凍結manifestの30,478 packから同じ順で復元抽出する。各更新で128枚の同じJPEGと教師を
4 microbatchに分ける。位置はobserved、存在の教師はobservedと確認済み不在を使う。
全条件で同じAdamWとlinear warmup＋cosine LR、Gaussian heatmapの空間平均Focal BCEを使う。
各更新の損失は全microbatchの有効フレーム数で割る。時間窓の端から教師を取り除かず、拡張は適用しない。

凍結validationの共通区間を、T=1はstride 1、T=8/32は既存のstride T/2と最近接中心の
owner規則で全件採点する。別途、[既存144アンカー](../run-i2637-ball-temporal-context/anchors.json)に
対して元と同じ中心位置（T=8で3、T=32で15、0始まり）から予測を取る。位置誤差は存在閾値適用前の
全observedを用い、元画像の幅−1、高さ−1に正規化座標を掛けてpxへ戻す。存在の閾値は
固定0.5と、全validation上の `(FPR + FNR) / 2` を最小化するものを使う。
最適バランス閾値は各条件の全validationで一度決め、各sourceとアンカーに同じ値を適用する。

## 再現

repo rootで実行する。`DATASET_ROOT` は既存配布の `dataset/` および
`prepared/{train_manifest.json,evaluation_manifest.json}` を含む絶対path。
`probe` と `train` は共有training queueの `--resource all` で実行する。
関連するmanifest、index、metadata、配布目録、アンカーのSHA-256をコードに固定している。
probeでは抽出したJPEG shard、本学習では全1,537 shardを既存の二重SHA-256で実測照合する。

```bash
run=knowledge/runs/run-i2697-ball-temporal-training
out=outputs/ball-temporal-training
bash "$run/reproduce.sh" audit --dataset-root "$DATASET_ROOT"
bash "$run/reproduce.sh" probe --dataset-root "$DATASET_ROOT" --output "$out/probe"
bash "$run/reproduce.sh" budget --probe "$out/probe/probe.json" --output "$out/plan.json"
bash "$run/reproduce.sh" train --dataset-root "$DATASET_ROOT" --output "$out/s42" --plan "$out/plan.json"
bash "$run/reproduce.sh" collect --training "$out/s42" --output "$out/record"
bash "$run/reproduce.sh" plot --table "$out/record/comparison.csv" --output "$out/record"
```

本学習が中断した場合は同じ出力とplanを指定し `train --resume` で再開する。保存済みの更新・抽出位置と
optimizer/RNGを厳密に復元する。全フレームの予測、重み、更新ログは実行出力先で保存する。
集計は固定マスク・採点ID・アンカー教師座標と予測のSHA-256を照合し、[既存比較CSV](../run-i2637-ball-temporal-context/comparison.csv)から
#2637の元閾値での数値も取り込む。図にはNoto Sans CJK JPを使用する。

## 軽量検証

```bash
OMP_NUM_THREADS=2 PYTHONPATH=.:$run .venv/bin/pytest -o addopts= --import-mode=importlib -q "$run/test_ablation.py"
```

CPUの合成入力で、初期重みとT=1のバッチ独立性、共通packの全128フレームと教師の順序、
評価窓の端およびアンカーのframe対応、位置の単位と存在判定の分母を確認する。
