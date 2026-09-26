# 実データの診断

通常の単体テストには含めない、実データ・固定bundleでの数値診断です。

- `coco17_placement.py`: 身体配置の数値診断。入力と使い方は[motion_alignment](../../src/tennis_scene/motion_alignment/README.md#保存済みデータでの確認)を参照。
- `component_pipeline.py`: 既定`pipeline.yaml`で構造化clipを1本処理する実clip qualification。変更する設定はroot path・device・`execution.ball_detection=load`だけ。
  ball・人物対応は[確認済みデータのimport](../../src/tennis_scene/pipeline/imports/README.md)で埋め、`evaluation.json`の`imported_nodes`に列挙する。
  storeは`--report`配下に作り、clipの`annotations/`へは書かない。import以外の全component実行（sideはimportしたballから`court_side`が決める）、scene export、全段load-only再開を検査する。
  DINO拡張はrepo rootから`build_dino_extension.sh`を実行してrun directory内にbuildし、`PYTHONPATH`に加える。GPU実行は共有training queue経由:

  ```bash
  R=/home/kamimura/projects/tennis-lab; OUT=$R/outputs/tennis_scene/evaluate/<run>
  bash tests/benchmarks/build_dino_extension.sh $R $OUT/dino_extension && \
  PYTHONPATH=.:$OUT/dino_extension/lib .venv/bin/python tests/benchmarks/component_pipeline.py \
      --repo $R --clip $R/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000 --report $OUT
  ```

  `--dataset <dataset>`を付けると、clipのstoreを`<clip>/annotations/tennis_scene`に作り、
  `generate_pseudo_annotations`で`annotation.json`を公開した後、SLCSとPLCS residualのreaderで読み戻す
  （v1 layoutのdatasetを新しい出力先へ再生成する経路）。`--seed-from <source clip>`はその前に
  clipを作る（mediaはhard link、import入力はcopy、`dataset.json`へ登録。既存clipには書かない）。
  SLCS学習に使うDINO特徴は、その後`python -m src.tasks.slcs.scripts.precompute_dino_tokens data.dataset_root=<dataset>`で作る
  （`paths.output_root`の末尾を`slcs`などtask名にすると、task出力の先頭と衝突してpath contractが停止する）。
- `court_side_clips.py`: 構造化datasetの全clipで、[court side](../../src/tasks/court_side/README.md)を検出器のballと外注注釈のball（`observed`点だけ、参照）の両方から決め、
  判定・停止理由・全仮説のscore・一致を`--report`の`<--name>.json`（既定`decisions.json`）へ書く。`observe`（GPU、共有training queue経由）は
  court検出・校正とball検出だけを`--report/stores/<clip>`へ実行し、人物・身体は無効にする。`decide`（CPU）は保存済みartifactを読むだけで、
  `--override court_side.<field>=<value>`で閾値を変えて再判定できる。

  ```bash
  R=/home/kamimura/projects/tennis-lab; OUT=$R/outputs/court_side/evaluate/meiji_clips/<run-id>
  PYTHONPATH=. .venv/bin/python tests/benchmarks/court_side_clips.py --repo $R \
      --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset --report $OUT
  ```
