# タスクの出力パス

この文書を `src/tasks` の学習・生成・評価・可視化の配置規約の正本とする。
パスのroot検証は `src/utils/configuration/paths.py`、名前の組み立ては
`src/utils/configuration/output_layout.py` が所有する。

## 実験とデータの配置

| 種類 | 設定するroot | root相対の配置 |
|---|---|---|
| 学習 | `paths.output_root` | `<task>/train/<experiment>/<run-id>/` |
| 定量評価 | `paths.output_root` | `<task>/evaluate/<experiment>/<run-id>/` |
| 可視化・preview | 入口ごとのOUTPUT/ARTIFACT（下表） | `<task>/visualize/<experiment>/<run-id>/` |
| 分布・予測解析 | `paths.output_root` | `<task>/analyze/<experiment>/<run-id>/` |
| 生成処理のログ | `paths.output_root` | `<task>/generate/<recipe>/<run-id>/hydra/` |
| 特徴抽出のログ | `paths.output_root` | `<task>/precompute/<recipe>/<run-id>/hydra/` |
| BLCS/PLCS合成データ | `paths.data_root` | `<task>/<dataset-version>/` |
| 抽出モーション | `paths.data_root` | `plcs/motions/<source>/<collection-version>/` |
| 実クリップの教師データ・RGB特徴 | `paths.data_root` | `<dataset-version>/<manifestで指定したclip>/annotations/<producer>/` |

各タスクの既定 `output_root` と `artifact_root` はともに `outputs`。
README掲載用に選別した成果物だけを `assets` へ公開する。
入力データ・配布checkpoint・外部モデルのrootは用途ごとの `paths` 設定を使う。
別rootの入力を使うときは対応するrootを明示し、相対fragmentへ `data/`、
`outputs/`、`ckpt/` を重ねて書かない。絶対パスはrootに設定する。

学習runの内部構造は全タスク共通である。

```text
outputs/<task>/train/<experiment>/<run-id>/
  config.yaml                      # 解決済み設定・絶対root・固定run-id
  hydra/                           # CLIログ
  logs/version_0/                   # TensorBoard（resume時は次のversion）
    checkpoints/                   # validationで選んだckpt、last.ckpt
    qualitative/epoch_XXXX/         # validation可視化
    repro/output_dir.txt           # queue外実行のcheckpoint位置
  predictions/
    pred_test.npz
    metrics.json
    diagnostic_metrics.json
```

queue実行は予測を共有queueの `repro/<job-id>/predictions/` にも保存する。
実験登録時に `knowledge-control` が再現情報・評価を `knowledge/runs/` へ昇格する。
queue・checkpoint削除規約は training-queue スキルを参照する。
各worktreeは元repoのqueueを共有する。

## 新しい実験の設定

通常の設定は次のresolverで名前を組み立てる。

```yaml
run:
  output_dir: ${tennis_output:plcs,train,${model.name},${tennis_run_id:}}
```

`tennis_run_id` はUTC時刻とランダムsuffixからなり、一つの合成設定内で固定される。
task・用途・実験名・run-idはそれぞれ一つのpath component。`/`、空文字、
親への移動は許可しない。`tennis_dataset:<task>,<version>` は日時を加えない。
生成データは後の学習から安定して参照する必要があるため、意味のある版名で固定する。

新しい施策にはタスクの `configs/train_<experiment>.yaml` で既存設定を継承し、
変える条件と名前だけを記述する。例えば:

```yaml
defaults:
  - train
  - _self_
run:
  seed: 42
  output_dir: ${tennis_output:slcs,train,meiji_rgb_v1,s42-001}
training:
  trainer:
    max_epochs: 60
```

実験名はデータ・施策の比較単位、run-idはseed・試行の識別子とする。
異なる設定・seedの実行には新しいrun-idを割り当てる。
CLIで指定する場合も `run.output_dir=slcs/train/meiji_rgb_v1/s43-001` の同じ階層を守る。
可視化は `visualization.save`、previewは `preview.output_dir`、評価は
`evaluate.output_dir` / `evaluation.output_dir` / `run.output_dir` の各入口で同じ規約を使う。
ファイル出力にはこの階層の下に `scene.gif` 等のファイル名を付ける。
`--cfg job --resolve` で実行前に確認できる。

新規学習は既存の `config.yaml` があるrunへ書き込めない。
継続学習は `run.resume` と既存の `run.output_dir` を明示する。
別施策への重み継承は、新しいrun-idと `run.init_weights` を指定する。
staged ball trainingは `train/staged/phase1`〜`phase4` を明示的なrun-idとして使い、
前phaseのcheckpointを次phaseが参照する。別のstaged実験では4つの出力と入力参照を
同じ新しい実験名に揃える。

データ内容・split・座標契約・教師checkpoint・生成seedを変える場合は新しい
dataset-versionを作る。`tennis_scene` の実RGB build/assembleでは、同一recipe・入力hashを
検証して固定seedの出力を再利用し、完了状態を最後に記録する。旧BLCS/PLCS generatorや
各データ変換CLIに同じ再開保証があるわけではない。各入口のoverwrite/resume契約を確認する。
配布元データや過去の結果を新規runの都合で改名・移動しない。
旧成果物の利用は入力rootと旧相対パスを明示する。

## 入口ごとのroot契約

次表の `OUTPUT`、`ARTIFACT`、`DATA`、`CHECKPOINT` は、それぞれ
`paths.output_root`、`paths.artifact_root`、`paths.data_root`、`paths.checkpoint_root`。
rootを別々にしても役割は変わらない。Hydra CLIの一覧は
`src/utils/configuration/inventory.py` が所有し、CPUスモークはその一覧から入口を読む。
HydraログはすべてOUTPUT配下の対応する用途・実験・run-idに保存する。

| タスク・入口 | 成果物・生成データのrootと設定 |
|---|---|
| 全5タスク `train`、派生train、ball `train_staged`、court `train_mixed` | OUTPUT / `run.output_dir`。新規checkpointもこのrunの `logs/version_*/checkpoints` |
| ball `eval` | OUTPUT / `run.output_dir` |
| ball/court `visualize` | GIFはARTIFACT / `visualization.save`、HydraログはOUTPUT / `run.output_dir`。相対run階層は共通 |
| ball `evaluate_manifest` | OUTPUT / manifest内 `output_dir`。CLIログは `evaluate/manifest/<run-id>/hydra`、比較成果物はmanifestが独立に生成するrun-id。再開には同じmanifest出力を明示 |
| ball/court/BLCS/PLCS `preview_augmentation`、ball/court `preview_heatmaps` | OUTPUT / `preview.output_dir` |
| BLCS/PLCS `visualize`（`visualization.mode=predict`を含む） | OUTPUT / `visualization.save`。GIFとHydraログは同じmode・run-id |
| PLCS `analysis/*` | OUTPUT / `run.output_dir`（angle_velocity、dataset_distribution、loss_dominance、rotation_error_samples） |
| ball `analyze_web_bbox_ratio` | OUTPUT / `analyze.output_dir` |
| SLCS `evaluate`、`predict_clip`、`analyze_predictions` | OUTPUT / `evaluate.output_dir`、`predict.output_dir`、`analysis.output_dir` |
| `scripts.analysis.evaluate_slcs_run` | 明示的な絶対 `--output-root` / `--output slcs/evaluate/<experiment>/<run-id>`。選定receipt・条件別config/予測/metricsを保存。入力 `--training-run` も同じrootからの相対train階層 |
| `scripts.analysis.calibrate_slcs_ball_velocity` | 明示的な絶対 `--output-root` / `--output slcs/analyze/<experiment>/<run-id>` に `calibration.json`。入力 `--training-run` は同じrootからの相対train階層。既存出力は拒否 |
| `scripts.analysis.compare_slcs_ball_transitions` | 比較対象のevaluate run内へ `ball_transition_comparison.json` を絶対 `--output` で指定。既存JSONは拒否。新しいモデル推論や学習は行わない |
| BLCS/PLCS `generate_dataset` | DATA / `run.output_dir`。dataset-versionは固定、生成ログだけ独立run |
| BLCS/PLCS `generate_dataset_samples` | DATA / `samples.datasets[*].path` の `samples/`。dataset付属のGIFとmanifestであり実験runとは別 |
| BLCS API server | ディスクdatasetを作らない。サーバーログはOUTPUT / `blcs/generate/api_server/<run-id>/hydra` |
| PLCS `extract_gvhmr_motions` | DATA / `run.output_dir`（`plcs/motions/gvhmr/<collection-version>`）。抽出元もDATA、外部モデルは別のroot |
| ball `convert_web_dataset` | DATA / `convert.output_dir`。既存共有dataset `tennis/web/unified` を維持 |
| ball/court YouTube準備・annotation、ball SSL画像抽出・clip予測 | DATA配下の設定されたdataset・clip・annotation。既存データ配置を維持し、処理ログはgenerate run |
| court `generate_masks`、`generate_line_masks`、`materialize_targets` | DATA配下の派生教師・target store。line maskのpreviewはOUTPUT / `generate_line_masks.preview_dir`（同じgenerate run内の `preview/`）。`materialize_targets` のログ用途はprecompute |
| SLCS `make_splits` | DATA / `data.split_file` |
| SLCS `precompute_dino_tokens` | DATA / `data.dataset_root` 内のmanifestが示すclipの特徴ファイル。ログはprecompute run |
| tennis_scene `pipeline` | 最終NPZはOUTPUT / `output_directory` / `<output_name>.npz`。stage JSONはARTIFACT / 各stageの `output_path`。既定では同じrunの相対階層 |
| tennis_scene `reference_clip` | OUTPUT / `output_dir`。既定はgenerate/reference_clip run |
| tennis_scene `build_slcs_dataset`（broadcast profileを含む） | DATA / `dataset_output_directory` に版固定の教師・RGB特徴。生成記録はOUTPUT / `output_dir`、再利用する観測cacheはOUTPUT / `observation_directory` |
| tennis_scene `assemble_slcs_dataset` | DATA / `dataset_directory` に統合dataset・固定split・`assembly.json`。実行configとHydraログはOUTPUT / `output_dir` |
| tennis_scene `report_slcs_dataset_quality` | OUTPUT / `output_dir` に品質JSON・CSV・実行configとHydraログ（analyze run） |
| tennis_scene `visualization`、`visualize_tasks` | OUTPUT / `output`・`preview_output`、`output_directory`。入力sceneはARTIFACT |
| tennis_scene `clip_studio`、`export_clips`、`generate_dataset` | DATAのsource/dataset/clipに付随する編集・生成データ。HydraログだけOUTPUTのgenerate run |

実RGBの生成・統合・品質レポートの手順は[生成ガイド](../tennis_scene/dataset_pipeline/README.md)を参照。

入力checkpointはCHECKPOINT、既存hparamsやsceneなどの実験成果物入力は入口の
ARTIFACT契約を使う。CHECKPOINTの既定はball/court/PLCSが `outputs`、BLCSが
`ckpt`、SLCSが `checkpoints`。これは既存入力の配置を表すもので、新規学習checkpointの
保存rootではない。実RGB学習設定がcheckpoint_rootを明示する場合はその設定が優先する。
学習出力を移した後で再開・評価するときは、入力checkpoint_rootも対応する場所へ設定する。

## 非Hydraの閲覧・推論UI

非Hydra入口は引数のrootを `NonHydraPathBoundary` で検証する。Hydraのrunや
`config.yaml` は生成しない。`--outputs-root` 等の名前だけで出力先と判断せず、
checkpoint探索の入力rootとして扱う。

| 入口 | ディスクへの保存と設定 |
|---|---|
| ball/court/BLCS/PLCS `review_dataset`、PLCS `review_accad_motion` | dataset・モーションを読み取り、JSON・画像・バイナリをHTTP応答として返す。永続的な編集・可視化ファイルは作らない。入力rootは各CLIのdata/ACCAD/SMPL引数 |
| ball/court/BLCS `inference_ui`、PLCS `serve_inference_ui` | 推論要求と応答は共有repoの `.training_queue/ui_requests/<task>-<unique>/` に保持。入力は `request.json`、成功はatomicに公開する `result.bin`、失敗は `error.json`。queue実行ログは同じ共有queueの `logs/`。推論結果はHTTP応答としても返す |
| base `inference_worker` | 上記requestディレクトリ内に結果を保存。CLIのrequest引数はその境界内の既存ファイルに限定 |

UIの推論要求ファイルは障害調査用の永続IPCであり、学習runの成果物ではない。
保存場所は `inference_queue.shared_repository_root()` がgit共通ディレクトリから決め、
worktreeやUIのcheckpoint探索rootを変えても共有queueを使う。
この入口には任意の成果物保存先を選ぶ設定はなく、UI終了時にも要求・結果を削除しない。
BLCS/PLCS tracking推論のsplit補助ファイルだけは `TemporaryDirectory` 内で作成して破棄する。
UI表示を再利用可能なGIF等へ保存する場合は、上表の可視化CLIの保存設定を使う。

## CPUスモークの検証範囲

`tests/unit/utils/test_output_layout.py` は全Hydra入口のcompose・validator・path解決を確認する。
抽出モーションでは外部モデルの存在確認だけをmockする。
`tests/integration/tasks/base/test_output_artifacts_smoke.py` は全5タスクのtrain設定を合成し、
データ・モデルhookを小さなCPU fixtureに置き換えて共通runnerを1 step実行する。
実際に保存したconfig、TensorBoard、checkpoint、qualitative GIF、予測NPZとmetric JSONを
読み直し、root分離と同じrunへの新規上書き拒否も確認する。task固有モデルの本学習や
生成品質の検証ではない。非Hydraのqueue入出力は
`tests/unit/tasks/base/visualization/test_inference_queue.py` が一時ディレクトリとfixture応答で検証する。
これらは本学習・モデル精度・外部動画取得・GPU推論の成功を保証しない。
