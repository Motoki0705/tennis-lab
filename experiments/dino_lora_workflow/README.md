# Grounding DINO LoRA ワークフロー

このディレクトリは、Grounding DINO tiny を中心にした LoRA 学習用データを作るためのワークフロー一式です。

基本方針は次の通りです。

- ローカルで行うこと: YouTube 動画取得、フレーム抽出、手動アノテーション、疑似ラベル選別、`.tar.zst` アーカイブ作成。
- Colab で行うこと: Grounding DINO 推論、LoRA 学習。
- Drive との往復: ディレクトリを直接同期せず、`.tar.zst` に固めて push/pull する。
- 学習入力: 各 round は必ず `guardrail/` と `pseudo/` の2つだけを見る。
- pseudo label: court/role/ball などでフォルダを分けず、round ごとに1つの `pseudo/` にまとめる。
- guardrail: 信頼できる教師ありデータを1つにまとめ、pseudo label による学習崩壊を防ぐ。

## 標準データ配置

ローカル側の標準配置です。

```text
data/
  tennis/
  court/
  youtube/
    videos/
      av1/
      h264/
  dino_workflow/
    sources/
      youtube/
        urls.txt
        frames/
    court/
      manual100/
      manual100_kp20/
      pseudo_round000/
      pseudo_round001/
    tennis/
      ball_guardrail/
    guardrail/
      current/
    pseudo/
      round_001/
    training_sets/
      round_001/
    archives/
```

Drive 側は次を想定します。

```text
MyDrive/
  tennis_lab/
    data/
      tennis.tar.zst
      court.tar.zst
      dino_workflow/
        *.tar.zst
    outputs/
      dino/
        training/
```

`data/tennis` と `data/court` は、Colab では Drive 上の `tennis.tar.zst` / `court.tar.zst` を展開して使います。

## 全体ワークフロー

全体は4つの大きな流れに分けます。

1. CourtKP20 guardrail を確定する
2. 既存 ball LoRA adapter と court LoRA adapter を teacher として使う
3. YouTube 動画から pseudo label を作る
4. `guardrail + pseudo` で round 学習し、数 epoch ごとに pseudo label を再生成する

### 1. CourtKP20 Guardrail 確定

court は人間の選別が2回入ります。ここが一番重要です。

1回目の選別は、Court Full LoRA を作るための中間データを作る工程です。2回目の選別で採用された KP20 を、最終的な court guardrail に入れます。

```text
data/court の KP14
  |
  |  manual100 を抽出
  v
manual100 に不足5点を手動アノテーション
  |
  |  net_center は4隅から自動計算
  v
manual100 KP20
  |
  |  manual100 KP20 + KP14 guardrail で bootstrap LoRA
  v
Court Bootstrap LoRA
  |
  |  manual100 以外の court 画像へ KP20 推論
  v
Court Pseudo Round 0
  |
  |  人間が採用/不採用を2値選別 1回目
  v
selected KP20 round0
  |
  |  selected round0 を使って Court Full LoRA
  v
Court Full LoRA
  |
  |  manual100 以外の court 画像へ再度 KP20 推論
  v
Court Pseudo Round 1
  |
  |  人間が採用/不採用を2値選別 2回目
  v
最終 Court Guardrail
  = manual100 KP20 + 2回目で採用された KP20
```

この最終 Court Guardrail では、採用されなかった court 画像は学習に使いません。

### 2. Teacher の役割分担

推論 teacher は分けて考えます。

```text
Court teacher
  - court KP20 だけを推論する
  - query: configs/queries/court_kp20.txt

Role/Ball teacher
  - player, sport ball, umpire, line judge, ball boy などを推論する
  - query: configs/queries/tennis_roles.txt
  - 既存の ball FT adapter を teacher manifest に指定して使う
```

複数 teacher は `teacher_manifest` に並べます。`run_gdino_inference.py` は court 用、role 用に分けず、manifest の内容で挙動を切り替えます。

### 3. YouTube Pseudo Label 生成

YouTube はローカルで取得し、フレーム化します。重い推論は Colab で実行します。

```text
URL txt
  |
  v
ローカル: yt-dlp で video_000001.mp4 ...
  |
  v
ローカル: frame_000000.jpg ...
  |
  v
.tar.zst に pack
  |
  v
Drive へ push
  |
  v
Colab: unpack
  |
  v
Colab: court teacher + role/ball teacher で推論
  |
  v
raw_predictions.jsonl を .tar.zst に pack
  |
  v
ローカルへ pull
  |
  v
review_queue 作成
  |
  v
人間が採用/不採用を2値選別
  |
  v
selected_annotations.jsonl
```

### 4. Round 学習

各 round の学習入力は必ず次の形にします。

```text
data/dino_workflow/training_sets/round_001/
  guardrail/
    images/
    annotations.jsonl
  pseudo/
    images/
    selected_annotations.jsonl
  manifest.json
  README.md
```

学習では guardrail を常に混ぜます。pseudo label の比率や重みは `train_gdino_lora.yaml` の `pseudo_loss_weight`, `guardrail_repeat`, `pseudo_repeat` で調整します。

数 epoch ごとに次を繰り返します。

```text
train
  -> infer
  -> local select
  -> apply decisions
  -> build round dataset
  -> train next round
```

## スクリプト一覧

```text
experiments/dino_lora_workflow/scripts/
  download_youtube_videos.py
  convert_h264_and_extract_frames.py
  archive_pack.py
  archive_unpack.py
  drive_push_archive.py
  drive_pull_archive.py
  convert_points_to_gdino_boxes.py
  build_guardrail_dataset.py
  court/prepare_manual_court_kp20_seed.py
  court/annotate_missing_court_kp20.py
  train_gdino_lora.py
  run_gdino_inference.py
  build_review_queue.py
  review_binary.py
  apply_review_decisions.py
  build_round_dataset.py
```

すべて Hydra config を使います。設定は `experiments/dino_lora_workflow/configs/` にあります。

## YouTube 動画取得

URL は1行に1つずつ書きます。空行と `#` 始まりの行は無視されます。

```text
data/dino_workflow/sources/youtube/urls.txt
```

例:

```text
# tennis match videos
https://www.youtube.com/watch?v=example001
https://www.youtube.com/watch?v=example002
```

dry-run:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/download_youtube_videos.py \
  dry_run=true
```

実行:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/download_youtube_videos.py
```

デフォルト出力:

```text
data/youtube/videos/av1/
  video_000001.mp4
  video_000002.mp4
  manifest.json
```

`manifest.json` には URL、タイトル、保存ファイル、サイズ、ダウンロード時刻が記録されます。`skip_existing_urls=true` なので、同じ URL は再実行時にスキップされます。

## YouTube フレーム抽出

AV1 の元動画は H.264 に変換してからフレーム抽出します。H.264 変換済みファイルは `data/youtube/videos/h264/` にキャッシュされます。

dry-run:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py \
  dry_run=true \
  frame_stride=30
```

実行:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py \
  frame_stride=30
```

一部区間だけ抽出:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_h264_and_extract_frames.py \
  start_frame=900 \
  end_frame=4500 \
  frame_stride=15 \
  max_frames_per_video=200
```

出力:

```text
data/dino_workflow/sources/youtube/frames/
  manifest.json
  video_000001/
    frame_000000.jpg
    frame_000030.jpg
    frames.jsonl
    frames_manifest.json
```

## Archive Pack / Unpack

ローカル、Drive、Colab の間では `.tar.zst` を使います。

pack:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py \
  input_paths='[data/dino_workflow/guardrail/current]' \
  output_archive=data/dino_workflow/archives/guardrail_current.tar.zst
```

共通 base から相対パスを保って pack:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py \
  input_paths='[data/dino_workflow/sources/youtube/frames]' \
  base_dir=data/dino_workflow \
  output_archive=data/dino_workflow/archives/youtube_frames.tar.zst
```

unpack:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_unpack.py \
  input_archive=data/dino_workflow/archives/guardrail_current.tar.zst \
  output_dir=/content/guardrail
```

中身確認だけ:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_unpack.py \
  dry_run=true \
  input_archive=data/dino_workflow/archives/guardrail_current.tar.zst \
  output_dir=outputs/tmp/unpacked_guardrail
```

`archive_pack.py` は `archive_manifest.json` をアーカイブ内に入れます。`archive_unpack.py` は path traversal を防ぎ、`verify_manifest=true` ならサイズと SHA-256 を検証します。

## Drive Archive Sync

この2つはアーカイブファイルをコピーするだけです。pack/unpack は行いません。

Drive へ push:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/drive_push_archive.py \
  local_archive=data/dino_workflow/archives/guardrail_current.tar.zst \
  drive_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/guardrail_current.tar.zst
```

Drive から pull:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/drive_pull_archive.py \
  drive_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/pseudo_raw_round001.tar.zst \
  local_archive=data/dino_workflow/archives/pseudo_raw_round001.tar.zst
```

`rclone` remote も使えます。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/drive_push_archive.py \
  local_archive=data/dino_workflow/archives/guardrail_current.tar.zst \
  drive_archive=google:tennis_lab/data/dino_workflow/guardrail_current.tar.zst
```

## Point から Grounding DINO bbox へ変換

`convert_points_to_gdino_boxes.py` は point annotation を Grounding DINO 用 JSONL に変換します。

対応形式:

- `tennis_center_csv`: `data/tennis/game*/Clip*/Label.csv`
- `court_kp_json`: `data/court/data_train.json`, `data/court/data_val.json` の KP14
- `court_kp20_points_jsonl`: manual UI が出力した KP20 point JSONL

ball center CSV:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py \
  source_format=tennis_center_csv \
  input_paths='[data/tennis]' \
  output_dir=data/dino_workflow/tennis/ball_guardrail \
  box_size_px=10
```

CourtKP14:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py \
  source_format=court_kp_json \
  input_paths='[data/court/data_train.json,data/court/data_val.json]' \
  images_dir=data/court/images \
  output_dir=data/dino_workflow/court/kp14_guardrail \
  label_source=supervised
```

manual CourtKP20:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/convert_points_to_gdino_boxes.py \
  source_format=court_kp20_points_jsonl \
  input_paths='[data/dino_workflow/court/manual100_kp20/annotations_points.jsonl]' \
  output_dir=data/dino_workflow/court/manual100_kp20 \
  label_source=manual
```

出力:

```text
<output_dir>/
  images/
  annotations.jsonl
  manifest.json
```

bbox は固定サイズで、画像境界に clip されます。デフォルトは `box_size_px=12` です。

## Guardrail Dataset 作成

信頼できる annotation JSONL を1つにまとめます。court, ball, role はフォルダで分けず、各 annotation の `task`, `label`, `query` で区別します。

dry-run:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_guardrail_dataset.py \
  dry_run=true \
  input_annotation_files='[data/dino_workflow/court/manual100_kp20/annotations.jsonl,data/dino_workflow/tennis/ball_guardrail/annotations.jsonl]'
```

作成:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_guardrail_dataset.py \
  input_annotation_files='[data/dino_workflow/court/manual100_kp20/annotations.jsonl,data/dino_workflow/tennis/ball_guardrail/annotations.jsonl]' \
  output_dir=data/dino_workflow/guardrail/current
```

最終 Court Guardrail を確定する例:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_guardrail_dataset.py \
  input_annotation_files='[data/dino_workflow/court/manual100_kp20/annotations.jsonl,data/dino_workflow/court/pseudo_round001/selected_annotations.jsonl,data/dino_workflow/tennis/ball_guardrail/annotations.jsonl]' \
  output_dir=data/dino_workflow/guardrail/current
```

出力:

```text
data/dino_workflow/guardrail/current/
  images/
  annotations.jsonl
  manifest.json
```

## CourtKP20 Manual100

既存の `data/court` には KP14 が存在します。manual100 では不足分の5点だけを手動クリックします。

手動クリック対象:

- `15 left_post_base`
- `16 left_post_top`
- `17 right_post_base`
- `18 right_post_top`
- `19 center_strap_top`

`14 net_center` は手動対象ではありません。4隅 `0, 1, 2, 3` から計算します。

manual100 queue 作成:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/court/prepare_manual_court_kp20_seed.py
```

出力:

```text
data/dino_workflow/court/manual100/
  images/
  annotation_queue.jsonl
  manifest.json
  README.md
```

アノテーション UI:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/court/annotate_missing_court_kp20.py
```

GUI なし smoke check:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/court/annotate_missing_court_kp20.py \
  num_images=2 \
  dry_run=true \
  output_dir=outputs/tmp/court_manual_kp20_dry_run
```

操作:

- 左クリック: 現在の点を置いて次へ
- `Enter` / `n`: 完了画像を保存して次へ
- `u` / Backspace: 直前の点を取り消し
- `p`: 前の画像へ戻る
- `s`: skip
- `r`: 現在画像を reset
- `q` / Esc: 終了

出力:

```text
<output_dir>/annotations_points.jsonl
<output_dir>/annotation_queue.jsonl
<output_dir>/queue_manifest.json
<output_dir>/manifest.json
<output_dir>/skipped.jsonl
```

完了後、`convert_points_to_gdino_boxes.py source_format=court_kp20_points_jsonl` で bbox JSONL に変換します。

## Grounding DINO LoRA 学習

学習スクリプトはローカル smoke check と Colab 学習の両方で使います。

入力構造:

```text
guardrail/
  images/
  annotations.jsonl

pseudo/
  images/
  selected_annotations.jsonl
```

dataset だけ確認:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \
  dry_run=true \
  skip_model_load=true \
  guardrail_dir=data/dino_workflow/guardrail/current \
  pseudo_dir=data/dino_workflow/pseudo/round_001
```

Colab で学習:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \
  guardrail_dir=/content/guardrail \
  pseudo_dir=/content/pseudo \
  output_dir=/content/drive/MyDrive/tennis_lab/outputs/dino/training/round001 \
  epochs=2 \
  batch_size=1 \
  grad_accum=8 \
  save_archive=true
```

前 round の adapter から再開:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \
  guardrail_dir=/content/guardrail \
  pseudo_dir=/content/pseudo \
  resume_adapter_dir=/content/adapters/round000/adapter \
  output_dir=/content/drive/MyDrive/tennis_lab/outputs/dino/training/round001
```

`.tar.zst` の adapter から再開:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \
  guardrail_dir=/content/guardrail \
  pseudo_dir=/content/pseudo \
  resume_adapter_archive=/content/drive/MyDrive/tennis_lab/outputs/dino/training/round000/adapter.tar.zst \
  output_dir=/content/drive/MyDrive/tennis_lab/outputs/dino/training/round001
```

デフォルトでは vision backbone と text backbone を freeze し、LoRA は encoder/decoder attention の linear layer に入ります。

主な設定:

- `epochs`: epoch 数
- `pseudo_loss_weight`: pseudo label の loss 重み
- `guardrail_repeat`: guardrail の繰り返し倍率
- `pseudo_repeat`: pseudo の繰り返し倍率
- `freeze_vision_backbone`: vision backbone freeze
- `freeze_text_backbone`: text backbone freeze
- `save_archive`: adapter を `.tar.zst` として保存

## Grounding DINO 推論

`run_gdino_inference.py` は teacher manifest に従って推論します。court 専用、role 専用の別スクリプトは作りません。

teacher manifest 例:

```json
{
  "teachers": [
    {
      "name": "court_kp20",
      "task": "court_kp20",
      "base_model": "IDEA-Research/grounding-dino-tiny",
      "adapter_dir": "/content/adapters/court_full/adapter",
      "adapter_archive": null,
      "queries_file": "experiments/dino_lora_workflow/configs/queries/court_kp20.txt",
      "threshold": 0.25,
      "text_threshold": 0.20,
      "nms_threshold": 0.50,
      "nms_mode": "per_label",
      "max_detections_per_image": 80
    }
  ]
}
```

dry-run:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/run_gdino_inference.py \
  dry_run=true \
  image_dir=data/dino_workflow/sources/youtube/frames \
  teacher_manifest=experiments/dino_lora_workflow/configs/teacher_manifest.example.json
```

Colab で推論:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/run_gdino_inference.py \
  image_dir=/content/images \
  teacher_manifest=/content/manifests/court_and_roles_teachers.json \
  output_dir=/content/pseudo_raw_round001 \
  write_overlays=true
```

出力:

```text
raw_predictions.jsonl
manifest.json
overlays/
```

`raw_predictions.jsonl` は画像単位です。1画像の中に複数 teacher の予測をまとめられます。

## Review Queue 作成

Colab から raw pseudo label を pull した後、ローカルで review queue を作ります。

CourtKP20 は画像単位でまとめて選別します。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_review_queue.py \
  review_unit=image \
  raw_predictions_file=data/dino_workflow/pseudo/round_001/raw_predictions.jsonl \
  output_dir=data/dino_workflow/pseudo/round_001/review \
  allowed_tasks='[court_kp20]'
```

role/ball は bbox 単位で選別します。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_review_queue.py \
  review_unit=annotation \
  raw_predictions_file=data/dino_workflow/pseudo/round_001/raw_predictions.jsonl \
  output_dir=data/dino_workflow/pseudo/round_001/review \
  allowed_tasks='[tennis_roles]' \
  min_score=0.25
```

Colab 上の絶対パスがローカルで存在しない場合は、ローカル画像 root を渡します。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_review_queue.py \
  review_unit=annotation \
  image_search_roots='[data/dino_workflow/sources/youtube/frames]'
```

出力:

```text
review_queue.jsonl
review_assets/
manifest.json
```

## 2値選別 UI

court と role/ball は同じ UI で選別します。違いは `build_review_queue.py` で作った `review_unit` だけです。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/review_binary.py \
  review_queue_file=data/dino_workflow/pseudo/round_001/review/review_queue.jsonl \
  output_decisions_file=data/dino_workflow/pseudo/round_001/review/review_decisions.jsonl
```

操作:

- `a` / `Enter`: 採用
- `r` / Backspace: 不採用
- `?` / `m`: 保留
- `j` / 右矢印: 次へ
- `k` / 左矢印: 前へ
- `u`: 直前 decision を取り消し
- `s`: 保存
- `q` / Esc: 終了

headless smoke test:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/review_binary.py \
  headless_auto_decision=accept \
  max_items=2
```

出力:

```text
review_decisions.jsonl
review_summary.json
```

## 選別結果を pseudo dataset に変換

`review_decisions.jsonl` を `selected_annotations.jsonl` に変換します。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/apply_review_decisions.py \
  raw_predictions_file=data/dino_workflow/pseudo/round_001/raw_predictions.jsonl \
  review_queue_file=data/dino_workflow/pseudo/round_001/review/review_queue.jsonl \
  review_decisions_file=data/dino_workflow/pseudo/round_001/review/review_decisions.jsonl \
  output_dir=data/dino_workflow/pseudo/round_001
```

出力:

```text
data/dino_workflow/pseudo/round_001/
  selected_annotations.jsonl
  rejected_annotations.jsonl
  selection_manifest.json
```

`unsure` はデフォルトでは無視されます。採用に含める場合:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/apply_review_decisions.py \
  include_unsure_as_selected=true
```

## Round Dataset 作成

`build_round_dataset.py` は、1 round の学習入力を `guardrail/` と `pseudo/` の2つに整えます。

dry-run:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py \
  dry_run=true \
  guardrail_dir=data/dino_workflow/guardrail/current \
  pseudo_dir=data/dino_workflow/pseudo/round_001 \
  output_dir=data/dino_workflow/training_sets/round_001
```

作成:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py \
  guardrail_dir=data/dino_workflow/guardrail/current \
  pseudo_dir=data/dino_workflow/pseudo/round_001 \
  output_dir=data/dino_workflow/training_sets/round_001
```

出力:

```text
data/dino_workflow/training_sets/round_001/
  guardrail/
  pseudo/
  manifest.json
  README.md
```

デフォルトは symlink です。持ち運ぶ場合は copy にします。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py \
  link_mode=copy \
  output_dir=data/dino_workflow/training_sets/round_001
```

pseudo がまだない guardrail-only round:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py \
  allow_empty_pseudo=true \
  pseudo_dir=null \
  output_dir=data/dino_workflow/training_sets/round_000
```

round dataset から学習:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/train_gdino_lora.py \
  guardrail_dir=data/dino_workflow/training_sets/round_001/guardrail \
  pseudo_dir=data/dino_workflow/training_sets/round_001/pseudo
```

Colab 転送用に archive も作る:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/build_round_dataset.py \
  archive.enabled=true \
  archive.output_path=data/dino_workflow/archives/round_001_training_set.tar.zst
```

または、作成後に明示的に pack します。

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py \
  input_paths='[data/dino_workflow/training_sets/round_001]' \
  output_archive=data/dino_workflow/archives/round_001_training_set.tar.zst
```

## Colab 往復の標準手順

ローカルから Colab へ渡す場合:

```text
local directory
  -> archive_pack.py
  -> drive_push_archive.py
  -> Colab mount Drive
  -> archive_unpack.py
```

Colab からローカルへ戻す場合:

```text
Colab output directory
  -> archive_pack.py
  -> Drive へ保存
  -> drive_pull_archive.py
  -> archive_unpack.py
```

推論結果を戻す例:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/archive_pack.py \
  input_paths='[/content/pseudo_raw_round001]' \
  output_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/pseudo_raw_round001.tar.zst
```

ローカル側:

```bash
.venv/bin/python experiments/dino_lora_workflow/scripts/drive_pull_archive.py \
  drive_archive=/content/drive/MyDrive/tennis_lab/data/dino_workflow/pseudo_raw_round001.tar.zst \
  local_archive=data/dino_workflow/archives/pseudo_raw_round001.tar.zst
```

## Query Files

query は1行に1つです。空行と `#` は無視されます。

```text
experiments/dino_lora_workflow/configs/queries/
  court_kp20.txt
  tennis_roles.txt
```

court teacher は `court_kp20.txt` だけを使います。role/ball teacher は `tennis_roles.txt` を使います。

## 注意点

- すべての project command は `.venv/bin/python` で実行します。
- `data/youtube/videos/av1` は YouTube から取得した元動画、`data/youtube/videos/h264` はフレーム抽出用キャッシュです。
- pseudo label は必ず人間の2値選別を通してから学習に入れます。
- 最終学習では `data/tennis`, `data/court`, 確定済み pseudo label を guardrail として混ぜ、pseudo label だけで学習しないようにします。
- `train_gdino_lora.py` の画像参照は主に `image` / `image_path` を想定しています。外部 JSONL が `image_file` のみの場合は、変換時に `image` または `image_path` を持たせてください。
- 現在の round loop はスクリプトを組み合わせて手動で回す設計です。完全自動の multi-round controller はまだ別スクリプトにしていません。
