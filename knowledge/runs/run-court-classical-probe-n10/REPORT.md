# gchlebus/tennis-court-detection 古典baseline 実測レポート (/tmp作業)

## 1. 固定版とライセンス
- upstream: `gchlebus/tennis-court-detection` commit `762d077541a77abf4923f5f8f689a1410927d35e` (2019-07-16)
- license: BSD-3-Clause (`/tmp/tcd_work/tcd-src/LICENSE`, Copyright (c) 2018 Grzegorz Chlebus)
- 入力clone: `/tmp/field_align_scout/tcd1` を `/tmp/tcd_work/tcd-src` へ `git clone --no-hardlinks` し固定commitをcheckout

## 2. ビルド(再現手順)
- 定義: `/tmp/tcd_work/Dockerfile` (base `ubuntu:22.04`, apt `libopencv-dev 4.5.4+dfsg-9ubuntu4`, g++ 11.4, cmake 3.22.1)
- パッチ: `/tmp/tcd_work/patches/tcd_opencv4.patch` sha256 `12422aec27f589b92be68971e56f74ca13431decfbec7482f0f5cc45a2d8f555` (5 files, +26 -11)
  - CMakeLists: Conan廃止 → `find_package(OpenCV REQUIRED)`
  - `CV_CAP_PROP_*` → `cv::CAP_PROP_*`, `CV_WINDOW_AUTOSIZE` → `cv::WINDOW_AUTOSIZE`
  - `CV_RGB2YCrCb`(3.4.3=37) → `cv::COLOR_RGB2YCrCb`(4.5.4=37)。upstream 3.4.3 の `types_c.h` で数値一致を確認
  - `TennisCourtModel.cpp` に `#include <fstream>` 追加 (OpenCV4 のヘッダは推移的に入れないため)
  - `main.cpp` に `TCD_OVERLAY` 環境変数指定時のみ overlay PNG を保存する追加(未設定時は upstream と同一挙動、検証用)
- パッチ再現性: pristine clone に `git apply` して生成ツリーが `diff -r` で完全一致
- 実行: `bash /tmp/tcd_work/build_and_run.sh` (clone→patch→docker build→入力生成→smoke)
- image `tcd-opencv4:local` = `sha256:4938a367e9ac922267ad2415905958b0206fa6328f4a4fa0c991eefbb24d8d68`
- binary `/build/build/detect` sha256 `1a805c96389e6d337a5666b06f38b0d15bf8a73119eb9d3e0c4107a136614443`

## 3. CPU-only 証拠
- base image `ubuntu:22.04`、OpenCV は Ubuntu の CPU ビルド (`libopencv-core4.5d 4.5.4+dfsg-9ubuntu4`)
- `cv::getBuildInformation()`: `Unavailable: cudaarithm ... cudastereo` (CUDA モジュール無し)
- `ldd /build/build/detect` に CUDA/NVIDIA ランタイム無し
- コンテナ内に `/dev/nvidia*`, `/dev/dri` 無し (`out/build_evidence.txt`)
- `docker info` default runtime `runc`、`--gpus` 未使用

## 4. 入力 (すべて lossless)
| 入力 | 由来 | サイズ | AVI |
|---|---|---|---|
| real | `data/court/images/EF-hx40Q4Mg_700.png` | 1280x720 | FFV1(bgr0) 3frame |
| synth primary | B00 test `court-sample-000897` (dataset_index 897) | 959x539 | FFV1 3frame |
| synth supplementary | B00 test `court-sample-001493` (可視点最大の test sample) | 959x539 | FFV1 3frame |
- synth は canonical な `rgb.f32.npz` を `.venv/bin/python` + `src/utils/data/float32_store.py` で読み `uint8 = clip(round(f32*255))` で PNG 化 (`rgb.png` と max abs diff 0)
- AVI は `ffmpeg -loop 1 -frames:v 3 -c:v ffv1 -pix_fmt bgr0`
- OpenCV が読む中央フレームが元 PNG と画素完全一致 (max abs diff 0)、中間 index = 1

## 5. smoke 結果 (既定値、threshold 調整なし)
| ケース | exit | 点数 | 有限 | 画像内 | bbox(W/H比) | wall | user | maxRSS |
|---|---|---|---|---|---|---|---|---|
| usage (引数なし) | 255 | - | - | - | - | 0.11s | 0.07s | 67.5MB |
| real | 0 | 16 | 16 | 16 | 0.79 x 0.51 | 4.71s | 4.49s | 124.7MB |
| synth primary (000897) | 0 | 16 | 16 | 16 | 0.54 x 0.24 | 0.49s | 0.53s | 105.9MB |
| synth supplementary (001493) | 0 | 16 | 16 | 16 | 0.03 x 0.09 | 1.06s | 1.05s | 105.7MB |
- real: overlay はコート線にほぼ一致 (目視良好) `out/real_overlay.png`
- synth primary: exit 0 だが overlay は実コートと不一致の位置・形状 `out/synth_primary_overlay.png`
- synth supplementary: exit 0 だが16点が約33x48px に収束した縮退フィット `out/synth_supplementary_overlay.png`

## 6. 追加サンプルでの失敗率 (既定値のまま)
10 real + 10 B00 test を決定論的に抽出 (`scripts/run_sweep.py`, 各 run 180s 上限)。
- real 10/10 が exit 0、16点有限、bbox 0.64-0.84 x 0.49-0.61。real_01 のみ bbox が画像外へはみ出し (16点中8点が画像内)。wall 0.7-68.6s、maxRSS 124-126MB
- synth 10件: exit 0 が 6件 / exit 3 が 1件 / クラッシュ 3件
  - クラッシュ: `terminate called after throwing an instance of 'cv::Exception' ... (-215:Assertion failed) scn + 1 == m.cols in function 'perspectiveTransform'`
  - 原因: `TennisCourtFitter` がフィットを得られないまま既定 `TennisCourtModel` (空の `transformationMatrix`) を返し、`writeToFile` の `perspectiveTransform` が assert。`main` は `std::runtime_error` しか捕捉しないため異常終了 (SIGABRT 134 / SIGSEGV 139)
  - exit 0 の 6 件も目視した 2 件は誤フィット。synth_04 は bbox 2.39 x 56.77 (16点中12点が画像内) で幾何破綻
- B00 test 238 件すべて `projection.courts` が 2 (2コートが写るシーン)。古典実装は単一コート前提
- 決定性: 同一入力の再実行で出力16点ファイルがバイト一致 (`sha256 faaa170c...`)

## 7. 制約・未確認
- 失敗率は 10 件ずつの標本 (B00 test 238 件全部ではない)
- 幾何妥当性は「bbox がフレームの 10% 未満なら縮退」という事前宣言した補助基準と目視のみ。GT との点数比較は未実施
- overlay 追加分のみ upstream と差がある (未設定時は同一挙動)
- Docker client hang を 1 回観測 (container は正常終了、client が返らない)。run は名前付き + 180s タイムアウト + 1 回再試行で実施
