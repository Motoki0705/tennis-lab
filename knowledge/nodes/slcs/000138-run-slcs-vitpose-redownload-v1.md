---
task: slcs
sequence: 138
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-vitpose-redownload-v1
type: run
title: ViTPose-H重みの固定revision再取得とatomic置換
provider: codex
date: '2026-09-19'
status: done
config:
  model: ViTPose_huge_coco_256x192
  loss: not applicable (checkpoint integrity validation)
  data: pinned camenduru/GVHMR checkpoint
metrics:
  checkpoint_bytes: 2549075546
  state_dict_tensors: 403
  nonfinite_tensors: 0
  zip_members: 405
  sha256_match: true
  atomic_replacement: true
repro:
  commit: 3bebe44d52f4a2cc2a77f2bc2dba1eea31be306e
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-vitpose-redownload-v1/repro.sh NEW_OUTPUT_DIRECTORY
artifacts:
  run_dir: knowledge/runs/run-slcs-vitpose-redownload-v1
  output_dir: outputs/tennis_scene/analyze/vitpose_redownload/s42-001
parents:
- run-slcs-meiji-v9-observation-reuse-v2
relations: []
tags:
- slcs
- vitpose
- checkpoint-integrity
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
---

## 考察 / Findings

### 要約
camenduru/GVHMRの固定revisionからViTPose-H重みを独立した新規ファイルへ再取得した。公開サイズとSHA256が一致し、ZIP CRCとCPU構造検査を通過したため本体へatomic置換した。旧ファイルは同filesystem上のハードリンクで保持した。

### アーキテクチャ詳細
GVHMR公式INSTALL.mdが指定する同名variantを、Hugging Face上のcamenduruミラーから取得した。配布元を公式ミラーとは扱わない。取得revision・URL・公開digestとHTTP情報はsource.json、配置・バックアップの絶対パスとinodeはinstallation.jsonを正とする。モデルや設定は変更していない。

### メトリクスの解釈
新ファイルを1回全体走査し、同じstreamをhashlib/OpenSSLとCPython組込み_sha256へ渡した両digestが公開SHA256に一致した。ZIP全405 memberのCRC異常なし。torch.load(weights_only=True, map_location='cpu')でmeta/state_dictを読み込み、403 tensor（637,209,875要素）と非有限値なしを確認した。patch embeddingの(1280, 3, 16, 16)、headの17関節出力も確認した。詳細はverification.jsonを参照。GPUロード・推論、実モデルへのstrict state_dictロードはこのrunの範囲外。

### アーキテクチャ⇄メトリクスの因果考察
ここで確認したのは取得ファイルの完全性と基本的な構造互換性であり、姿勢推定精度を測定したものではない。旧ファイルで報告されていた読み取り差異の原因は調査しておらず、本runの成功から原因解決は断定しない。

### 既存実験との比較
公開SHA256は既存pinと同一であるため、checkpoint pinやdataset versionの変更は不要。旧ファイルのhash再検査や成功するまでのhash繰り返しは実施しなかった。

### 次に有効な実験
親タスクで新しいreport directoryを用いた既存reuse driverを1回実行し、実際の利用時の検証結果を別runとして記録する。repro.shは新規保存先への取得とCPU検査のみを再現し、本体への再置換は行わない。
