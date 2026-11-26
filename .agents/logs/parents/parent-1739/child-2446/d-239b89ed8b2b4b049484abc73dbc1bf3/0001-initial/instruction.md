# #2446 新経路の下流接続と段階2の比較（親 #1739、担当親 Joseph）

Issue #2446 の本文が仕様の正本。最初に次を読む。
- `gh issue view 2446`
- 親 #1739 の本文。特に「2026-11-24 午後の判断（ユーザー）」と評価の扱い。

## 起点

このworktreeの起点 `71e8884e` は、次の2つのHEADをmergeしたcommit。
- #2445（PR #2469、3D時間追跡と3D軌跡による選手選別。#2444 の上に積んである）
- #2459（PR #2461、段階1の比較ページ）

衝突は `src/utils/configuration/inventory.py` の境界登録だけで、両方の登録を残した。

使える出力:
- **#2445 の出力**: `[local path]`
  - `multiview_pose_tracks` v1 の clip store。
  - 読み方は `src/tennis_scene/multiview_tracking/README.md` と `knowledge/runs/run-i2445-multiview-tracking/README.md` にある。
- **段階1の比較ページ**: 親が Artifact として公開済み。
  - 生成コードは `src/tennis_scene/visualization/pose_comparison/`。
  - 出力は `[local path]`。

## 優先順位（この順に進める）

### 1. 段階2の比較ページを先に作る（CPUだけで可能）

ユーザーが定性評価を待っている。
- 段階1と同じ代表16 clip・同じ形式で作る。旧経路｜新経路の同期4視点＋コート上面図。
- 新経路側は #2445 の時間ID・選手選別まで含める。
- ページでは段階1と段階2を切り替えられるようにする。
- 表示内容:
  - 時間IDの色を固定する。
  - 選別された選手と非選手を区別する。
  - 実観測と補間を区別する。
  - 生成・消失・再出現・IDの曖昧さを示す。
- 旧経路側は `i2363-all-v1/inputs/dataset` の現在の採用scene（#2458 でボール入りに切替済み）を使う。
  - #2460 が後でダブルス幅の選手で再切替するので、再生成できるコマンドを残す。
- ボール3Dは段階1と同じく両側に出す。
- 制約は段階1と同じ:
  - 各mp4 ≤ 15 MB、全体 ≤ 200 MB、H.264。
  - 自己完結のディレクトリにする。
- 完成したら、出力ディレクトリのパスを報告する。Artifactとしての公開は親が行う。

### 2. 下流への接続と新旧比較（Issueの残り）

- 新しい recipe を作り、新旧経路を明示的に選べるようにする。
  - 新経路が成り立たないときは停止する。旧経路へ黙って切り替えない。
  - 成果物に方式の識別を保存する。
- 3D ID からカメラ別のcrop系列を組み立てる。
  - 2D Re-ID で同一性を解き直さない。
  - 人物ID・観測対応・品質maskを `body_view_selection`・`gvhmr`・`body_placement`・`scene_assembly` へ渡す。
- 既定 recipe の実行契約と結果が変わらないことを回帰確認する。
- 新旧比較:
  - Meiji 3cam はラベルで採点する。
  - meiji_4cam は coverage・停止clip数・未使用視点への再投影・処理時間・メモリで比べる。
- #2445 は Meiji 3cam の遠側の断片化（fragment 190、選別recall 61.43%）の改善を続ける予定。その結果で数値が変わる前提で、再計算できる形にする。

## GPU とメモリ

- GVHMR・HMR2特徴などGPUの段は、training queue 経由だけで実行する。
  - `TRAINING_QUEUE_DIR=[local path]` を使う。
  - workerを起動・serveしない。
- 現在、GPUロックファイル（`[local path]` と `.gate`・`.slot-0`・`.slot-1`）のACLが壊れていて、開けない。
  - GPUの段の前に `exec 9<>[local path]` で開けるか確かめる。
  - 開けなければ、CPUでできる範囲（契約・テスト・段階2ページ）を仕上げ、停止事項として報告する。
- 本日、巨大JSONを `jq` で丸ごと読み、WSLがメモリ枯渇で停止した。
  - 大きなJSON・npzは、必要な部分だけをPythonで読む。
  - 並列数は、他作業と共有するメモリ約40 GBを考えて抑える。

## 終わり方

- テストを設計してから実装する。pre-commit・pytestを通す。
- draft PRを日本語で出す。
  - baseは `main`（2つのPRに積んでいるため）。
  - 本文に、積んだ境界commit `71e8884ee1cec65a9158a29158a975a69a82f538` と依存PR #2469・#2461 を明記する。
  - 本文1行目は `Closes #2446`。
- 段階2のページだけが先にできた時点で、いったん次の3点を報告して終えてよい（親が公開して、ユーザーに見せる）。
  - draft PR
  - 出力ディレクトリ
  - ページで分かること
- 下流接続の続きは、親が `--resume` で渡す。
- 最後にPR番号と完全なHEADを宣言する。
