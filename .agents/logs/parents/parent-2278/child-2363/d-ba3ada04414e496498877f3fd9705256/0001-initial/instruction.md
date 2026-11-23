# 担当: #2363（親 #2278 Phase 2: バッチサイズの確定、CPUボトルネック記録、全クリップの TRT FP16 E2E）

Issue #2363 の本文（2026-11-22 に人間の判断で改訂。Acceptance checklist を含む）が正本。親 #2278 の合意（v5）・対象外・共通原則も守る。

## 人間の判断（2026-11-22）
- E2E は TRT FP16 の1条件だけを実行する。PyTorch FP32 の全クリップ実行と、FP32 との出力比較は行わない。
- パイプライン全体の CPU ボトルネックを特定したいので、そのための記録を残す。改善の実装は対象外（特定と候補の提示まで）。
- 進める順序は「記録の実装 → E2E 検証」。

## 前提（main に入っている成果）
- #2362（PR #2379）: 5モデルの TRT FP16 とバッチサイズ設定。`src/tennis_scene/pipeline/README.md` の TensorRT FP16 の節が仕様。
- #2336（PR #2373）: 画像系モデルのバッチ計測。knowledge node `000080-run-i2336-vision-batch-sweep`。
- #2355（PR #2380）: GVHMR のバッチ化の調査。knowledge node `000082-run-i2355-gvhmr-pred-cam-clamp` と `knowledge/runs/run-i2355-gvhmr-pred-cam-clamp/`。
  - B>1 のエンジン作成は、attention の mask を加算形（`scores + Where(mask, -32768, 0)`）にすると成功する。`-inf` を使う加算形は L=177 で非有限値が出るので使わない。
  - knowledge の実験コードは import しない。必要な部分は常設コードとして整理し直す。

## 進め方（作業単位ごとにコミットする）
1. **調査・設計（上限3時間）**
   - 次を読む。
     - `src/utils/inference_recording.py` と `.md`
     - `src/tennis_scene/pipeline` の `video_batch.py`・`runner.py`・`batching.py`・`inference_runtime.py`
     - 動画の prefetch
     - #2362 の TRT 実装
   - 次の設計を、PR本文の最初に書く。
     - CPU 記録の形式・集計
     - GVHMR の加算 mask の組み込み方
     - バッチサイズの設定値
2. **GVHMR の B>1 有効化（上限6時間）**
   - ONNX export 後のグラフの mask を加算形（-32768）に書き換える処理を、#2362 の export・engine 作成の経路に常設で入れる。
   - 書換え前とのビット一致を、使う全 shape で ONNX Runtime で確かめる。
   - 層精度検査を、加算 mask 付きの fused MHA と、K=2 の `learned_pos_linear` を扱えるように拡張する。
   - 境界（L=177 など）で出力の有限性を確かめる検査を残す。
   - そのうえで、「GVHMR は TRT で B=1 のみ」の制限を外す。
3. **CPU ボトルネックの記録（上限10時間）**: Issue 本文「具体的な変更内容」の 3 をすべて満たす。
   - span と node ごとに、thread CPU 時間と process CPU 時間を記録する。
   - 一定間隔の資源サンプリングを、時刻つきで記録する。対象は次のとおり。間隔は設定で指定し、既定値と負荷を README に書く。
     - process 全体と、役割名を付けた thread ごとの CPU 使用率
     - コアごとの CPU 使用率
     - RSS
     - NVML の GPU 使用率と使用メモリ
   - 次の待ちと処理を、それぞれ span として区別する。
     - decode prefetch での待ち
     - CPU の前処理・後処理
     - GPU への転送
     - CUDA の同期
   - 集計を出力する。
     - stage・node・モデル別の wall 時間、CPU 時間、GPU が使われていた割合
     - GPU が遊んでいた区間に実行されていた CPU 処理の順位表
   - 記録を有効にしても、成果物のバイト列・artifact ID・execution key が変わらないことをテストする。
   - `src/utils` は下流への影響が大きいので、テストは設計してから書く（tests/README.md）。
4. **バッチサイズの確定（上限4時間）**
   - 初回の候補は次のとおり。根拠表（throughput・VRAM・単体精度）と最終値を記録する。
     - ViTPose-H B=8（反転前の crop 数）
     - HMR2 B=16
     - DINO GridSample B=1
     - CLIP-ReID B=16
     - GVHMR B=4
   - #2362 で検証した engine の範囲を超えるバッチは、明示コマンドで engine を作り直し、識別情報を記録する。
   - 代表クリップで、パイプライン全体が VRAM 16GB・WSL メモリ 40GB に収まることを確かめる。
5. **E2E（上限12時間。実行待ちは含まない）**
   - 対象: meiji_4cam の全クリップ（video_000 の 69 clip）。
   - 処理: video 一括モード、横断バッチ有効、5モデルすべて TRT FP16。プレーヤーのみ（Ball 無効）、Court は固定条件。
   - 先に代表クリップで、CUDA 同期つき計時が全体の時間をどれだけ変えるかを、同期あり・なしで比べて記録する。
   - 全クリップを1ジョブで実行する。出力先は worktree の outputs（E: 側）にする。
   - 失敗したクリップがあれば、その根拠を記録する。黙って除外しない。
6. **分析と記録（上限5時間）**
   - Issue 本文「具体的な変更内容」の 5 をすべて満たす。
   - 代表クリップと外れ値クリップの可視化動画を作る。
   - knowledge-control skill に従って、新しい `knowledge/runs/<run-id>[local path]` と node を作る。
     - node の番号は、作成時点の main の `.sequence` を見て割り当てる。他の PR と番号が重なったら振り直す。

全体の上限は40時間（実行待ちは含まない）。同じ方向の試行が3回続けて失敗したら、方向を変えるか、そこまでの範囲で記録して止める。

## 作業場所・基準
- worktree は parent CLI が作る。メインworktreeは変更しない。
- `.venv`・`data`・`ckpt` はメインへのsymlink。`outputs` は E: 側の永続出力。
- 共有 `.venv` の依存は変えない。サンプリングプロファイラなどの新しい依存が必要になったら、止めて親に聞く。
- GPU・大メモリ処理は、すべて training queue 経由で行う。対象は engine の作成、実GPUの照合、代表クリップの実行、全クリップの E2E。
  - `TRAINING_QUEUE_DIR=[local path]` を使い、worktree の cwd から `--resource all` で add する。
  - queue worker の start・serve はしない。
- **待機の受け渡し**
  - queue に投入したら、出力された `.job` 付きの ID を、最終メッセージ末尾の `PARENT_RESULT` で queue 待ちとして宣言し、ターンを終える。
  - PR の作成・push 後は CI 待ちを宣言する（`resume_on_success: true`、next_step は「ドラフト解除と最終報告」）。
  - 自分で待たない。
- **静かなフォールバックの禁止**
  - engine がない、識別が合わない、バッチが profile の範囲外のときに、PyTorch・FP32・小さいバッチへ黙って戻さない。
  - 計測の記録で失敗したときも、黙って記録を欠かさない。

## 止めてよい条件
(a) Acceptance をすべて満たした。
(b) 人間の操作が要り、残る全作業がそれに依存する。
(c) 共有 `.venv` に新しい依存が必要になった。
(d) 親 #2278 の合意範囲の判断が要る。

## 返す成果
- `Closes #2363` を本文に書いた日本語の PR。対象 HEAD の `Python` CI が成功したら、ドラフトを解除する。
- 最終報告
  - 設計（CPU 記録の形式・集計、GVHMR の加算 mask の組み込み）
  - 確定したバッチサイズと根拠表
  - E2E の結果
    - 完走したクリップ数と状態
    - 総時間、モデル別・stage 別の内訳
    - Peak VRAM・RSS
  - CPU ボトルネックの順位表と、改善の候補
  - 品質の診断（mask・追跡・人物対応）と外れ値の原因
  - 可視化動画のパス
  - 未確認事項
