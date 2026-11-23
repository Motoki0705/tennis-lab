# 担当: #2362（親 #2278 Phase 2: 5モデルのTensorRT FP16とバッチサイズ設定の本番統合）

Issue #2362 の本文（Acceptance checklist を含む）が正本。親 #2278 の合意（v4）・対象外・共通原則も守る。

## 人間の判断（2026-11-22）
- 対象は5モデルすべて（ViTPose-H、DINO、CLIP-ReID、HMR2、GVHMR）。既定は FP32 のままにする。
- TensorRT・ONNX 関連は、pyproject の任意依存として共有 `.venv` に入れる。
- DINO の MultiScaleDeformableAttention は GridSample 版を使う。プラグイン版は対象外。
- バッチサイズは設定から指定できる仕組みだけを作る。値の確定は、#2336・#2355 の計測後に後続の子で行う。

## 進め方（作業単位と時間上限）
1. **調査・設計（上限4時間）**
   - 次を読む。
     - Phase 1 の knowledge run: `run-i2297-clipreid-tensorrt`、`run-i2303`、`run-i2304`、`run-i2305-gvhmr-tensorrt`、`run-i2335-trt-batch-sweep`（`knowledge/runs/` 配下）。
     - `src/submodules` の各モデルの推論入口。
     - `src/tennis_scene` の `pipeline.yaml`・`precision` 設定・`ModelSpec`・component / publication cache identity。
   - 次の設計を決め、PR本文の最初に書く。
     - 設定の形（backend・precision・batch_size・engineの指定）。
     - エンジンの保存先と命名、識別情報。
     - identity への入れ方。FP32 の既存identityは変えない。
     - `src/utils` に置く共通部品。
   - 実装が大きく分かれる場合は、共通基盤 → 1モデル → 残り4モデルの順に、作業単位ごとにコミットする。
2. **依存と共通基盤（上限6時間）**
   - 版は `$HOME/.venvs/tennis-lab-trt` で検証済みのものに揃える。
     - `tensorrt-cu13 10.14.1.48.post1`、`onnx 1.21.0`、`onnxruntime-gpu 1.30.0`、`onnxscript 0.7.2`、`onnx-graphsurgeon 0.6.1`、`polygraphy 0.53.6`。
     - 実際に必要なものだけを入れる。
   - 共有 `.venv` に入れる前に、uv の解決結果（lock の差分）で既存パッケージの版が変わらないことを確かめる。torch などが変わる場合は止める（止めてよい条件 (c)）。
   - 共通基盤は次を満たす。
     - ONNX export と engine build は、明示コマンドで行う。
     - 識別が一致しないエンジンは読み込まずに止める（元ONNXのdigest・TRT版・GPU・精度・バッチprofile・入力形状）。
     - エンジンがない・識別が一致しない・バッチがprofile範囲外の場合は、非ゼロで止める。
3. **5モデルへの接続（上限14時間）**
   - Phase 1 の export・変換・照合のコードは、常設コードとして整理し直して使う。knowledge の実験コードを import しない。
   - GVHMR は、TensorRT で B>1 を設定すると止まるようにする（#2355 で調査中）。
4. **検証（上限6時間）**
   - CPU テスト: 設定の検証、識別の不一致ごとの停止、identity の差分、FP32 の無回帰。
   - 実GPUで確かめる。
     - 5モデルの engine build と、Phase 1 と同じ指標での TRT FP16 と PyTorch FP32 の出力照合。
     - 短いクリップで、TRT FP16 を指定した pipeline の smoke 実行。
     - 結果は新しい `knowledge/runs/<run-id>[local path]` に置く（knowledge-control skill に従う）。

全体の上限は30時間。同じ方向の試行が3回続けて失敗したら、方向を変えるか、そこまでの範囲で記録して止める。

## 作業場所・基準
- worktree は parent CLI が作る。メインworktreeは変更しない。
- `.venv`・`data`・`ckpt` はメインへのsymlink。`outputs` は永続出力で、E: にあるので `data` との hardlink は作れない。
- 共有 `.venv` を変えてよいのは、任意依存の追加とその install だけ。install は `uv sync` など、pyproject・lock と一致する方法で行い、実行したコマンドと結果をPRに書く。
- 共有 `.venv` を変える前に、メインworktreeで動いている他の作業を壊さないことを確かめる。新しい依存の追加だけで、既存の版が変わらないことが条件。
- GPU・大メモリ処理（engine build、実GPU照合、smoke 実行）は、すべて training queue 経由で行う。
  - `TRAINING_QUEUE_DIR=[local path]` を使い、worktreeのcwdから `--resource all` で add する。
  - queue worker の start・serve はしない。
  - 関連する小さい処理は1ジョブにまとめて投入してよい。各処理の終了状態は個別に記録する。
- **待機の受け渡し**
  - 投入したら、出力された `.job` 付きのIDを、最終メッセージ末尾の `PARENT_RESULT` の queue 待ちで宣言してターンを終える。
  - PR作成・push後はCI待ちを宣言する（`resume_on_success: true`、next_step: ドラフト解除と最終報告）。
  - 自分で待たない。投入の準備ができたら同じターンで投入する。
- **静かなフォールバック禁止**
  - TRT FP16 指定で、エンジンがない・使えない場合に、FP32 や PyTorch へ黙って戻らないこと。
  - バッチサイズを黙って切り詰めないこと。
  - エンジンを暗黙に作らないこと。

## 止めてよい条件
(a) Acceptanceをすべて満たした、(b) 人間の操作が要り、残る全作業がそれに依存する、(c) 任意依存の追加で既存パッケージの版が変わる、または `.venv` の変更が他の作業を壊す、(d) 親 #2278 の合意範囲の判断が要る（設定の形が既存の利用者に影響する場合を含む）。

## 返す成果
- `Closes #2362` を本文に書いた日本語PR。対象HEADの `Python` CI成功後にドラフトを解除する。
- 最終報告
  - 設計（設定の形、エンジンの保存先・命名・識別情報、identity）。
  - 依存の追加内容と install 結果。
  - 5モデルの照合結果の表（Phase 1 との比較）。
  - smoke 実行の結果。
  - 後続の子（バッチサイズの確定と全クリップE2E）で、設定値だけを変えれば済むこと。
  - 未確認事項。
