# #2664 担当: RF-DETR-L・RTMPose-X をシーンパイプラインの人物検出・2D pose として組み込む（既定は変えない）

あなたは親 #2492（人物モデル比較）の子 #2664 の統合担当です。まず #2664 の本文（目的・範囲・対象外・受入条件）を読み、それを正とします。

## ユーザー判断（2026-11-26、#2494 に記録済み）

- 採用方針: 人物検出は **RF-DETR-L TRT16**、2D pose は **RTMPose-X TRT16**。
- 旧経路（DINO 4-scale Swin-L・ViTPose-H）は残し、設定で選べるようにする。**検出と pose は別々に選べる**ようにし、backend も別々に指定する。
- **既定は今回は変えない。** 既定の切り替えは、#2487 で代表6クリップの新旧比較をして、ユーザーが横並び動画を確認した後に、別の PR で行う。
- 既存の成果（69クリップの再構成、公開データセット、multiview-v1）は再生成しない。
- 重み: repo の方針に従う。第三者の重みは repo・Hugging Face から再配布せず、LICENSES の「Weights obtained separately」に公式 URL を載せる。取得手順は SHA-256 で照合する（`src/submodules/smpl_assets.py`・`install_smpl_assets` の形を参考にする）。
  - RF-DETR-L は Apache 2.0。
  - RTMPose-X の重みは、商用利用・再配布の条件が上流で未確定。その旨を明記する（`knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose.md` の記述を正本に）。
  - engine は GPU ごとに作るので配布しない。

## 出発点（main にある記録・コード）

- **比較時の計測スクリプト**（前処理・復号の正本。これと数値が一致することを示す）:
  - `knowledge/runs/run-meiji-player-model-comparison/rfdetr.py` と `rfdetr.md`
    - 704×704 bilinear・ImageNet 正規化、上位300の query/class、sparse COCO category 1。
    - TRT16 は vendor の ONNX exporter と FP16 cast。plan はバッチ固定。
  - `knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose.py`・`rtmpose_convert.py`・`rtmpose.md`
    - 384×288、padding 1.25、Halpe26 の先頭17点、flip の logits 平均、SimCC 復号。
    - TRT16 はバッチ 1〜16、opt 8、workspace 1536 MiB。
- **既存の型付きモデル**: `src/submodules/models/{dino,vitpose,ecpose,fasthmr}` と `_base`、`src/submodules/README.md`。
- **パイプライン**:
  - `src/tennis_scene/pipeline/README.md`（TensorRT FP16 の節、export.json・engine.json の識別）。
  - `src/tennis_scene/configs/pipeline.yaml` の `people_models`。
  - `src/tennis_scene/configuration.py`、`pipeline/definition.py`・`orchestrator.py`・`components/`。
  - ECPose の組み込み方（`ECPOSE.md`・`components/ecpose.py`）も参考にする。ただし今回は recipe ではなく、標準経路の中で component のモデルを選ぶ形にする。
- **#1739 の入力検査とデータセット公開の検査**: `src/tennis_scene/preflight.py`・`source_identity.py`・`dataset_publication.py`。いまは DINO の設定を要求している。

## 実装の要件

1. **RF-DETR-L 検出器**
   - `src/submodules/models/` に追加する。`PersonDetectionRequest`[local path]`PersonDetectionResult` と `predict_batch` の契約は DINO と同じにする。
   - backend は PT32/PT16/TRT16。
   - `rfdetr==1.12.1` は任意の依存（extra）として `pyproject.toml` と `uv.lock` に固定する。依存が足りないときは、明確なエラーで止める。
   - 重みは `ckpt/` 配下に置き、SHA-256 を固定する。
2. **RTMPose-X の 2D pose**
   - ViTPose と同じ COCO17 の契約にする（動画＋box、明示した frame 列、`predict_crops` 等で、下流が ViTPose と同じ呼び方で使えるように）。
   - 実行時に MMPose・mmcv・xtcocotools を要求しない。CSPNeXt と RTMCC head の必要部分を `src/submodules/vendor/` などに取り込む。その際、Apache-2.0 の LICENSE・NOTICE・origin（固定 commit）を付け、`LICENSES.*` の vendor 表にも足す。
   - checkpoint は strict に読み込む。
   - 取り込んだ実装が公式 MMPose（比較時と同じ固定 commit）と一致することを、CPU で実画像の box について確かめ、記録する。
3. **TensorRT**
   - 両モデルの export・build・実行を、シーン共通の TensorRT の仕組み（`export.json`・`engine.json` の照合、profile の範囲外はエラー）に加える。
   - 作成コマンドを README に書く。
4. **設定**
   - `people_models` で、検出（`dino` / `rfdetr-l`）と pose（`vitpose-h` / `rtmpose-x`）を個別に選べるようにする。
   - 各モデルの backend・precision・batch_size・engines を別々に指定できるようにする。
   - **既定値は DINO＋ViTPose の現行どおり**にする。DINO＋ViTPose を選んだときの component・publication の識別と出力が、変更前と一致することを回帰テストで示す。識別が変わるなら、既存の cache 互換の仕組みに従って理由を説明する。
   - モデル・backend・重み・engine の違いを成果の識別に含める。重み・engine・依存がないときは止める（暗黙の fallback 禁止）。
5. **score と confidence の棚卸し**
   - 検出 score と関節 confidence を読む下流をすべて洗い出し、表にする。対象は、重複 box の統合、追跡（StrongSORT++ の pose・CLIP の手がかり）、選手選別、GVHMR への入力、三角測量の重み、観測 mask、可視化など。
   - 各箇所について、閾値、値の意味、新しいモデルでの値の範囲（比較時の記録から分かる範囲）を書く。
   - RF-DETR の confidence の初期値（DINO は 0.3、比較時の cutoff は 0.25）や、confidence を換算するかどうかは、根拠を添えた案として親に返す（`needs_parent`）。**勝手に決めて確定しない。** 換算や補正を入れる場合は、明示的な設定にする。
6. **入力検査**
   - #1739 の入力検査とデータセット公開の検査を、新しい組み合わせ（および混ぜた組み合わせ）を正しく識別して受け付けるように広げる。
   - 想定外の設定は、引き続き拒否する。
7. **テスト**
   - `tests/README.md` の方針に従い、先にテストを設計する。単体テストは CPU のみで、重みファイルを使わない。
   - 対象: 契約、関節の対応、復号、設定の選択、識別の違い、engine・重み・依存がないときのエラー、既定経路の回帰。
8. **GPU での確認**（必ず training queue の `all` 予約で、`.agents/skills/training-queue/SKILL.md` に従う）
   - 本番のアダプタと比較時の計測スクリプトについて、固定入力（#2496 の速度用入力、または #2495 の bundle の一部）で、PT16・TRT16 の出力（box・score・関節）を照合する。一致の基準は事前に決めて記録する。
   - RF-DETR-L TRT16 + RTMPose-X TRT16 を選んで、代表クリップ1本（4カメラ）をパイプラインに通す。各段が成果を出すこと、または停止の理由を記録する。比較の評価は #2487 で行うので、ここでは通るかどうかの確認でよい。
9. **文書**
   - `src/submodules/README.md`、パイプラインの README（TensorRT の表と設定例）。
   - `LICENSES.md`・`LICENSES.ja.md`・`LICENSES.zh-CN.md` の「別に取得する重み」と vendor 表。
   - 取得手順（公式 URL から取得し、SHA-256 で照合する CLI）。重みは公式 URL から取得し、git には入れない。
   - 同じ事柄を2か所に書かない。

## 進め方

- worktree の中で作業する。メインの worktree のファイルは変更しない。
- Python は `.venv/bin/python`。commit 時の pre-commit（ruff・mypy）を通す。
- **途中で親に返すとき**（`needs_parent`）:
  - 第5項の confidence の案が出たとき。
  - 設計上の分岐で判断が要るとき（例: 識別が変わってしまう、依存の固定で競合する）。
  - どちらも、棚卸し表と案を最終回答に書く。
- PR は日本語で書く。本文に `Closes #2664` と `Refs #2487` を入れ、実行証拠（HEAD、テスト、GPU 照合の記録）を書く。push と PR 作成まではしてよい。正式レビューとマージは親が行う。
- **禁止事項**:
  - 既定の切り替え、既存成果の再生成、multiview-v1 の変更、Meiji 3cam データの使用。
  - 認証・ライセンスの回避、重みの無断での差し替え、GPU の直接起動。
  - Issue・PR に、ローカル絶対パス・セッション ID・秘密情報を書くこと。
- 最終回答は日本語で、何を変えたか、検証結果、残課題、PR の URL を書く。最後に `PARENT_RESULT {"schema_version":1,"state":"<done|needs_parent>","wait":null}` を付ける。
