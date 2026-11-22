# シーンのTensorRT FP16統合検証

[#2362](https://github.com/Motoki0705/tennis-lab/issues/2362)の実行用bundleです。
共有 `.venv` の本番export・buildコマンドで5モデルのエンジンを作成し、同じ入力で
本番ラッパーのPyTorch FP32とTensorRT FP16を比較します。
固定入力と許容差は[Phase 1プロトコル](../run-i2297-clipreid-tensorrt/PROTOCOL.md#精度の指標と許容差草案)を使います。

| モデル | 固定入力 | 照合 |
|---|---|---|
| CLIP-ReID | #2297の236 crop | cosine P05、カメラ間最近傍ID |
| ViTPose-H | #2303の236 crop、flipあり | UDP復号後のxy L2、confidence差と有効mask |
| HMR2 | #2303の40 crop | cosine P05、相対L2 P95、特徴の絶対差 |
| DINO | #2304の750×1333評価bucket | 画像別box対応、IoU、score差、欠落・追加 |
| GVHMR | #2305の145・177 frame各4人物 | SO(3)の回転差、betas・transl・COCO-17関節差、length・mask |

画像モデルはmin 1、opt/max 2を指定し、参照のB=1とTensorRT B=2を比較します。
GVHMRはB=1で系列長2/145/177のprofileを使います。
smokeは5モデルすべてTensorRT B=1で `meiji_4cam` の `video_000/clip_037` を処理します。
コート向きは人物から判定し、ボール検出とボール再構成を無効にします。

## 再現

[本番の導入・設定](../../../src/tennis_scene/pipeline/README.md#tensorrt-fp16)に沿って依存を導入し、
DINOとDINOv3のsubmoduleを初期化します。
このworktreeのrootから、実行タグ、先行runのoutputsを含むディレクトリ、Phase 1のDINO CUDA binaryを渡して
[共有training queue](../../../.agents/skills/training-queue/SKILL.md)へ登録します。
比較スクリプトは個々の先行runのtensor lockを確認します。

```bash
printf -v command '%q ' bash knowledge/runs/run-i2362-tensorrt-pipeline/qualify.sh \
    "$BUILD_TAG" "$PHASE1_OUTPUT_ROOT" "$TENNIS_LAB_DINO_EXTENSION"
bash .agents/skills/training-queue/scripts/training_queue.sh add "$command" \
    --name i2362-phase2-qualification --resource all --issue 2362
```

`outputs/tensorrt/qualification/<tag>/status.tsv`にモデル別のexport・build・比較と
pipeline smokeの終了状態を記録します。同じ場所に個別ログ、比較指標とsmokeの証跡を保存します。
ビルドしたONNXとplanは `outputs/tensorrt/<model>/<shape>/<tag>/` に残します。

## 初回の結果と再実行

`phase2-b2-v1` では5モデルのONNX exportが完了しました。全buildはprofile登録で終了値1になりました。
TensorRT 10.14.1の `IOptimizationProfile.set_shape` は戻り値が `None` で、
初回実装はこれを失敗と判定していました。修正後はmin/opt/maxを読み戻して照合します。
初回の出力照合とsmokeは未実施です。 [終了状態とONNXのdigest](attempts.json)

前回のONNXを再利用するには、再現コマンドの `qualify.sh` の第4引数にexportタグを渡し、
`BUILD_TAG` に今回のログ用タグを指定します。buildは既存ONNXとその外部重みのdigestを再照合します。
初回からの再実行では第4引数に `phase2-b2-v1` を指定します。
