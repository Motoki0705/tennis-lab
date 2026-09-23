# sceneを入力とするSLCS学習準備

完成済みの構造化scene datasetを検証し、DINOv3特徴と収録単位のsplitを同じdatasetへ追加します。
scene生成と元datasetの構造は [tennis_scene](../../../tennis_scene/README.md) が所有します。
動画からの3D再推論・教師補正・dataset統合は行いません。

## 実行と入出力

```bash
.venv/bin/python -m src.tasks.slcs.scripts.generate_dataset \
  data.dataset_root=tennis_multivew/processed/meiji_3cam/dataset
```

`data.dataset_root` は必須のDATA相対パスです。`dataset.json` に登録された全clipについて、
scene、metadata、完成マーカーを検証します。未完成clipやSLCS必須配列の欠落はエラーです。
単独NPZやPLCSだけのsceneはこの入口の入力になりません。

```text
<DATA>/<data.dataset_root>/
├── dataset.json
├── splits.json                           # 新規作成または検証して再利用
└── videos/<video>/clips/<clip>/
    ├── clip.json
    ├── media/<camera>.mp4                 # 特徴を生成する場合に読む
    └── annotations/
        ├── tennis_scene/                 # 入力。内容は変更しない
        └── dino_v3/
            ├── annotation.json
            └── <camera>.npz
```

特徴仕様・品質規則・windowは共通の `data` 設定、encoderは `precompute`、
splitは `splits` を使います。既定はtrain/val/test=70/15/15、seed=0で、
同じvideoのclipが複数splitへ入ることはありません。`data.split_file` はdataset内に限定します。
教師には保存済み3D予測と既存の可視性重みを使い、既存 `label_quality` があれば適用します。
新たな幾何品質判定は追加しません。

## 再実行・失敗

既存splitの入力集合・割当・seed・比率を先に検証します。DINOはmanifest digest、camera一覧、
spec、配列、抽出間隔から計算したフレーム列の完全一致を検証します。
さらに `precompute.checkpoint_path` の内容SHA-256を完成マーカーの `generator.checkpoint_sha256`
と照合します。同じパスでも内容が変われば再利用を拒否し、同じ内容を別パスへ移した場合は再利用できます。
再利用だけの場合はencoderをロードしませんが、照合用のcheckpointファイルは必要です。
内容SHA-256の記録がない旧特徴も明示的な再生成が必要です。
必要な特徴も元動画もなければ失敗します。不一致を自動修復せず、再生成は
`precompute.overwrite=true`、split変更は `splits.overwrite=true` で明示します。

clip別の特徴生成失敗は理由を報告し、新しいsplitを公開せず非0で終了します。
完了済みの特徴は保持され、再実行時に検証して再利用できます。
明示overwriteが中断した特徴は完成マーカーを持たず、完成結果として扱いません。
元sceneの再生成はtennis_scene側で行ってください。

個別の `precompute_dino_tokens` と `make_splits` も引き続き利用できます。
単独の `make_splits` は従来どおり、既存splitへの実行をoverwrite指定なしでは拒否します。
GPUでの特徴抽出は共有training queue経由で実行します。
