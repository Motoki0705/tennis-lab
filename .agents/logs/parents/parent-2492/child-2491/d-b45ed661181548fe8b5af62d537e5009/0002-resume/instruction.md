# 再開（親 #2492 から）: CameraHMR資産の扱いの決定

ユーザーが **CameraHMRサイトで登録・ライセンス同意し、3資産を自身で取得する** と決めた（2026-11-24、親 #2492 本文の「ユーザー判断」に記録済み）。ユーザーは次の3ファイルを親が用意したスクリプトで取得し、次の場所に置く。

- `ckpt/fasthmr/feature_extractors/cam_model_cleaned.ckpt`
- `ckpt/fasthmr/body_models/smpl_mean_params.npz`
- `ckpt/fasthmr/body_models/SMPL_NEUTRAL.pkl`

`ckpt/` はメインworktreeの `ckpt/` へのsymlinkなので、このworktreeからも同じパスで見える。HMR 2.0ベースの構成は、公開コードでMask-ToMeを再現できないため実行不能と記録する（ユーザー了承済み）。

## やること

1. 上の3ファイルが無ければ、取得不要な公開資産を先に用意する。
   - 対象: FastHMRの固定revisionのcheckpoint（`camerahmr.pth.tr`、`camerahmr_diffdecoder.pth.tr`、`vae.pth.tr`）、`yolo11x-seg.pt`、`J_regressor_h36m.npy`。
   - 置き場所は `ckpt/fasthmr/` の所定の位置。
   - それぞれのsha256・取得元・ライセンスを記録する。YOLO11はAGPL-3.0なので、評価利用と本番統合の可否を分けて記録する。
   - 公式Python依存の導入も済ませる。
2. 3ファイルがそろっていれば（ユーザーが先に置いた場合）、sha256を `assets.py` に固定し、実重みでのPT32/PT16推論の確認、#2496 harnessによる速度計測、parity、代表6 clipの比較動画へ進む。GPUはtraining queue経由で使う。
3. 3ファイルがまだ無ければ、1を終えてcommit・pushし、`needs_parent` で「CameraHMR資産待ち」と返す。親がユーザーの取得を確認してから再開する。資産を別の入手先から取ったり、代わりのものを使ったりしない。
4. #2496 の固定速度入力・基準値がPR #2497 にまだ無い場合、速度はその公開を待つ。速度だけ独自の入力で測らない。
5. 終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してpushする。最終行の `PARENT_RESULT` は1行の正しいJSONにする。
