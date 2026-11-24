# 再開（親 #2492 から）: CameraHMR資産がそろった

ユーザーがCameraHMRサイトで登録・ライセンスに同意し、3資産を取得して配置した（2026-11-24）。親が中身を確認した（HTMLではなく実ファイル）。

| パス | サイズ | 形式 | sha256 |
|---|---:|---|---|
| `ckpt/fasthmr/feature_extractors/cam_model_cleaned.ckpt` | 805,295,419 B | PyTorch zip | `cab4a0db880d8d822e87723bd7cd839ba98b27c043ec1d7e437ae966a7bf287e` |
| `ckpt/fasthmr/body_models/smpl_mean_params.npz` | 1,310 B | npz | `6fd6dd687800da946d0a0492383f973b92ec20f166a0b829775882868c35fcdd` |
| `ckpt/fasthmr/body_models/SMPL_NEUTRAL.pkl` | 247,186,228 B | pickle（protocol 0） | `4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688` |

`smpl_mean_params.npz` は、既存の `ckpt/body_models/gvhmr/smpl_mean_params.npz` とsha256が一致する。

## やること

1. 3資産のsha256を `assets.py` に固定し、出典とライセンス（CameraHMR：非商用研究、再配布禁止、ユーザーが同意済み）を記録する。資産はrepoにcommitしない。
2. 実重みでの PT32/PT16 推論を確認する。続けて、#2495 の固定入力manifestでの実行、TRT16の変換とparityの試行、代表6 clipの比較動画（元映像と、各モデルの同期メッシュoverlay、配置前後の3D比較）へ進む。
   - GPUはtraining queue経由で使う。
   - TRT16が変換できない、またはparityが成立しない場合は、理由を記録して実行不能とする。
3. 速度は #2496 の固定入力とharnessで測る。PR #2497（head d64962a28）にまだ固定速度入力と基準値が公開されていなければ、速度以外を先に進め、速度は「#2496 待ち」と明記する。
4. このbranchには、親が #2498 の採点器修正 head `0d4bf0990`（同IoU候補の行順で姿勢スコアが変わる問題の修正）をmergeした（b23da0b21）。テストを通してpushする。
5. 待機で終える前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終える。最終行の `PARENT_RESULT` は1行の正しいJSONにする。
