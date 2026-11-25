FastHMRのCameraHMR経路を `recipe=fasthmr` で選べるようにし、既存の身体配置とSceneResultへ接続しました。既定GVHMR経路の保存済みキャッシュを再利用できることと、FastHMRのPT32成果をPT16設定で読み込むと拒否されることを確認しました。 [Draft PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501)・[経路設計](https://github.com/Motoki0705/tennis-lab/issues/2491#issuecomment-7129950015)

CPUテスト67件とpre-commitが通過しました。モデル推論はスタブで検証し、固定manifestの6 clip・1,711推論フレーム・6,844カメラ画像の入力を照合しました。CIは実行中です。branchは `codex/parent-2492-issue-2491-primary`、HEADは `08e34581b6e1ecbb54d1b5e240595757d47fff65` です。PRのbaseには #2498 のbranchを使いました。

**親への依頼**　CameraHMRの利用条件を人間に確認してもらい、同意済みの `cam_model_cleaned.ckpt`、`smpl_mean_params.npz`、`SMPL_NEUTRAL.pkl` を `ckpt/fasthmr` 配下の所定の場所に用意してください。実機の速度・VRAM・parityと比較動画は、この3資産と #2496 の固定速度入力・基準値が揃ってから検証します。 [公式登録](https://camerahmr.is.tue.mpg.de/register.php)・[利用条件](https://camerahmr.is.tue.mpg.de/license.html)・[配置先](https://github.com/Motoki0705/tennis-lab/blob/08e34581b6e1ecbb54d1b5e240595757d47fff65/src/tennis_scene/pipeline/FASTHMR.md)

公開版のMask-ToMeパッチはCameraHMRのブロック専用です。HMR 2.0の完全構成は上流の対応待ちで、CameraHMRのTRT16は変換とparityが未確認です。 [公式パッチ](https://github.com/TaatiTeam/FastHMR/blob/dc5c8daf0beea56a419c631f494ef7bd7ce35d3c/model/hmr2_tools/patch_vit.py)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}