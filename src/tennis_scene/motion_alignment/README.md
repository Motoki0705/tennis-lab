# GVHMR world motion の PLCS 整合

GVHMRのワールドモーションを、トラック全体に共通の
`s * Rz(yaw) * point + translation` でPLCSのコート軌道へ整合します。
フレームごとの補正、時間平滑化、ファインチューニングは行いません。

## パイプライン契約

整合は`TennisSceneOrchestrator.run()`内で常に実行されます。実行するかどうかを
選ぶ設定はありません。PLCS配置を上書きせず、`SceneResult`には両方を保存します。

| 内容 | `SceneResult`フィールド |
|---|---|
| PLCS配置 | `player_position`, `player_yaw` |
| incam GVHMR | `smpl_global_orient`, `smpl_vertices_local` |
| 整合後の配置 | `gvhmr_aligned_player_position`, `gvhmr_aligned_player_yaw` |
| 整合後のGVHMR | `gvhmr_aligned_smpl_global_orient`, `gvhmr_aligned_smpl_vertices_local` |

`smpl_body_pose`と`smpl_betas`は整合の前後で変わらないため共有します。下流は用途に
応じてPLCS配置または`gvhmr_aligned_*`を明示的に選びます。

world `transl`を持たない旧GVHMR artifact、識別不能なトラック、ソルバ失敗では
PLCSだけを保存するフォールバックを行わず、パイプラインをエラーにします。

## 整合後フィールド

`A = SMPL_Y_UP_TO_COURT_Z_UP`、推定したスケール・yaw・並進を`s`/`psi`/`b`、
正準頂点を`V_can`、そのjoint 0 rootを`c`、Y-upのworld回転を`R_world`、
court-frame headingを`theta_G`とすると、整合後の4フィールドは次の値です。

```python
gvhmr_aligned_player_position = s * Rz(psi) @ p_src + b
gvhmr_aligned_player_yaw = wrap(theta_G + psi)
gvhmr_aligned_smpl_vertices_local = s * (V_can - c)
gvhmr_aligned_smpl_global_orient = matrix_to_axis_angle(
    R_world.T @ A.T @ Rz(theta_G) @ A
)
```

これらはレンダラーの配置規則
`Rz(player_yaw) @ A @ R(global_orient).T @ (vertices_local - root) + player_position`
で、world頂点の直接相似変換を再現します。

`SceneResult.metadata["gvhmr_alignment"]`には`scale_mode`と、選手ごとのtrack ID、
scale、yaw、位置・heading残差、solver診断、confidence clipの有無を保存します。
PLCSの参照軌道は既存の`player_position` / `player_yaw`に保持されるためmetadataへ
重複保存しません。

## 設定

`configs/pipeline.yaml`の`player_motion.scale_mode`(`fixed`/`free`)と
`player_motion.alignment`配下の重み・正則化・ソルバ設定が推定方法を制御します。
既定値は位置誤差尺度0.5m、heading誤差尺度30°、heading項重み1、scale prior 1、
scale範囲0.5–2です。

`similarity.py`はfloat64で推定し、heading差の重み付き円周平均と閉形式スケールで
初期化します。headingが弱い場合はXYの重み付きSVDを使い、重力固定の5変数
（スケール固定時4変数）をSciPy TRFで最適化します。

fitのsource rootは、global meshへ`smpl_neutral_J_regressor[0]`を適用した点です。
raw `transl`を実pelvisとしては使いません。PLCSへの一致度は真の3D精度ではなく、
位置・heading残差はPLCSとの整合を監査する診断値です。
