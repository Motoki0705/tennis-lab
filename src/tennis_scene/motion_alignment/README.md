# GVHMR world motion の PLCS 整合

保存済みGVHMRワールドモーションを、トラック全体に共通の
`s * Rz(yaw) * point + translation` でコートへ配置するCPU後処理です。
`scripts/align_gvhmr_world.py` がスケール固定・可変の両方を実行します。
フレームごとの補正、時間平滑化、ファインチューニングは行いません。

## パイプラインからの利用

同じ推定器を`tennis_scene`パイプラインの`TennisSceneOrchestrator.run()`から直接
使えます。`configs/pipeline.yaml`の`player_motion.source`が選手モーションの供給元を
選びます。

- `plcs`(既定): 現行どおりPLCSのコート軌道をGVHMRのincam SMPLで配置します。
- `gvhmr_alignment`: トラックごとに1つの重力固定相似変換を推定し、配置後のroot・yaw・
  SMPLパラメータを整列後の値へ差し替えます。`player_motion.scale_mode`(`fixed`/`free`)と
  `player_motion.alignment`配下の重み・正則化・ソルバ設定で推定を制御します。
  world `transl`を持たない旧GVHMR artifactでは、欠落フィールド名と
  「world-motion対応前に生成された」旨を示してエラーにします。

`src/tennis_scene/scripts/visualize_tasks.py`の`gvhmr_alignment`タスクは、整列後の
位置・yaw(実線)と`SceneResult.metadata["player_motion"]`のPLCS参照(破線)を
コート上面図で重ね、scale・yaw・位置/heading残差の中央値を表示します。

### 整列後のSMPLフィールド契約

`A = SMPL_Y_UP_TO_COURT_Z_UP`、推定したスケール・yaw・並進を`s`/`psi`/`b`、
正準(向きを除いた)頂点を`V_can`、そのjoint 0 rootを`c`、Y-upのworld回転を
`R_world`、court-frame headingを`theta_G`とします。`player_motion.source=gvhmr_alignment`の
`SceneResult`は次の4フィールドを持ちます。

```python
player_position     = s * Rz(psi) @ p_src + b
player_yaw          = wrap(theta_G + psi)
smpl_vertices_local = s * (V_can - c)                       # Y-up, root相対
smpl_global_orient  = matrix_to_axis_angle(R_world.T @ A.T @ Rz(theta_G) @ A)
```

`smpl_body_pose`と`smpl_betas`は変更しません。この4フィールドは現行レンダラーの配置規則
`Rz(player_yaw) @ A @ R(global_orient).T @ (vertices_local - root(V)) + player_position`
で、world頂点の直接相似変換
`s * Rz(psi) @ (A @ (R_world @ V_can - R_world @ c)) + (s * Rz(psi) @ A @ g_world + b)`
を再現します(合成データで最大絶対誤差<1e-4 m、float32保存)。
`SceneResult.metadata["player_motion"]`は`source`/`scale_mode`と、選手ごとの`track_id`、
`scale`、`yaw_rad`/`yaw_deg`、位置・heading残差サマリ(`median`/`rmse`/`p90`/`count`)、
`fit`診断(`success`/`nfev`/`optimality`/`jacobian_rank`/`n_free_parameters`/`initializer`)、
confidence clipの有無、可視化用のPLCS参照配列(`reference_position`/`reference_yaw`)を
保持します。

## 実行

専用worktreeをカレントディレクトリにして実行します。各パスは実環境に合わせて指定してください。

```bash
/home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tennis_scene.scripts.align_gvhmr_world \
  --clip-dir /home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000 \
  --motion-dir /home/kamimura/projects/tennis-lab/data/plcs/motions/gvhmr/meiji_3cam/video_000/clip_000 \
  --asset-repository-root /home/kamimura/projects/tennis-lab \
  --body-models-dir /home/kamimura/projects/tennis-lab/third_party/GVHMR/inputs/checkpoints/body_models \
  --output-dir outputs/plcs_gvhmr_similarity/clip_000 \
  --render-video
```

出力先が存在する場合は上書きせずエラーにします。必要な入力はclip manifest、
`annotations/tennis_scene/scene.npz`とmetadata、および各選手の
`cam*.gvhmr.npz`（`plcs_gvhmr_global_smpl_v2`）です。
従来の`gvhmr_result_cam*.json`はincamの姿勢・local頂点のみで、world `transl`がないため使えません。
外部motion抽出タスクのコードはimportせず、保存形式を検証して読みます。
クリップID・カメラ・fps・フレーム数を検証し、2D腰位置で選手を照合します。
独立した抽出器のtrack IDを同一視しません。曖昧な対応や選手不足はエラーです。

## 構成と推定

- `similarity.py`: float64の推定器。heading差の重み付き円周平均と閉形式スケールで初期化。
  headingのresultantが弱い場合はXYの重み付きSVDで初期化し、理由を診断に記録します。
  重力固定の5変数（スケール固定時4変数）をSciPy TRFで最適化します。
  `w * Huber(r²)`をcustom lossで実装し、重みでHuber閾値が変わることを避けます。
  `lambda_s * log(s)²`の事前項にはHuberを掛けません。欠測はゼロ重み、識別不能・非収束はエラーです。
- `artifacts.py`: SMPL-XからSMPL頂点をCPU再構成し、root・SMPL24・COCO17を取得。
  既存の`[x,y,z] -> [x,-z,y]`でZ-upへ揃えます。全関節に同じ相似変換を適用できます。
- `experiment.py` / `diagnostics.py`: 全区間推定、前半推定→後半評価、時間4分割残差、
  2D再投影、共通の低速足首マスク、相対回転・骨長スケールの保存性を比較します。
- `visualization.py`: 時系列PNG、コート上での3方式の同期比較動画。

既定値は位置誤差尺度0.5m、heading誤差尺度30°、heading項重み1、scale prior 1、
scale範囲0.5–2です。これらはCLIで変更できます。PLCS固有の不確実性は保存されていないため、
位置は腰の2D信頼度、headingは肩・腰の2D信頼度と回転行列X軸の水平投影長を使います。
コート可視性も掛けます。検出confidenceの[0,1]へのclipは診断と結果の説明に明記します。
sourceで未観測のフレームはfit・指標から除外しますが、出力モーションには保持します。

## rootと出力の契約

fitのsource rootは、global meshへ`smpl_neutral_J_regressor[0]`を適用した点です。
これは現行SceneResultレンダラーが`player_position`へ置くrootと同じ定義です。
**raw `transl`および既存motion archiveの`root_translation_m`を、実pelvisとして使いません。**
ただしPLCS学習データ生成はAMASSの`trans`をpelvisと呼んで使っています。
学習targetと身体モデルrootの意味の完全な統一は上流の未解決事項であり、本後処理で解消したとは扱いません。

出力は元のSceneResultを変更しない専用archiveです。
`player_<id>_{fixed,free}.npz`にはコート座標のroot、SMPL24、COCO17、回転行列、
body pose、betas、補正したglobal orientとtransl、そして必須の`body_scale`を保存します。
SMPLパラメータからのmesh再構成は必ず次の契約で行います。

```python
world_vertices = body_scale * SMPL(
    body_pose=body_pose, betas=betas, global_orient=global_orient, transl=0
).vertices + transl
```

元のSMPL-X rest pelvisを`c`、軸変換を`A`、`C=Rz(yaw)@A`とすると、
`R'=C@R`、`transl'=s*C@(c+transl)+b-s*c`です。
回転角をscale倍しません。SMPL-X→SMPL topology変換を使う場合は元と同じ変換を使います。
**このarchiveのtranslだけを通常のSMPLへ渡すと、scale可変時に正しいmeshになりません。**

`comparison_arrays.npz`は指標再計算と可視化用、`metrics.json`は設定・入力SHA256・診断です。
`inputs/`には利用したglobal SMPLを固定コピーします。
「直接PLCS配置」は同一source articulationから現行レンダラーの配置規則を再現した比較条件です。
元のsceneのincam meshそのものとの比較ではありません。

## 指標の限界

PLCSへの一致度は真の3D精度ではありません。再投影も保存済みの単平面近似カメラと
2D検出への一致度であり、歪み補正済みの正解カメラではありません。
足首速度0.2m/s未満のsource区間は接地の代理指標で、足裏接地ラベルではありません。
共通変換は元の足滑りを除去せず、速度をscale倍します。
身体サイズも同じscale倍になるため、大きなscaleが得られてもそのまま採用すべきとは限りません。

方法の背景: [GVHMR論文](https://arxiv.org/html/2409.06662v1)、
[SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)、
[SMPL-X](https://github.com/vchoutas/smplx)。実データの結果と考察は`knowledge/nodes/`に記録します。
