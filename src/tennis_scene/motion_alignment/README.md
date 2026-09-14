# GVHMR world motion の PLCS 整合

保存済みGVHMRワールドモーションを、トラック全体に共通の
`s * Rz(yaw) * point + translation` でコートへ配置するCPU後処理です。
`scripts/align_gvhmr_world.py` がスケール固定・可変の両方を実行します。
フレームごとの補正、時間平滑化、ファインチューニングは行いません。

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
