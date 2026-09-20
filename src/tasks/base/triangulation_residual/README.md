# 三角測量からのPLCS / BLCS残差学習

各cameraの`[p_obs, p_reproj, p_obs-p_reproj, court14, X_init, camera]`とconfidence・maskを入力し、観測由来の3D初期値を補正する新profile。座標はphysical court（XY地面、Z上）、出力はmetre。

## 出力とroot

PLCSのrootはCOCO17左右hip（11,12）の中点。既存PLCSのSMPL translation / yaw-canonical poseと異なる契約で、checkpointを読み替えない。

```text
r_init = mean(X_init[11], X_init[12]); P_init = X_init - r_init
output = delta_root(T,3), delta_relative(T,17,3)
r_final = r_init + delta_root; P_final = P_init + delta_relative
X_final = r_final + P_final
```

delta_relativeの左右hip平均をゼロへ射影し、root補正との重複を除く。相対姿勢はコート軸のままでyawを除去しない。BLCSはJ=1、position_residual(T,3)だけを出力する。

## 入力・モデル

`geometry.py`は学習・推論共通。内部2D特徴をW/H、global 3Dを共有court scaleで正規化し、relative poseはmのまま使う。cameraはfx/W,fy/H,cx/W,cy/H,R,C/scale。confidence、観測/三角測量/使用view/再投影mask、視線角も付与する。
1 camera×1 frameの特徴数はPLCS 367、BLCS 79。MLP埋め込み後、camera self-attentionとtime self-attentionを交互に適用する。camera任意indexへ位置encodingを付けずK/R/Cを使う。timeだけ秒×30のRoPEを適用する。camera順序に依存しないpooling後に残差headを読む。headの最終層をゼロ初期化するため未学習状態は幾何seedに一致する。

## 欠測

confidence閾値以上の2 view以上で三角測量する。退化・負深度・絶対座標100m以上は無効。raw点はNaN＋valid=falseで保存する。
neural input用seedだけ、同じjointの観測済み3Dを時間補間（端は最寄り値）する。全期間欠測jointは観測由来rootへ置き、root観測も全期間なければ失敗する。GT/templateで補わず、補った点のvalidをtrueにしない。
学習sampleではカメラ部分集合をランダム順に列挙し、観測からrootを初期化できる最初の組を使う。全候補が失敗したときだけ次のcorruption roundへ進み、最大8 roundで停止する。camera選択と試行回数を保存し、scene・時間窓・split・GTは交換しない。GT誤差の小ささによる候補選択は行わない。

## 合成誤差・損失

PLCSはACCAD由来のsingle_object_camera_view_v2のhuman_kp_3d.npyを使い、元motion fileのsplit重複を拒否する。subject-disjointではない。BLCSは同名の物理シミュレーションdatasetを使い、実クリップreplayを混ぜない。

1. GT worldとtrue cameraからclean 2Dを投影。真の画角もaugmentationする。
2. 入力へ白色・時間相関ノイズ、持続bias、低confidence外れ値、点/区間欠測、時間ずれ、radial distortionを加える。court14も破損する。
3. 別コピーのestimated cameraへ焦点距離・主点・回転・位置の誤差を加える。
4. noisy観測＋estimated cameraからX_initを復元し、同じestimated cameraで再投影する。

値の正本はconfigs/geometric_residual.yaml。clean/通常/hardを混合する。**誤差分布は初回学習用の合成仮定で、実測から推定した分布ではない。** estimated cameraは独立摂動で作り、sampleごとにnoisy Court14からcamera fitを再実行する方式ではない。実推論では保存済みCourt14由来の近似cameraを使う。

root/relative Smooth-L1にworld joints、true camera/clean UV再投影、GT速度、GT骨長の補助損失を加える。BLCSにはrelative/bone損失を付けない。速度をゼロへ近づけるsmoothnessではなく、GT速度との差を使う。true cameraとclean UVはlossだけが消費し、model.forwardの引数はfeatures/view_valid/time_positionsだけ。
train samplingはepoch/indexで再現可能。val/testは固定seed・窓。FPS間引き後の実時刻をRoPEと速度lossへ渡す。

## 学習・推論

BaseLightningModuleのoptimizer/repro保存、BaseTrainingRunnerのcheckpoint/queue連携を再利用する。最小val/world_mpjpe_mのcheckpointを明示loadしてtestし、evaluation.jsonへ選択を記録する。実クリップ/testを選択に使わない。

```bash
# GPU実行は共有training queueから。worktreeではmainのdata/output rootsを明示。
.venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual paths.data_root=/absolute/repo/data
.venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual paths.data_root=/absolute/repo/data
.venv/bin/python -m src.tasks.base.triangulation_residual.inference \
  --task plcs --run-dir /absolute/run --clip /absolute/clip_000 \
  --output /absolute/comparison --device cuda
```

推論はnative frameを保持し、学習と同程度の秒数のwindowで残差を予測し三角重みで融合する。predictions.npzへ元2D・raw/filled初期値・mask・補正後3D・残差を保存する。同じcamera/2D観測集合で前後比較し、3D正解のない実映像で再投影誤差を3D精度と解釈しない。BLCS観測のobserved/interpolated/occlusion_estimated/unresolvedは重み・出典と共にmetadataへ保存する。

各taskのtriangulation_residual/data.pyとreal_clip.pyがartifact固有I/Oを、共有contracts/geometry/corruption/model/losses/configuration/data/training/inference/visualizationが残差profileを担当する。
