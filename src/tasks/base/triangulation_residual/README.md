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

`model.ffn_type`でcamera/time両方のTransformerのFFNを選択する。共有設定の既定値は`swiglu`で、`mlp`等の対応値は共通の`SUPPORTED_FFN_TYPES`で検証する。

### 残差特徴の入力比較（opt-in）

v1/v2とも既定値は`features.residual_encoding=raw features.residual_scale=1.0`で、従来の入力数値を維持する。比較時に`features.residual_encoding=asinh features.residual_scale=0.01`を指定すると、W/H正規化済みの`p_obs-p_reproj`だけを符号付き`asinh(residual / scale)`へ変換する。対象は特徴ベクトルの`[4J:6J]`で、特徴数・重み形状・その他の入力・幾何seed・教師・乱数列は変わらない。scaleは正の有限値を明示し、rawでは1.0に固定する。

この変換は小さいUV残差をMLP埋め込みへ渡す際の数値分解能を比較するための仮説であり、GT統計や実クリップのGT誤差を使わない。欠測残差のゼロとmaskを維持し、`GeometryInput.residual_uv`および推論保存のUV診断値はrawのまま保持する。v2の誤差生成・損失比較とは独立に指定でき、精度改善はpaired runで検証する。

## 欠測

confidence閾値以上の2 view以上で三角測量する。退化・負深度・絶対座標100m以上は無効。raw点はNaN＋valid=falseで保存する。
neural input用seedだけ、同じjointの観測済み3Dを時間補間（端は最寄り値）する。全期間欠測jointは観測由来rootへ置き、root観測も全期間なければ失敗する。GT/templateで補わず、補った点のvalidをtrueにしない。
学習sampleではカメラ部分集合をランダム順に列挙し、観測からrootを初期化できる最初の組を使う。全候補が失敗したときだけ次のcorruption roundへ進み、最大8 roundで停止する。camera選択と試行回数を保存し、scene・時間窓・split・GTは交換しない。GT誤差の小ささによる候補選択は行わない。

## 合成誤差・損失（v1の再現）

PLCSはACCAD由来のsingle_object_camera_view_v2のhuman_kp_3d.npyを使い、元motion fileのsplit重複を拒否する。subject-disjointではない。BLCSは同名の物理シミュレーションdatasetを使い、実クリップreplayを混ぜない。

1. GT worldとtrue cameraからclean 2Dを投影。真の画角もaugmentationする。
2. 入力へ白色・時間相関ノイズ、持続bias、低confidence外れ値、点/区間欠測、時間ずれ、radial distortionを加える。court14も破損する。
3. 別コピーのestimated cameraへ焦点距離・主点・回転・位置の誤差を加える。
4. noisy観測＋estimated cameraからX_initを復元し、同じestimated cameraで再投影する。

値の正本はconfigs/geometric_residual.yaml。clean/通常/hardを混合する。**誤差分布は初回学習用の合成仮定で、実測から推定した分布ではない。** estimated cameraは独立摂動で作り、sampleごとにnoisy Court14からcamera fitを再実行する方式ではない。実推論では保存済みCourt14由来の近似cameraを使う。

root/relative Smooth-L1にworld joints、true camera/clean UV再投影、GT速度、GT骨長の補助損失を加える。BLCSにはrelative/bone損失を付けない。速度をゼロへ近づけるsmoothnessではなく、GT速度との差を使う。true cameraとclean UVはlossだけが消費し、model.forwardの引数はfeatures/view_valid/time_positionsだけ。
train samplingはepoch/indexで再現可能。val/testは固定seed・窓。FPS間引き後の実時刻をRoPEと速度lossへ渡す。v1のrecipe・乱数列・損失は履歴checkpointを再現するため維持する。

## Court14校正・持続誤検出（v2）

`train_triangulation_residual_v2`は既存ACCAD/物理datasetのGT worldとsplitを使い、2Dを作り直す。保存sceneのcamera metadataは検証用に保持し、学習cameraは`cameras.py`の四隅4台＋両端中央フェンス付近2台から生成する。サイド中央と遠方のbroadcast cameraは含まない。真cameraの位置・高さ・画角にも変動を与える。trainは2–6 view、validation/testのview数はv2設定で明示する。

各sampleのcorruption roundで6候補すべてを一度だけ生成する。

1. true cameraから対象点・Court14を投影し、同じradial distortionと各観測誤差を加える。
2. noisy Court14の画像内・confidence閾値以上の点だけからcameraを推定する。`src/utils/geometry/planar_camera.py`のcoreは実pipelineの`court_reference.fit_camera`と共有し、中心主点・fx=fy・歪みなしを仮定して焦点探索＋PnPを行う。
3. fit成功cameraのsubsetを、同一の破損済み観測から選び三角測量する。失敗viewの理由、校正回数、geometry試行回数を記録する。再試行も最初の誤差種別・severityを保持する。GT cameraへの代用、GT 3D誤差による採否は行わない。8 roundで観測由来seedが得られなければ例外にする。

fitは点のcoverage・正depth・地上camera・焦点範囲を検査する。confidenceは点の採用に使い、目的関数は実pipelineと同じ均等重み。現段階ではRANSACを使わないため、高confidence Court外れ値の誤差も推定cameraへ伝播する。単一平面から自由な全intrinsicsを復元する方式ではない。

`persistent.py`は秒単位の区間イベントを生成する。PLCSは末端関節の持続offset/drift、左右交換、位置固定、BLCSは各viewで独立した偽軌道と静止点を使う。複数viewイベントは時間区間を共有できる。confidenceも時間相関を持ち、高confidence誤検出を含む。偽検出の存在はGTの可視性と分離し、画像内の偽点を残す。single-person ACCADに存在しない別人のGT軌道は生成しない。イベントmask/kindは診断専用でmodelへ渡さない。

`corruption_v2.py`は成分ごとに独立した乱数列を持つ。`v2.error_mode`は`clean / calibration / observation / temporal / persistent / combined / mixed`を明示選択できる。mixedはclean・通常・hardを含み、通常例で各誤差成分を分離して学習する。確率・継続時間・誤差量の正本は[geometric_residual_v2.yaml](configs/geometric_residual_v2.yaml)。分布は未校正の実験仮定であり、MeijiのGT誤差に合わせた値ではない。

モデル構造と入力の数値定義はv1と同じにして、`v2.loss_mode=legacy`と`balanced_regret`を同一入力で比較する。後者はsample内平均の後、存在するclean/通常/hard群を均等に平均する。root/relative/worldは3D距離のHuber、BLCSのroot/world重複は除き、PLCSのworld結合項は補助とする。true-camera再投影・GT速度・GT骨長は維持し、`relu(補正後3D誤差−初期3D誤差−許容値)`を追加する。GTとseverityは損失・診断だけに使い、推論gateや入力正規化には使わない。

`diagnostics.py`は平均・中央値・p95、point/sample改善率、初期誤差bin、誤差成分別、イベント中/外、clean raw-valid補正量、最大補正1%の改善寄与を出す。validation各epochを保存し、少数の大誤差改善だけで全体平均が下がるケースを検出する。損失変更が多数例の改善を保証するとは扱わず、paired runで検証する。

## 学習・推論

BaseLightningModuleのoptimizer/repro保存、BaseTrainingRunnerのcheckpoint/queue連携を再利用する。最小val/world_mpjpe_mのcheckpointを明示loadしてtestし、evaluation.jsonへ選択を記録する。実クリップ/testを選択に使わない。

このprofileのDataLoaderは、`num_workers>0`では`spawn`でpersistent workerを作り、各workerのOpenCVを1 threadに固定する。epoch共有値も同じspawn contextで作り、親でのepoch更新をworkerへ伝える。native libraryの状態をforkで引き継がないためのruntime方針であり、誤差生成・seed・教師の定義は変えない。`num_workers=0`での単一process実行も可能。

```bash
# GPU実行は共有training queueから。worktreeではmainのdata/output rootsを明示。
.venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual paths.data_root=/absolute/repo/data
.venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual paths.data_root=/absolute/repo/data
# v2。旧損失の比較runは末尾へ v2.loss_mode=legacy を付ける。
.venv/bin/python -m src.tasks.plcs.scripts.train_triangulation_residual \
  --config-name train_triangulation_residual_v2 paths.data_root=/absolute/repo/data
.venv/bin/python -m src.tasks.blcs.scripts.train_triangulation_residual \
  --config-name train_triangulation_residual_v2 paths.data_root=/absolute/repo/data
# 入力conditioningの比較では、上記コマンドの末尾へ次を追加する。
# features.residual_encoding=asinh features.residual_scale=0.01
.venv/bin/python -m src.tasks.base.scripts.infer_triangulation_residual \
  --task plcs --run-dir /absolute/run --clip /absolute/clip_000 \
  --output /absolute/comparison --device cuda
```

推論はnative frameを保持し、学習と同程度の秒数のwindowで残差を予測し三角重みで融合する。predictions.npzへ元2D・raw/filled初期値・mask・補正後3D・残差を保存する。同じcamera/2D観測集合で前後比較し、3D正解のない実映像で再投影誤差を3D精度と解釈しない。BLCS観測のobserved/interpolated/occlusion_estimated/unresolvedは重み・出典と共にmetadataへ保存する。

新checkpointのcontractはschema 2で、FFNとresidual encoding/scaleを埋込configと照合する。通常の学習configはこれらの必須項目を省略できない。履歴のschema 1は既知のPLCS/BLCS residual v1/v2 contract全体が一致する場合だけ、`training.migrate_legacy_checkpoint()`が警告付きでSwiGLU/raw/1.0のコピーへ移行する。公式`evaluate_clip`もこの入口を使い、元ファイルを変更せず、metadataの`checkpoint_migration`へ元contractと移行内容を残す。異なるroot・単位・task・特徴設定は拒否し、raw checkpointをasinhの入力定義へ読み替えない。

各taskのtriangulation_residual/data.pyとreal_clip.pyがartifact固有I/Oを、共有contracts/geometry/corruption/model/losses/configuration/data/training/inference/visualizationが残差profileを担当する。
