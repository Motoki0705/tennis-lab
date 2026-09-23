# 指定された最大変化の調査

読み取り専用scoutの調査と親による元映像確認。対象は最終pipeline scene、f807→808のBLCSとplayer1 f778→779のPLCS。

- T=1010、window=128、overlap=64、sample_stride=1。window startは0,64,...,832,882。両箇所とも同じ[704,832),[768,896)に属し、窓の追加・削除位置ではない。
- BLCSのblend weightは25/65・40/65から24/65・41/65、PLCSは54/65・11/65から53/65・12/65へ連続して変わる。src/utils/inference/windowed.py:18,55,106。
- Ball最大変化と同時にcam2 UVが約(+335.8,+34.7)px変わり、cam0は(-6.7,-5.9)px。両frameの可視性は[true,false,true]。Court点変化は各camera最大約1.6–3.4px。原映像ではcam2の予測が対象位置から隣接コート側へ移っている。
- PLCSではcam0のvisibility>=0.15が4/17→17/17となり13関節のmaskが復帰。cam2の共通可視関節は平均8.2px、最大31.9px変化し、cam1 Courtも平均4.8px/最大9.6px変化。src/tennis_scene/pipeline/components/plcs.py:328。
- 3DモデルはFPSを入力にせず、59.94006fpsは保存position差の速度換算だけに使う。stride=1の時間復元はidentity。固定scaleのmetre復元とreference→physical変換はframe間で同一。src/tasks/blcs/inference/predictor.py:273、src/tasks/blcs/model_io/contracts.py:405、src/tasks/plcs/inference/predictor.py:164。

窓結合・軸順・FPS倍率・二重denormalizeの不整合を示す証拠は見つからなかった。2D入力変化に伴う推論の不安定さで説明可能だが、未blendの個別窓出力は保存していないため寄与の分離は未確認。3D実測GTはない。元映像とcropはfigures/*-max-step-*.jpg。
