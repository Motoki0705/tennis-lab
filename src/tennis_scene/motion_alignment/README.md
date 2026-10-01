# COCO17によるGVHMRの時系列配置

標準Tennis Sceneは、関節姿勢を固定して、三角測量COCO17へ各frameの位置・yawを合わせます。
全区間共通の相似変換と関節姿勢の最適化は提供しません。身体の形状は人物内で共通、
スケールは対応骨長比の中央値から人物の全有効区間を使って一度だけ推定します。

## 入力・目的関数

`temporal.py`は、コート方向へ回転済み・COCO左右hip中心を原点とした17点、
同じ人物・frameの三角測量点と品質重みを受け取ります。body pose、betas、roll/pitchは
最適化しません。scaleを`s`、yaw補正を`psi_t`、hip中心位置を`q_t`とし、
`J_t = s * Rz(psi_t) @ J_source_t + q_t`を推定します。

17点の3Dベクトル距離にはpseudo-Huber損失、位置とyawにはnative FPSに応じた
加速度正則化を使います。解析的な疎Jacobianを用いるfloat64 CPU最小二乗です。
失敗・未収束は理由付き欠測にし、他方式へ切り替えません。

設定の唯一の入口は`configs/pipeline.yaml`の`player_reconstruction.placement`です。
`data_sigma_m`、加速度のsigma・`temporal_weight`が目的関数、`min_joints`がframe支持、
`min_scale_pairs`とscale上下限が体格推定、`max_nfev`が計算量を制御します。
再投影の閾値と重みsigmaは1920×1080を基準とし、画像対角長の比で変換します。

使用するのは三角測量で採用された2視点以上のjointです。confidence平均を
`1+(reprojection_rms/sigma)^2`で割り、顔5点は0.5倍します。高さ[-0.25,3.5)m、
設定した再投影RMS以内の点だけを使います。confidenceは校正済み確率ではありません。
hips自体が欠けても、他の十分な17点から位置・yawが観測可能なら配置できます。

## 欠測と連続性

有効点不足、yawの幾何的退化、身体復元区間の切れ目で時間方向の最適化を分割します。
欠測frameを補間・外挿して有効3Dに復活させません。body poseの元frameへのSO(3)補間は、
対応が確定した同一GVHMR区間内だけで行います。人体の関節角を観測へ再fitする処理ではありません。

身体スケールの根拠不足・範囲外はその人物の配置を棄却します。ソルバ失敗は区間単位で
棄却します。`body_placement` metadataへ人物別scale、骨長支持数、区間・収束診断・
理由別frame数を保存し、`player_rejection_code`とv2 validity maskへ伝えます。
欠測コード101〜105の定義は`PlacementRejection`です。既存の無身体=1、速度棄却=6も維持します。

## SMPLとrendererの境界

`mesh_placement.py`はincam meshをroot中心のcanonical posed verticesへ変換します。
fitの原点はCOCO hip中心ですが、SceneResultの`player_position`はSMPL joint0です。
両者のoffsetにもscaleとyaw補正をかけ、`vertices_local`へ同じscaleを適用します。
`global_orient`は既存renderer式に整合する回転へ変換し、原姿勢の傾きを保持します。
3D関節・mesh・SMPL rootが同一の変換を受けるため、位置の二重加算はありません。

SceneResultのvalidity・archive契約は[tennis_scene README](../README.md)を参照してください。
旧v1 archiveの`gvhmr_aligned_*`とその閲覧機能は読込互換のため保持しますが、旧推定器はありません。
床接地・非貫通・関節IKはこの配置の目的関数には含めません。

## 保存済みデータでの確認

`tests/benchmarks/coco17_placement.py`は、保存した三角測量・GVHMRの固定bundleから
本番fitterとmesh/renderer境界をCPUで再検証します。ニューラル推論や学習は行いません。
`--repo`にSMPL-X資産を持つrepo、`--inputs`に`triangulation.npz`、
`scene_observations.metadata.json`、`cam1.gvhmr.npz`、`cam2.gvhmr.npz`を持つ
Meiji比較bundle、`--output`に別の出力先を指定します。入力SHAと数値診断を保存します。
