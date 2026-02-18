# tennis_scene 可視化改修調査レポート

## 1. 調査対象
- `src/tennis_scene/rendering/tennis_scene_renderer.py`
- `src/tennis_scene/scripts/visualization.py`
- 関連: `src/tennis_scene/pipeline/orchestrator.py`, `src/tennis_scene/utils/transforms.py`, `src/tennis_scene/pipeline/components/gvhmr.py`

## 2. 現状の事実（コード + 実データ確認）

### 2.1 現在の `smpl_vertices_global` は「local に PLCS を直接適用」
`orchestrator.py` で以下を実行しています。
- `smpl_vertices_global = apply_plcs_transform_batch(smpl_vertices_local, player_position, player_yaw)`

実データ `outputs/tennis_scene/clip.npz` で照合した結果:
- `max abs error`（再計算 vs 保存済み `smpl_vertices_global`）= `3.814697265625e-06`
- つまり保存済み `smpl_vertices_global` は、現在の変換式と一致しています。

### 2.2 `smpl_vertices_local` は「pure local」ではない
`GVHMRModule._run_inference()` は `pred["verts"]` を `smpl_vertices_local` として保存します。

実データで root joint（J_regressor[0]）を確認した結果:
- root がフレームごとに大きく移動（標準偏差が大きい）
- これは `smpl_vertices_local` に既に回転/並進（少なくとも並進）が含まれることを示唆

さらに `smpl_global_orient` の回転量:
- 角度ノルム min/mean/max = `2.361 / 2.968 / 3.141 rad`
- 無視できない回転が入っています。

### 2.3 既存変換は PLCS の root 位置と一致しない
現行式（`local -> rotate(yaw) -> +position`）で root を見ると:
- player 0: root と `player_position` の平均距離 ~`15.27 m`
- player 1: root と `player_position` の平均距離 ~`25.78 m`

### 2.4 `global_orient` を戻してから PLCS を適用すると root 整合が取れる
以下の手順を試験:
1. `root_local` を引いてメッシュ中心化
2. `smpl_global_orient` の逆回転を適用
3. PLCS yaw 回転を適用
4. `player_position` を加算

この場合 root と `player_position` の誤差は `~1e-6` オーダーでした。

## 3. 重大な不整合（追加発見）

### 3.1 回転軸が座標系と不一致
- `src/tennis_scene/utils/transforms.py` は **Y軸回転** を使っています。
- しかし court schema / renderer / PLCS データ生成は **Z-up（yaw は XY 平面回転）** 前提です。

したがって、PLCS yaw を適用する回転軸は Y ではなく Z が正しいです。

## 4. どこをどう変更すべきか

## 4.1 `src/tennis_scene/rendering/tennis_scene_renderer.py`

必要変更:
1. `smpl_vertices_global` 依存を固定にしない
- 現在は `scene.smpl_vertices_global` を直接描画。
- ここを「描画前に頂点を再構成できる」構造に変更。

2. `smpl_vertices_local` + `smpl_global_orient` + `player_position` + `player_yaw` から描画用頂点を生成する経路を追加
- 新規内部関数例:
  - `_recover_pose_local_vertices(...)`
  - `_compose_vertices_with_plcs(...)`
  - `_build_render_vertices(scene)`

3. 変換アルゴリズム（推奨）
- `root_local = einsum(J_regressor, verts_local)[..., root_idx, :]`
- `verts_centered = verts_local - root_local`
- `R_go = axis_angle_to_matrix(smpl_global_orient)`
- `verts_pose_local = R_go^T @ verts_centered`
- `R_plcs = rot_z(player_yaw)`
- `verts_court = R_plcs @ verts_pose_local + player_position`

4. 後方互換モードを残す
- 既存 `smpl_vertices_global` をそのまま使うモード（`legacy`）を残す。
- 新方式は `deorient_plcs` などの mode で切替。

5. `player_representation=skeleton` でも同じ頂点ソースを使って joint 回帰
- 現在 skeleton は `smpl_vertices_global` から直接 joint 回帰。
- ここも同じ変換ポリシーで統一する。

## 4.2 `src/tennis_scene/scripts/visualization.py`

必要変更:
1. 可視化設定を追加
- 例:
  - `style.vertex_source`: `"global" | "local_recompose"`
  - `style.transform_mode`: `"legacy" | "deorient_plcs"`
  - `style.plcs_yaw_axis`: `"z"`（デフォルト）

2. `TennisSceneRenderer` へ上記設定を渡す
- 既存は `TennisSceneStyle` に限られ、変換方針が指定できない。

3. エラーメッセージを明確化
- `local_recompose` 選択時に `smpl_vertices_local` / `smpl_global_orient` / `player_position` / `player_yaw` が無ければ明示的に失敗。

4. （任意）GVHMRの `Renderer` を使う別出力経路
- ユーザー提示コード相当（`render_with_ground`）を `backend="gvhmr"` として追加可能。
- その場合 `display=true` は無効化し、MP4保存専用にするのが実装上安全。

## 5. 影響範囲と注意点
- `src/tennis_scene/utils/transforms.py` の Y軸回転は実質バグ候補。
- 既存 `smpl_vertices_global` は旧定義で作られているため、修正後は見た目が変わります。
- 互換性のため、旧結果を再現する `legacy` モードを残すべきです。

## 6. 推奨実装順
1. `tennis_scene_renderer.py` に「local再構成 + mode切替」を実装
2. `visualization.py` で mode を指定可能にする
3. 小さな検証スクリプトで以下確認
- root が `player_position` と一致するか
- 2選手ともフレーム全体で破綻しないか
- `legacy` と `deorient_plcs` の出力比較

## 7. 受け入れ基準（最低限）
- `deorient_plcs` モードで root と `player_position` の誤差が十分小さい（例: 平均 < 5cm）
- `legacy` モードで既存と同等の動画を再現
- `player_representation=smpl/skeleton` の両方で動作
- 欠損キー時に明確な例外メッセージ

