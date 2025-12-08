# ACCAD データ構造と SMPL-X 利用メモ

## データ構造 (ACCAD / AMASS)
- ルート: `data/ACCAD`
  - 各サブフォルダ: 例 `Female1General_c3d`。フォルダ名から subject (`Female1`) とカテゴリ (`General`) を推定できる。
- ファイル: `*_poses.npz`
  - `poses`: `(T, 156)` = 52 ジョイント * 3 軸角 (Global 1 + Body 21 + Hands 15+15)。
  - `trans`: `(T, 3)` 平行移動 (メートル)。`poses` と同一 T。
  - `betas`: `(16,)` まで存在するが、多くの SMPL-H モデルは 10 次元をサポート。
  - `gender`: `b'female'` 等のバイト列。`demo/check.py` では `str(gender_raw.item())` で文字列化。
  - `mocap_framerate`: 例 120 Hz。欠損時は 60 Hz デフォルト。
- データ数: `demo/check.py` 結果より 252 シーケンス / 192,553 フレーム / 約 26.7 分。カテゴリは Walking/General/Running... など 10 種程度。

## SMPL-X (SMPL-H) での利用手順
1. `demo/smpl.py` の `load_amass_sequence` で `poses`,`trans`,`betas`,`gender` を読み込み。
2. `split_smplh_poses` で `poses` を `(T,3)` global / `(T,21,3)` body / `(T,15,3)` hands に分割。
3. モデル入力に合わせて `reshape(T, -1)` し、`body_pose=(T,63)`、`hand_pose=(T,45)` に整形。`global_orient` は `(T,3)` を維持。
4. `betas` はモーション全体で共通なので `(1, num_betas_model)` に切り出し、`repeat(T,1)` でブロードキャスト。モデルが 10 係数までの場合は `min(betas_dim, 10)` で切り詰める。
5. `smplx.create(..., model_type="smplh", use_pca=False)` を使用。`use_pca=False` で 15×3 の指アングルをそのまま渡す。
6. `model(..., return_verts=False)` を `torch.no_grad()` で呼び、`output.joints` `(T, 73, 3)` を取得。必要に応じ `np.savez_compressed` で `_joints3d.npz` を保存。

## 実行のポイント
- 実行例: `uv run python demo/smpl.py --amass_npz data/ACCAD/Female1General_c3d/A1 - Stand_poses.npz`
  - `frame0 pelvis` などのサンプル出力で数値確認可能。
- データ全体の俯瞰: `uv run python demo/check.py --root data/ACCAD` でシーケンス数やカテゴリ分布を把握し、バッチ処理やフィルタリング条件を決めやすくなる。
