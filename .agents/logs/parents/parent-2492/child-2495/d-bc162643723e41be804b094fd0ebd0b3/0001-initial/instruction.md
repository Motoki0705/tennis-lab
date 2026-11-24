# #2495 共通疑似GT・入力manifest・精度採点器（親 #2492）

担当issue #2495 の本文が仕様の正本。親 #2492 の「全体方針・合意事項」（とくに「代表clip」「共通ベンチマーク multiview-v1」の既知の問題）も読む。代表clip6本と評価frameの規則は2026-11-24にユーザーが合意済みで、変えない。

## 親が確認済みの事実（着手時の手がかり。コードで再確認する）

- 各clipの `scene/exports/<id>[local path]` に `smpl_vertices_local (P,T,6890,3)`・`player_position`・`player_yaw`・`smpl_body_pose`・`smpl_global_orient`・`smpl_betas`・`player_smpl_valid`・`player_rejection_code`・`player_kp_3d`（三角測量。正解に使わない）・`human_kp_2d (P,C,T,17,2)`（画像幅・高さで正規化したViTPose観測）・`human_kp_vis`・`player_track_ids` がある。`scene.metadata.json` に `court_reference.camera_fits`（K・R・t、歪み補正なしpinhole）と `body_placement` の区間・source camera がある。
- GVHMRは `src/tennis_scene/pipeline/frame_sampling.py` の約30fps格子（59.94fpsの元動画で1フレームおき）で推論され、body_placement後のSMPLは全frameに埋まる。GVHMRが実際に推論したframeは、sceneの `components/body_view_selection`・`components/gvhmr` の成果から特定する（再計算で代用しない）。補間frameは評価対象から外す。
- SMPL→COCO17は既存資産 `smpl_coco17_regressor`（`src/tennis_scene/configs/pipeline.yaml` の `body_models/gvhmr/smpl_coco17_J_regressor.pt`、`src/submodules/smpl_assets.py`）を使う。
- #1739 の新経路コード（PR #2449・#2469・#2461・#2470）は未マージ。sceneの読み出しに必要なら、mainの `src/tennis_scene/archive.py` 等で足りるか先に確かめる。足りなければ、その事実を親へ返す（勝手にPRブランチを取り込まない）。
- ベンチマーク multiview-v1 は読み取り専用。生成物は新しい版付きディレクトリへ置く。

## 進め方

1. まず変換の正しさを小さく確かめる: 1 clip・1 frameで、SMPL頂点→COCO17→各カメラ射影の結果をViTPose観測・三角測量の射影と並べ、関節順・左右・座標系・二重変換の有無を確認する。系統的なずれ（例: 全関節で数十px以上の一定のずれ、左右の入れ替わり、上下反転）が出たら、原因を特定してから先へ進む。原因が契約の解釈（どの変換を疑似GTの正とするか）に関わり判断が必要なら、証拠を添えて親へ返す（`needs_parent`）。
2. 品質gate・疑似BBoxの規則は、6 clipの統計を見て採点前に固定し、固定した規則と理由を記録する。候補モデルの結果を見て変えることはない（この子では候補モデルを動かさない）。
3. 採点器は再利用部品としてsrcに置き、合成ケースのunit testで契約を固める。現行ViTPose観測（scene内）を入力にしたsmoke採点で、数値が妥当な範囲か確かめる（循環性があるので「良い値」になって当然であることも記録する）。
4. overlay動画は6 clip×4カメラ。人間が見るので、骨格の色（疑似GT/ViTPose観測）と疑似BBoxを見分けられるようにし、clip・カメラ・frame番号を画面に出す。動画のパスを最終報告に書く。
5. 生成のうち長いもの（目安1分以上のCPU処理）は `parent.local_jobs start` で起動して待機を宣言する。GPUは使わない見込み。使う場合はtraining queueへ `add` する。

## 返してほしいもの

- コード（manifest生成・疑似GT生成・採点器・テスト）のPRを早めに作る（draftでよい）。PR作成後、生成jobやCIの待機を宣言して終了してよい。
- 最終的に: PR番号と完全HEAD、生成物のディレクトリ・manifest ID・sha256、品質統計（clip・camera・選手ごとの対象数・除外数・理由）、変換検証の結果、overlay動画のパス、knowledge記録。
- 命名は機能・方式が伝わる名前にし、issue番号はPR本文・knowledgeの追跡情報に置く。この版の生成だけに使うscriptはknowledgeのrun bundleへ置く。
