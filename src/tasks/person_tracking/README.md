# person_tracking

保存済み人物検出から、カメラ内の姿勢・外観特徴と追跡を得る共通処理。
[選手poseのGPTレビュー](../../tennis_scene/chat_annotation/player_pose/README.md)が利用する。
コートによる選別、比較tracker、3D復元はこのモジュールに含めない。

| ファイル | 役割 |
|---|---|
| contracts.py | 検出row・box・COCO17姿勢・外観と実観測の型 |
| features.py | BGRフレームからViTPoseとCLIP-ReIDの特徴を取得 |
| pose_distance.py | bbox内で正規化した関節の照合 |
| strongsort.py | 位置・外観・姿勢によるオンライン対応 |
| strongsort_offline.py | AFLinkによる断片結合とGSIの別管理 |
| sequence.py | 採用済みStrongSORT++＋pose/CLIPの実観測系列 |

追跡IDはカメラ/clip内の暫定ID。raw IDを再利用・切り捨てしない。
補間bboxは別の再構成で、実観測の姿勢へ変換しない。
姿勢の第3channelは有限の回帰heatmap peakで、確率としてclipしない。
公開AFLink重みの出自と既存の利用判断は[NOTICE](strongsort_NOTICE.md)を参照する。

