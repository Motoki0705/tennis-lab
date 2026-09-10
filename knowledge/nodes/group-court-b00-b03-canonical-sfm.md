---
id: group-court-b00-b03-canonical-sfm
type: group
title: B00〜B03：SfM制約付きCourt dataset再生成
members:
- run-court-b00-canonical-sfm
- run-court-b01-canonical-sfm
- run-court-b02-canonical-sfm
- run-court-b03-canonical-sfm
parents: []
tags: []
---

## まとめ

B00〜B03の正式Court dataset生成・report公開と、公開後のalignment/Court owner再読込検証が完了した。B00の旧3,293枚は削除・置換済み。B01の手動配置3ファイルは生成前とbyte単位で一致する。

| Scene | コート数 | 採用枚数 | 採用率 | 拡大外周からの最小距離 |
|---|---:|---:|---:|---:|
| B00 | 2 | 2176 | 95.10% | 0.912 m |
| B01 | 3 | 2052 | 92.27% | 0.917 m |
| B02 | 1 | 2112 | 94.62% | 0.969 m |
| B03 | 1 | 2093 | 92.77% | 0.963 m |

全8,433枚の水平カメラ位置が、SfMカメラのconvex hullを頂点平均まわりに5%拡大した領域の、さらに0.5m以上内側にあることを検証した。各sceneでcircle/ellipse/rectangle/superellipseが採用されている。形状全体の包含は解析的support計算で制約している。高さは別の軌道設定で制御する。

B00・B02・B03はalignmentを現行コードで再推定した。v14は `ground-line-map.npz` の `semantic_ground_line_correspondences_v14` を指す。alignment JSON全体のバージョンではない。B02・B03のコート中心は旧配置と一致し、B00の対応中心移動は最大約1.2cm、尺度変化は約0.0473%。最新mainにある手動配置検証経路を使い、B01は再推定していない。

各sceneの16枚を抽出したプレビューを目視確認した。コート面・線を含む視点の多様性は確認できたが、ネット・背景建物・樹木にはぼけやにじみが残る。B02/B03は元シーンの夕方の暗さもある。SfM包含はアーティファクト除去や下流精度の保証ではない。定量的なアーティファクト比較や下流学習評価は未実施。

シーン別の画角・軸比・複合軌道中心設定、数値検証、プレビュー、カメラ位置図、実行ログは各run nodeと同梱bundleを参照する。正式ownerの再読込では `CourtArrayValidationMode.HEADERS_ONLY` を使用し、描画後の可視性・採用条件は生成時の正式assemblerで検証した。

途中の失敗・意図的中断は各成功nodeのparentsに記録した。B00の起動中にruntimeコードを変更したことによるクラス不一致も含む。最終3シーンは固定コード5589f446からCourt-only suffixで生成した。
