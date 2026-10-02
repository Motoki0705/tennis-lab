# run 29の3wrong caseの入力再現

元seed1、confidence seed29001、28条件・400sceneのRNGを最後まで進め、対象3件だけを分解した。
[analysis.json](run29-analysis/analysis.json)は元wrong-cases.jsonと全仮説float・pair数・判定入力の一致を要求して成功。
各NPZに全2D点、真の投影点、filter mask、distinct mask、synthetic false mask、元sceneのsource frame番号を保存した。
診断の最初の起動はtest namespaceのimportだけで停止し、同じ方法で修正して再実行した。GPU・媒体推論なし。

| 条件 / scene | camera順 | distinct前→後 | 後のpair支持数（01 / 02 / 12） | 誤って変わったcamera |
|---|---|---:|---:|---|
| false_0.10 / scene_007784 | cam_2 / cam_1 / cam_3 | 58→25 | 17 / 0 / 8 | cam_3 |
| false_0.20 / scene_006930 | cam_1 / cam_3 / cam_0 | 72→22 | 0 / 8 / 14 | cam_3 |
| window_150f / scene_006770 | cam_1 / cam_2 / cam_3 | 78→27 | 2 / 17 / 8 | cam_2 |

1件目ではcam_3のdistinct多視点観測44点中、真の球に近い27点がすべて消え、残った8点はすべてsynthetic falseだった。
真/誤の両仮説がcam_3の見えない17frameを支持し、誤仮説だけが残った偽点の4frameも支持する。
正解supportは38/58→17/25、誤仮説は15/58→21/25となり、margin .158586で誤採用。
reference cam_2と誤ったcam_3の直接pairは0だが、cam_1経由の8frame edgeが既存の接続条件を満たす。

2件目でもcam_3の真の球に近い32点がすべて消え、残った14点はすべてsynthetic false。
他cameraの残りはcam_1が8点（偽4/真4）、cam_0が22点（偽6/真16）。
誤仮説は18/22を支持し、そのうち14frameに偽点がある。正解は4/22しか支持しない。
元benchは正解が最良でもmargin .044263でSTOPしていた。filter後は誤仮説margin .168402で通過する。
reference cam_1とcam_3の直接pair0、cam_0経由の8/14frameのみとなる。

3件目には偽点がなく、残した点も全て真の投影から20px以内。
cam_2のdistinct多視点観測は65→10、reference cam_1との直接pairは61→2。
cam_3との8frame edgeを通じて接続は満たすが、向きを識別する観測が削られた。
正解supportは50/78→19/27、誤仮説は36/78→27/27、cost .091873 vs正解 .330644でmargin .238771となる。
これは「残した2D点が高精度なら向きも安全」という含意への反例である。

共通してfilter後の数/接続の変化が実測できる。cam3/2の単一pairだけではなく、全cameraの
実際の2D座標とsource frame、各仮説が支持するframe一覧はJSON/NPZを正本とする。
この事後解析だけから新しい必要frame数やedge閾値を選ばない。
