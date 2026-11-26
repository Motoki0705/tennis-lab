# テニスシーン3D再構成の論文設計

## 主題と範囲

同期した複数の固定カメラの映像から、プレイヤーの3D動作とボール軌道を共通のコート座標系に復元する。標準 pipeline の観測・幾何・時系列処理を中心に、同じ処理を複数クリップに適用する宣言型の実行システムまでを、日本語の技術報告としてまとめる。読者は、スポーツ映像の多視点再構成を評価・再現したい研究者と実装者を想定する。[親 #2627](https://github.com/Motoki0705/tennis-lab/issues/2627) · [要件合意](https://github.com/Motoki0705/tennis-lab/issues/2628#issuecomment-7190240176)

## 貢献

1. コートを参照とするカメラ校正、同一選手の視点間対応、プレイヤーとボールの3D復元を一つの手順として示す。出力には有効性 mask と理由コードを持たせ、どの時刻で再構成が成立したかを追跡できるようにする（第3–4章）。
2. 3DGS によるコート検出の学習データ生成、2D 追跡とコート座標による選手選別、ボールの外部注釈と時系列的な検出の難しさについて、アブレーションと失敗例を示す（第5–7章）。
3. component store、モデル共有、横断バッチ、TensorRT を備える実行システムを示し、保存出力の幾何整合・被覆と、処理時間・メモリ使用量・停止件数を測った条件とともに報告する（第8–9章）。

## 章立て、主張と根拠

主張は人間との合意、既存結果は #2630 の調査に基づく。検証結果が主張と異なる場合は数値と条件をそのまま残し、差を親に報告する。本文と図表に採る値は対応表に記録し、元の評価条件まで照合する。[根拠の対応表](evidence/claims.md) · [調査の統合コメント](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548)

設計と追加検証の成果を全文執筆に渡し、全文を揃えてから推敲・最終確認を行う。[全文執筆 #2638](https://github.com/Motoki0705/tennis-lab/issues/2638) · [推敲・最終確認 #2639](https://github.com/Motoki0705/tennis-lab/issues/2639)

### 1 はじめに

同期固定カメラの映像で、コートを基準にプレイヤーの姿勢とボール軌道を同じ座標系へ置く問題を導入する。課題は、視点ごとに異なる2D観測と欠測を扱うことと、同じ復元を多数のクリップに適用すること。公開3視点サンプルの SceneResult v2 保存を動作例に使う。[標準経路](../../src/tennis_scene/README.md#標準経路) · [公開サンプルの run](../../knowledge/nodes/tennis_scene/000028-run-multiview-sample-standard-pipeline-20261021.md) · [E2E・システム調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186)

### 2 関連研究

- **コート検出と校正**。TennisCourtDetector のキーポイント、PnLCalib の点と線の最適化、3DGS の新規視点合成を踏まえ、合成画像と幾何教師の生成を位置づける。一次資料は既存報告の書誌からたどる。[コート検出報告の関連研究・参考文献](../court_robustness/report.tex) · [コート調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189430057)
- **人物とボール**。StrongSORT++ のカメラ内追跡、GVHMR の単眼身体復元、TrackNet 系と WASB のボール検出を概観する。本報告で扱うコート座標での選別、カメラ間対応、時系列の利用へつなげる。[StrongSORT の出典](../../src/tasks/person_tracking/strongsort_NOTICE.md) · [GVHMR](../../knowledge/Papers/paper-2024-gvhmr/paper.md) · [TrackNetV2](../../knowledge/Papers/paper-2020-tracknetv2/paper.md) · [WASB](../../knowledge/Papers/paper-2023-wasb/paper.md) · [プレイヤー調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189419311) · [ボール調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189544500)

### 3 問題設定と出力形式

同期、FPS・解像度、カメラとクリップの軸、コート平面、クリップ内 ID を定義する。SceneResult v2 は元動画の時間軸で関節・身体・ボールを返す。実観測、関節の3D有効性、身体配置、球の3D有効性を個別の mask で示す。無効座標は0で保存し、有効性は mask で読み取る。理由コードを持つ配列ではコード0が有効を表す。型・形状と整合条件は実装のスキーマを使って表にする。[座標・対応](../../src/tennis_scene/README.md#座標対応) · [SceneResult v2](../../src/tennis_scene/README.md#sceneresult-v2) · [スキーマ](../../src/tennis_scene/schema.py) · [E2E・システム調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186)

### 4 再構成手法の全体

先頭フレームのコート観測と近似校正、カメラごとの検出・追跡、コート向きの決定、選手 ID のカメラ間対応、幾何検証と共通座標への校正、三角測量、GVHMR の位置・yaw 配置、SceneResult 保存を依存図で示す。ボール観測の運動整合・LOESS と、採用できる視点が足りない時刻の欠測を図に含める。公開サンプルの run で図の出力までの接続を照合する。[標準経路](../../src/tennis_scene/README.md#標準経路) · [pipeline の仕様](../../src/tennis_scene/pipeline/README.md) · [サンプル run](../../knowledge/nodes/tennis_scene/000028-run-multiview-sample-standard-pipeline-20261021.md)

### 5 コート検出

- **3DGS によるデータ生成**。SfM と3DGSでコート周辺を復元し、RANSAC 地面推定と多視点 LINE の投影で規格コートを配置する。方位・高さ・画角を変えた視点から RGB と幾何教師を生成する。既存報告の集計は4シーン・8,415枚・141軌道群。枚数は報告版の集計を採り、生成ノードの8,433枚とは版を照合して扱う。地面推定・LINE 投影・B00–B03 のアライメント・実画像の既存図を使い、対応表に元データを記録する。[既存報告と分割表](../court_robustness/report.tex) · [報告の枚数照合](../court_robustness/evidence/dataset_audit.json) · [生成グループと各 run](../../knowledge/nodes/court_detection/000007-group-court-b00-b03-canonical-sfm.md) · [図版](../court_robustness/figures/) · [元画像](../court_robustness/images/) · [図版の来歴](../court_robustness/README.md#同梱物)
- **視点軌道の多様性による堅牢性**。既存の固定枚数・更新数の比較では、合成1–4シーンの組合せで旧 TCD 実画像 val の平均 KP 誤差が 53.9、83.4、43.7、23.6 px と変わった（モデル入力座標、seed 42）。異なるシーン群の組合せの比較も既存の根拠とする。視点軌道だけの効果は同一シーン・枚数・モデル・学習予算で狭い分布と広い分布を比較し、収録分離の実画像と低角・横向き・見切れでの KP 誤差・H 成立率を測る。[シーン数の比較](../../knowledge/nodes/court_detection/000055-group-court-scene-diversity-colab-s42.md) · [シーン組合せの比較](../../knowledge/nodes/court_detection/000067-group-court-scene-pair-s42.md) · [コート調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189430057) · [検証 #2633](https://github.com/Motoki0705/tennis-lab/issues/2633)
- **校正への接続**。保存 KP・LINE を使った既存報告の三段階比較で共同推定後の線支持率を示し、共有ホモグラフィの run で固定カメラの動画単位の運用を説明する。支持率は保存 LINE との幾何整合、H の利用可否は校正工程の診断として扱う。[KP＋LINE 共同推定 run](../../knowledge/nodes/court_detection/000030-run-court-kp-line-hybrid-20260920.md) · [共同推定の保存値](../court_robustness/evidence/homography/results.json) · [動画共有 H の run](../../knowledge/nodes/court_detection/000078-run-court-video-drift-lines-20261117.md)

### 6 プレイヤーの3D空間上での追跡

- **2D アブレーションとその限界**。同じ保存検出を使った Meiji 3cam 開発比較では、StrongSORT++ に pose を加えて raw IDF1 が 0.930 から 0.955 に上がった。Deep OC-SORT＋pose/CLIP＋AFLink/GSI はこの StrongSORT++ より raw IDF1 が低く、カメラ間 pair F1 が高かった。高 IoU の重複 box 統合では採点指標が同値だった。追跡の表には採点した観測の被覆も載せる。[追跡方式 run](../../knowledge/nodes/person_tracking/000012-run-i964-tracker-hybrids-r10-20260930.md) · [重複 box run](../../knowledge/nodes/person_tracking/000013-run-i964-default-merge-r11-20260930.md) · [プレイヤー調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189419311)
- **コート座標での選手選別による精度向上**。標準経路では、追跡した box の足元をコート平面に逆投影し、滞在で候補を選び、視点間の対応を足元と外観の MILP で決める。既存の3視点比較は隣コート人物の除外、4視点の保存入力再生はダブルス幅での3D関節の有効選手フレーム増加を示す。純2D選別との比較は保存検出・追跡・特徴を固定し、開発と予約未見で精度・ID、4cam で完了・被覆を測る。[選手選別 run](../../knowledge/nodes/person_tracking/000004-run-i964-court-selection-r5-20260929.md) · [#2440 の幅比較 run](https://github.com/Motoki0705/tennis-lab/blob/63e9730fa23a57d46e7df4d9d4fed9e9204e9f94/knowledge/nodes/person_tracking/000014-run-i2440-player-selection.md) · [人物対応の実装](../../src/tasks/player_association/README.md) · [検証 #2634](https://github.com/Motoki0705/tennis-lab/issues/2634)
- **停止クリップと2D検出の因果**。人間が示した主張は、全件評価で止まった原因が2D検出にあること。#2363 の停止14件の判定箇所は人物対応13件・コート向き1件。保存2D観測のまま選別幅を変えると012・020・058は三角測量まで進み、020・058には4人目の検出と追跡が残っていた。残る11件の上流原因は判定できていない。#2636 では元映像に生 box・追跡 ID・選別結果を重ね、検出誤り、追跡誤り、幾何・対応の停止をクリップごとに照合し、可能な範囲で修正観測を再生する。[停止理由の run](../../knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md) · [固定観測での再生](https://github.com/Motoki0705/tennis-lab/blob/63e9730fa23a57d46e7df4d9d4fed9e9204e9f94/knowledge/runs/run-i2440-player-selection/comparison.json) · [停止クリップ調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189535218) · [検証 #2636](https://github.com/Motoki0705/tennis-lab/issues/2636)

### 7 ボール検出

- **GPT-6.2 Sol による外部注釈**。論文に採る外部注釈の生成モデルを、実際の出力に対応する campaign・実行時刻・ハッシュで確認する。仕様上は `gpt_annotation` が白紙注釈と固定幾何での再構成を接続し、既定の `model` は i935 の重みを使う。保存済み69クリップの外部座標を読み込んだ run には生成モデルの記録がなく、白紙注釈 plan に記録されたモデルは GPT-6.1 Sol。来歴の確定は #2637 に委ねる。[白紙注釈の仕様](../../src/tennis_scene/pipeline/direct_annotation/README.ja.md) · [外部座標の run](https://github.com/Motoki0705/tennis-lab/blob/669cd21156937bf899abcd96f796f8852427998d/knowledge/nodes/tennis_scene/000098-run-meiji-external-ball-20261124.md) · [ボール調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189544500) · [検証 #2637](https://github.com/Motoki0705/tennis-lab/issues/2637)
- **小さな球、解像度、カメラ品質、画面外**。容量系列の最小約8.0Mパラメータの条件は、固定の長系列 validation で平均/P95 が35.38/169.69元画像 px。T32/T64 の比較とともに、評価集合・採点単位ごとに表にする。難例は球像の小ささ、ブラーなどのカメラ画質差と縮小入力、遮蔽・画面外を扱う。Meiji の元映像は1920×1080で、512×288・1280×720はモデル入力。人工カメラ変動で平均誤差が15.4から32.2元画像 px に増えた実測と、69クリップの外部座標で境界外が4,136/184,912（大半が推定座標）だった診断を使う。実映像の球像径・ブラー・可視性は #2637 でカメラ別に定量化する。[容量比較](../../knowledge/nodes/ball_detection/000088-group-i1675-capacity-12k-s42.md) · [長系列 run](../../knowledge/nodes/ball_detection/000068-run-ball-sequential-length-20261025.md) · [入力サイズの診断](../../knowledge/nodes/tennis_scene/000043-run-scene-x3d-dpt-20261116.md) · [人工カメラ変動の run](../../knowledge/nodes/ball_detection/000060-run-i1049-convnext-query-posttrain-s42.md) · [ボール調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189544500) · [検証 #2637](https://github.com/Motoki0705/tennis-lab/issues/2637)
- **時系列の必要性と形状ベース手法の限界**。旧 TrackNet test の同系統の T1→T3 では F1 0.080→0.094。長系列の比較には追加更新数や方式差がある。時系列理解が難しさをどの程度下げるかを、同一評価集合の通常入力と時系列情報を除いた入力、色・円形度等に基づく対照で測り、形状だけで識別が難しい場面を今後の課題へつなげる。[短系列の比較](../../knowledge/nodes/ball_detection/000003-group-i551-dinov3-rope.md) · [長系列のアブレーション](../../knowledge/nodes/ball_detection/000094-group-i1899-controlled-query-ablation-s42.md) · [検証 #2637](https://github.com/Motoki0705/tennis-lab/issues/2637)

### 8 実行システム

- **宣言型 pipeline と component store**。clip ごとの入力、依存グラフ、成果物の型・識別子・上流のハッシュを宣言し、保存済み成果の再利用と続きの実行を可能にする。69件の CPU 事前計画の2,622ノードを例に、単一クリップと一括処理の入口を図にする。[実行計画の仕様](../../src/tennis_scene/pipeline/README.md#実行計画と入口) · [CPU 計画 run](../../knowledge/nodes/tennis_scene/000072-run-i1752-player-main-cpu-20261119.md) · [E2E・システム調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186)
- **モデル共有・横断バッチ・TensorRT**。モデルの寿命、互換形状のバッチ化、出力をクリップ・カメラ・フレームに戻す処理を示す。3クリップの管理対象7資源は各1回 load/unload、全件の TRT 評価で ViTPose・CLIP-ReID・HMR2 の充填率は99.86–99.98%。TensorRT FP16 は設定で選択し、モデル単体の速度と出力差を示す。全件の性能値は第9章に置く。[横断バッチの仕様](../../src/tennis_scene/pipeline/README.md#横断バッチの設定) · [TensorRT の仕様](../../src/tennis_scene/pipeline/README.md#tensorrt-fp16) · [モデル共有 run](../../knowledge/nodes/tennis_scene/000064-run-i1752-shared-fp16-20261118.md) · [一括 TRT run](../../knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md) · [姿勢モデルの単体 run](../../knowledge/nodes/tennis_scene/000077-run-i2303-vit-tensorrt.md)

### 9 実験

- **E2E 3D 精度の評価**。復元の成立、観測との幾何整合、絶対誤差を測る指標を定義する。旧標準の球は Meiji 3cam の1クリップで、カメラ1台を留保する LOVO の共通採点768点で再投影中央値/P90が8.97/25.40 px。固定校正は全視点由来で、評価対象1,462点に対する共通採点の被覆も示す。人物は #2363 の生成55件で関節96.04%・SMPL95.79%の有効性被覆、既存の疑似参照との比較は共有観測下での整合として示す。絶対3D誤差には、時刻・コート座標をそろえた独立3D参照で、球位置と同一 ID の関節・root の距離を測る必要がある。標準人物経路の未使用視点評価と合わせ、追加検証の担当は親 #2627 で判断する。[球の再投影 run](../../knowledge/nodes/tennis_scene/000033-run-ball3d-foundation-review-20261103.md) · [再投影・被覆の採点仕様](../../src/tennis_scene/evaluation/README.md) · [人物の被覆 run](../../knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md) · [疑似参照の run](../../knowledge/nodes/tennis_scene/000084-run-meiji-multiview-player-evaluation.md) · [E2E 調査と追加評価案](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186) · [親 #2627](https://github.com/Motoki0705/tennis-lab/issues/2627)
- **処理性能**。Meiji 4cam の固定コート、ボール無効、RTX 5060 Ti、TensorRT FP16 で69クリップを一括試行し、55件を生成した run の wall time 12時間10分58秒、およびサンプル最大 VRAM 1.76 GiB、RSS 6.16 GiB を示す。計時の開始点・終了点、完了基準、停止理由を表に添える。横断バッチ比較の同一 HEAD 逐次対照は別 run の初回1回があり、warm の単独効果には同一条件での反復比較が必要。追加計測の担当は親 #2627 で判断する。[全件 run](../../knowledge/nodes/tennis_scene/000083-run-i2363-trt-player-video.md) · [計時・資源ログ](../../knowledge/runs/run-i2363-trt-player-video/performance.json) · [同一 HEAD の初回逐次対照](https://github.com/Motoki0705/tennis-lab/blob/27c26f5364f664f38dc2a4f90795986718c447de/knowledge/nodes/tennis_scene/000080-run-i1757-serial-fb6255a-first-1.md) · [性能調査と追加計測案](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186) · [親 #2627](https://github.com/Motoki0705/tennis-lab/issues/2627)

### 10 限界と今後の課題

合成シーンの範囲と SfM のドリフト、未知収録での校正・ID・3D精度、ボールの画質・時間文脈・画面外、性能計測の条件をまとめる。追加検証で確定した結果と、独立3D参照や反復計測を必要とする項目を具体的に記す。[ドリフトの保存資料](../court_robustness/tables/sfm_drift.tex) · [コート #2633](https://github.com/Motoki0705/tennis-lab/issues/2633) · [選別 #2634](https://github.com/Motoki0705/tennis-lab/issues/2634) · [停止 #2636](https://github.com/Motoki0705/tennis-lab/issues/2636) · [ボール #2637](https://github.com/Motoki0705/tennis-lab/issues/2637) · [E2E・システム調査](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189522186)

### 11 おわりに

コート座標で統合する再構成手法、診断を持つ出力、同じ計画での一括処理が、どうテニス映像の再構成と評価を支えるかをまとめる。第5–9章で確認した成果を結論に対応づける。[標準経路](../../src/tennis_scene/README.md#標準経路) · [調査の統合コメント](https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548) · [全文執筆 #2638](https://github.com/Motoki0705/tennis-lab/issues/2638)

### 付録

実行時の設定と成果物の版、SceneResult v2 の mask・理由コード一覧、評価の母集団と追加表、図表の再生成入口を収録する。設定・配布重み・結果の出典はそれぞれの正本を参照する。[設定](../../src/tennis_scene/configs/pipeline.yaml) · [SceneResult v2](../../src/tennis_scene/README.md#sceneresult-v2) · [重みの登録簿](../../knowledge/registry/deployments.yaml) · [根拠の対応表](evidence/claims.md) · [全文執筆 #2638](https://github.com/Motoki0705/tennis-lab/issues/2638)
