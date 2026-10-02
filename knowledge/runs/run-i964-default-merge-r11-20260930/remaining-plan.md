# #964 残項目の実行計画（run 11では実行しない）

前提: [2026-09-30決定](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5908318161)の
StrongSORT++＋pose/CLIP、COCO全画面.30、固定選別/v3。mergeは今回の結果を提示して採否を別途確定する。
以下の費用は計画値であり、実測やGPU実行許可ではない。実行前にorchestratorが枠を割り当てる。

## 1. #933の再評価と再較正

仮説: 旧trackで決めたsigma_m/外観尺度が新trackへ合わない可能性がある。
pair F1の差が尺度に起因すると現時点では断定しない。box coverage・group化・停止を分離して検証する。

1. 先に入力manifestとhashを固定する。既存#933の**ラベル無し較正clip**から最大8 clipを選び、
   4 devラベルclip・予約未見・同時作業中の出力を除外する。選定はclip名/動画グループと長さだけで決め、
   新trackerの結果やdev labelで選ばない。除外後に不足なら必要量を報告して新しい枠を依頼する。
2. 新しい共通componentで特徴/trackを生成する。較正中のmerge設定は先に確定した値を固定する。
   2D tracker・検出器・コート選別・上限6・v3区間切断/補完制限はこの段階では変更しない。
3. #933の擬似対応規則を引き継ぐ。外観を使う対応から同じ外観尺度を自己学習しないよう、
   高信頼な幾何による同一/別人候補と、外観の独立証拠を分ける。
   ambiguous/handoff/少数frameの組を除外し、除外数・camera×near/farの支持を保存する。
   連続frameを独立sampleと数えず、clip/人物/非重複時間窓で重みを均等化する。
4. `geometry.rayleigh_scale`の定義に従ってsigma_mを推定し、外観は同一/別人のcosineから
   正則化した2parameter slope/centerをfitする。旧値も比較条件として必ず残す。
   元動画単位のleave-one-video-outで較正loss/尺度の安定性を確認する。
   1動画しか残らない場合は独立性不足として再較正の採用を保留する。
5. score clip・min margin・runner-up比等の閾値は、まず旧値を固定して新尺度だけを評価する。
   閾値変更が必要なら**dev採点前**に較正用動画上の拒否率/擬似負例誤接続率の許容値と
   最大3候補を別addendumで宣言し、その較正集合だけで選ぶ。dev pair F1を最大化する探索をしない。
6. 全設定/manifest/hashをcommitしてから、4 dev×3cameraを旧尺度/新尺度で一回比較する。
   pooled pair F1だけでなくclip別、decided数、label coverage、group accuracy、exclusion、
   switch/fragment、camera×近遠、異常な停止を併記する。clipをclusterとするbootstrapは
   4 clusterしかなく不安定な記述的区間として扱い、有意差を主張しない。
   期待に反しても同じラベルへの再調整を続けず、原因と追加の独立較正データの必要性を報告する。

費用: 実装/監査3–5時間。較正8 clipを新規特徴生成する場合 **GPU 60–120分、peak 8–10GB、
disk 1–2GB**（DINO→ViTPose/CLIPを順に載せ、実測が12GBへ近づけば停止/小batchの新計画）。
既存特徴のidentityが完全一致する場合はCPU再利用可能。その場合のGPUは0。
CPU追跡/較正/評価30–60分、最大4thread、RAM peak見積り3–5GB。

## 2. Meiji video_000/clip_000の全pipeline qualification

設定を凍結し、別store/run IDで **全component execute→scene.json/export→3cameraレンダリング**を行う。
旧artifact loadを成功扱いにしない。AFLinkは既存の公開weightをcheckpoint root内へ配置し、root相対設定とhashを保存する。
比較CLIの`--aflink`は絶対パスだが、pipeline設定は共通PathResolverのroot相対契約に従う。
court_sideのball根拠、v3人物対応の停止理由、選択後上限6、GSI非観測、body/ball成果物のtimelineを監査する。
sceneの再loadと依存鎖/hash検証、動画全frame読戻しまでを完走条件とする。
GPU枠は共有training queueで **45–90分、peak 10–12GB、disk 1–2GB** を提案。
ViTPose/HMR2 batchを控えめにし、モデルを段階ごとに解放する。12GBを超える見積りならenqueueせず再計画。
CPUのstore/動画監査10–20分。コード/設定修正が必要なら新versionで証拠を分離し、凍結をやり直す。

## 3. 予約未見の一回評価

1と2、およびmerge採否が確定した後、tracker・検出・選別・association・評価規則・レビュー窓規則の
hashをfreezeコメントへまとめる。予約manifest自体は変更しない。
**予約clipを探索/較正へ一切使わず**、全3cameraを一回だけ評価する。
ラベルが無ければ凍結後に人手のidentityレビューを作り、そのラベルでの変更/再評価はしない。
数値が悪くても結果を全て残し、次の開発用データとして扱うのは別の未見集合を予約した後とする。

費用（予約clip数をNとして予算申請）: GPU **1 clipあたり15–30分、peak 8–10GB**、
CPU評価/3camera動画 **1 clipあたり5–10分**、新規disk **1 clipあたり200–400MB**。
人手相当のラベル/動画確認 **1 clipあたり20–40分**。既存archiveが新しいidentityに完全一致するときだけ再利用する。
このrunでは予約映像/ラベルを開いておらず、上記時間は過去のdev特徴生成からの概算である。

## #935との引き渡し

#935 ownerがこのbranch上へ積み直し、`TrackingConfig`、`merge_person_boxes`、`FeatureExtractor`、
`track_sequence`を用いるよう文脈producerを接続する。schema versionと全設定/重みhashをmanifestへ保存し、
旧115件の文脈は混ぜない。JPEGとraw動画の入力差は出自として残す。ownerのbranch/worktreeは変更しない。
共通入口の合成テストはrun 11で実施済み。#935実producerとsceneとの実データ一致は積み直し後の別検証とする。
