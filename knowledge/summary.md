<!-- knowledge-review: 361f2fe8d326e4cab90e1eb9a2305249978788ceb27a3a1bce10f3a5d4577a59 on 2026-10-01 -->
# Tennis Lab Knowledge Summary

更新日: 2026-09-30（#964の追跡3方式・native KPR・下流group評価を反映）

実RGB SLCSの130ノードをタスク別保存形式へ統合し、実験結果と採否を確認した。補助CLIの削除は学習結果・固定splitを変更せず、頑健性未達・固定test未評価という判断を維持する。詳細は[結果総括](reports/slcs-real-rgb.md)を参照。

従来の横断調査基準commit: `5f64fbd9c8fffc75295027eb2ece2a2f72eb6f9d`。追加実験の版・差分は各runの再現性bundleを参照。

この文書は、Tennis Labの学習・実験から得られた**現在の到達点、主要な知見、判断保留事項、次に解くべき課題**を横断的に把握するための要約です。個々の数値、再現手順、因果考察の正本は [`nodes/`](./nodes) のrun / group nodeと [`runs/`](./runs) の再現性bundleです。この文書は正本を置き換えず、研究状況を短時間で理解するための入口として使います。

現行knowledge graphの正式node typeはrunとgroupです。評価契約が異なる実験を同じランキングへ混ぜず、production、benchmark、family、diagnosticを区別して整理します。

## 2026-09-30の人物source・コート選別（#964）

[#937のFT検出器比較](nodes/player_detection/000001-run-i964-detectors-val-meiji-r1-20260929.md)では、
重み選択に使ったchat validationでprecisionが改善した。Meijiの参照は旧COCO boxに基づくため、
そこでの数字は旧boxとの一致率であり検出recallではない。FTの不一致はcam0の小さい遠側人物に集中し、
閾値0.3での不一致をそのまま検出失敗とは扱えない。ユーザーは2Dを全人物の候補生成へ、選手判定をコート座標での滞在時間へ移すと決めた。
[遠側GPU診断](nodes/player_detection/000002-run-i964-far-r3-20260929.md)はこの方針変更でcancelled。保存済み23archiveのhashを確認し、CPU比較へ再利用する。
1080/1920は11/12 camera-clipに限り、高解像度・tileは追加実行しない。選別精度と動画をrun 6で確認した。
CLIP-ReID/SOLIDER/KPRと複数trackerの固定dev比較はrun 9までに実施した。既定採用と新clipの調整後一回の未見評価は未完了。
既存のcamera間対応の結論は旧検出・旧追跡での結果として維持し、新経路へはまだ一般化しない。

共通人物特徴の[初回smoke](nodes/person_tracking/000001-run-i964-features-smoke-r2-20260929.md)は、
ViTPoseの回帰heatmap peakを確率とみなす検査で停止した。実入力のCPU再現で有限の1超scoreを確認し、
生値を保持する契約へ修正した。[GPU再実行](nodes/person_tracking/000002-run-i964-features-smoke-r3-20260929.md)は
同じ入力の1超scoreを保持して3camera×120frameを完走し、同じ#937検出のUltralytics BoT-SORT baselineも完走した。
これは機能smokeに限り、追跡品質の比較ではない。
[保存済みデータのCPU選別診断](nodes/person_tracking/000003-run-i964-court-selection-cpu-r4-20260929.md)では、
高い旧box一致率でもcamera-local滞在選別で遠側の観測を失い、FT低閾値/unionは隣コート人物も残した。
CLIP付きの第2確認も全clipでは決定できず、この基準のまま既定に採用しない。
続く[原因分離と断片連結](nodes/person_tracking/000004-run-i964-court-selection-r5-20260929.md)では、
確認できる隣コートunitは選手との混在ではなく横余白で採用されていた。主コート内の滞在coreと外側境界を分け、
足元連続性と利用可能なCLIPで断片を連結すると、隣コートを除外しFT/unionのcam0 far保持を改善できた。
単frameの足元跳びで分割する初回案は投影ノイズで過分割になり不採用。時間窓と1秒以内のgapに修正したが、
他camera/旧経路の選手保持低下とコート内へ投影される非選手が残るため、既定へは採用しない。
[全画面COCOのqueue job](nodes/player_detection/000003-run-i964-coco-fullframe-r5-20260929.md)は12 camera-clip完了し、全archiveのhash一致を確認した。
[run 6](nodes/person_tracking/000005-run-i964-fullframe-selection-r6-20260930.md)では選択済み断片の全観測を保持するよう修正し、
元データ固定のauditでwide観測の大半を回復し隣コート除外を維持した。ROI前7条件のCPU比較を完了し、
ユーザーはCOCO全画面 .30を選択し、[run 7](nodes/person_tracking/000006-run-i964-default-solider-cpu-r7-20260930.md)でpipeline既定とコート選別/v3接続へ反映した。
unionはwide/cam0遠側に利点があるが他cameraの保持を落とす。低閾値COCOはraw候補と断片を増やした。
参照がCOCOに有利である制約は変わらない。SOLIDER推論portは実重みの2 dev cropで上流CPU forwardと一致し、run 7時点では精度比較の前段に留まった。
[特徴job回収](nodes/person_tracking/000007-run-i964-coco-person-features-r7-20260930.md)で全24 NPZ・各40,531rowのhash/値/出自一致を確認した。
[事前固定した2方式×2encoder比較](nodes/person_tracking/000008-run-i964-tracker-matrix-r8-20260930.md)では候補内でDeep OC-SORT+pose/CLIPを推薦する。
新検出+旧追跡はLab連結曖昧により1camera停止（11/12完走）、候補は全camera完走した。停止を予測空として扱う固定規則の下で
推薦候補の選手coverageは増えたが、IDF1は微減しcam1遠側とfragmentが悪化したため、既定採用を自動で進めない。
BoT候補は非選手残存と1clipの対応停止が多い。camera間encoderをSOLIDERへ替えても今回の固定尺度で最終対応は変わらなかった。
[KPRの実2crop CPU parity](nodes/person_tracking/000009-run-i964-kpr-cpu-parity-r8-20260930.md)はpositive/negative両promptで上流と差0。
KPRの[全12 archive回収](nodes/person_tracking/000010-run-i964-kpr-native-features-r8-20260930.md)では、40,531rowの元検出・pose・出自が一致し、native parts/visibilityのshape・有限値・normを確認した。この回収は特徴整合性の確認であり、精度比較は次のrun 9に分けた。
[run 9のoffline linking・native KPR比較](nodes/person_tracking/000011-run-i964-tracker-linking-r9-20260930.md)では、
追加したStrongSORT++/CLIPが固定raw-ID主指標の候補内推薦となった。全12camera完走しDeep OC-SORTのcam1遠側/断片化を改善するが、
追加group IDF1とcamera間pair F1では新検出+旧追跡に届かないため、既定採用の合格とはしない。
旧Labの候補への適用は3cameraで曖昧停止。KPRはnative距離で評価し、trackerのgroup指標には利点がある一方、
camera間では固定CLIP尺度の転用が大半で曖昧停止となった。重み条件未確認のAFLinkを含め最終採用はユーザー判断を要する。
cam1遠側の欠測には全件元検出があり、重複検出由来の競合IDと別人trackへの移行/選別除外が主因だった。
小cropの外観/pose不良だけでは説明できない。全pipeline完走・調整凍結後の未見一回評価は後続とする。

[ユーザー指定の2 hybridを固定比較したrun 10](nodes/person_tracking/000012-run-i964-tracker-hybrids-r10-20260930.md)では、
StrongSORT++＋pose/CLIPがraw/group IDF1の候補内推薦となった。poseなしよりswitchと選手保持は改善したが、
fragmentは増え、cam1遠側の改善も小さい。camera間pair F1はDeep+pose/CLIPより低く、下流での一律な勝利ではない。
Deep+poseへAFLink/GSIを足すとrawは改善するがgroupは悪化し、pair F1の差は僅かだった。
旧9条件の全144層と決定済みpair指標を完全再現し、変えたStrongSORTのオンライン出力もpose重み0で一致した。
GSI syntheticは別maskのままで評価の実観測へ入れていない。費用付き重複box対策は未実装の提案に留めた。
この時点では既定判断を保留した。次のrun 11でユーザー決定を反映した。

[run 11](nodes/person_tracking/000013-run-i964-default-merge-r11-20260930.md)で、ユーザーが選んだ
**StrongSORT++＋pose/CLIP**を標準pipelineと#935向け共通入口の既定へ接続した。
元検出row/pose/CLIPを選別後も保持し、GSIを実観測へ昇格しない。AFLink公開重みは利用条件が未確認のまま
当面使用し、継続利用か自前再学習かを後日判断する。重複boxのgreedy IoU>=.8統合は明示optionで既定off。
同じ4 dev×3cameraで26boxを削減したが、raw/group IDF1・pair F1・switch/fragment・選手保持は変わらず、
今回の証拠ではmerge offの維持を推薦する。全26件の前後画像/ラベル監査で別人削除は認めなかったが、
完全GTや未見の安全性は保証しない。offのrun 10完全一致とschema/共通経路テストは確認済み。
#935実producerへの積み直し、独立した無ラベルclipでの#933再較正、clip_000全pipeline、
設定凍結後の予約未見一回は費用付き計画だけを残し、今回実行していない。

## 2026-09-27のcamera間人物対応（#933）

[run 12の事前protocol](nodes/player_association/000003-run-i964-recalibration-r12-20260930.md)は、
新既定StrongSORT++＋pose/CLIP、ユーザー確定のmerge offを固定し、無ラベル6clipで
尺度/判定しきい値を較正してからdevを一度採点する計画。
[run14のfit](nodes/player_association/000005-run-i964-recalibration-fit-r14-20261001.md)は支持/安定性条件を満たし、
LOVO正例recall85.93%、全動画の負例誤結合0でAを選択した。A/B同点、Cはrecall不足。
video_001/clip_020の停止を母数へ含み、同動画recall54.97%という弱点も残る。
新尺度を名前付きの非既定YAMLとしてcommit/pushした後、devを一度だけ採点した。
旧/新とも4/4決定、pair F1=.957119、group accuracy=.763256で、全12cameraのID配列が同一。
旧尺度の再計算もrun11に一致した。今回の再較正でdev低下は改善せず、旧尺度が主因という説明は裏付けられない。
dev後の再fit/再選択は行わない。
[clip_000資格確認](nodes/tennis_scene/000026-run-i964-clip000-qualification-r14-20261001.md)は、
run15に失敗を回収した。19/28nodeと153配列はhash/型/依存・人物元row/box/poseを照合できたが、
court_sideがcam2反転のmargin .118504 < .15で停止した。scene export・全長動画は未生成。
資源制限ではなくball根拠の曖昧性。[同じruleのCPU診断](nodes/court_side/000003-run-i964-clip000-side-diagnosis-r15-20261001.md)で
全score/元point gateの再現を確認した。275frame中89はcam2に情報を持たず、全3viewのsupportは5/52。
同じ校正/規則のball反実仮想はobserved注釈margin .604826、e9 cache top-1 .381587、
e9＋現行score/gate .406997でFFTに決まる。e9は720p JPEG/採用窓も異なり、元MP4本番への一般化は未確認。
ball labelはclip_000の診断専用で使用し、production import・人物label再採点・未見の開封は0。
閾値/既定は変更していない。診断nodeに、#935と接続する入力整合/ball証拠改善と、
#932で別評価が要るball-only集約/rig蓄積の費用・不確実性を提示した。どの対策も選択せず、全pipeline受入は未達。
旧#933の結論は旧trackに限定したまま維持する。2026-10-01のユーザー判断により、
[run16](nodes/player_association/000006-run-i964-unseen-r16-20261001.md)で候補Aを既定にし、
人物設定と資産hashを未見開封前に凍結する。clip_000完走を最後の未完項目として残し、
予約3clipを同じ注釈ball由来side規約で一回だけ評価する予定だが、事前検査で
video_001/clip_003のcourt/side参照欠測と、現componentの見積り約157分（2時間grant超過）が判明した。
run16のGPU投入・人物推論/採点は0。run17では欠測の明示的な停止扱いと3時間枠が承認され、
同ノードの実行addendumとパス契約修正の準備追記でCPU検査を完了し、全9camera/11,124frameを1jobへ登録した。
全cameraの人物処理/動画と停止clipの母数を保持し、
人物freezeとft-e13/court_sideは維持する。未見の採点は保存出力から次runに一回だけ行う。
[run 13の回収](nodes/player_association/000004-run-i964-recalibration-resume-r13-20261001.md)で、
特徴jobの時間切れと9/18cameraの完全性を確認した。lock待ちはtimeoutに含まれず、
旧見積りは不足していた。run 14で再開jobの成功と全18cameraのhash/元rowを検証した。
新規9cameraは約76分、peak GPU4.26GBで完了した。実fitと固定後のdev一回は上記run14で完了した。
準備中に旧devの小crop外観maskとproductionの差（9/40,531row）が判明した。
既定を維持して9行を明示mask投影し、2cameraのCPU再追跡と元row/GSI検証を完了した。
run 11は保存特徴からの再現として有効だが、画像入口との完全同一性の証明とはしない。

幾何（box下端の足元距離の対数尤度比）とCLIP-ReIDの外観をMILP（`cluster_multiview`）で統合し、コートの各sideで在場の長いidentityを選手に選ぶ対応付けを
[Meijiの人手ラベル4 clipで評価](nodes/player_association/000002-run-i933-association-meiji.md)した（sideは注釈ballの判定）。
4 clipとも停止せず、pair precision 1.0・F1 0.997、group accuracy 0.983、対象外の除外recall 1.0（precision 0.987）、本物のID switch 1件を検知した。
人手の対応があるclip_000はpair F1・group accuracyとも1.0で、PLCS Re-IDが入れ替えた2人を正しく対応付けた。誤りはすべて選手のboxの取りこぼし（短い区間の除外）で、誤結合は0。
この4 clipでは幾何だけでも同じ対応になり、外観は決定のマージンを上げる（停止の判定に効く）。データから決める値（`sigma_m`、外観の尺度）はラベルの無い7 clipの擬似ラベルで当てはめたが、
方式の設計中に同じ4 clipの失敗を見ているため完全な未見testではない。ダブルスは合成unit testだけで、実データは未検証。
本番でこの対応付けまで進めるかは、sideが決まるか（検出器ballでは多くのclipが停止、#934）に依存する。
この対応付けを`player_association` component（既定execute、`person_identities` v3はframeごとのID）にして人物対応のimportを削除し、[Meiji clip_000をimportなしで再実行](nodes/tennis_scene/000025-run-i933-association-meiji-clip000-20260927.md)すると、評価ラベルとの照合は全指標1.0（最小マージン8.0）で、scene.npzの全配列が人物対応importの000024とbit単位で一致した（残るimportは注釈ballだけ）。

PLCSのpose-only Re-IDを置き換えるため、公開重みの外観特徴5候補を[比較](nodes/player_association/000001-run-i933-appearance-backbones.md)した。
Meiji 4 clipの人手ラベル（camera間のtrack対応）ではCLIP-ReID（ViT-B/16、Market-1501）がAUC 0.926・top-1 0.930で最良、OSNet-AINが0.870・0.814で次点、DINOv3のCLSは偶然以下（0.41）だった。
chat-player-v1（単視点の放送映像）は全候補で飽和し（test AUC 0.997〜1.000）、候補の差を測れない。単視点で当てはめた類似度の閾値はcamera間に移らない。
足元の幾何は、足首ではなくbox下端をコート面へ投影する（低いcam2では足首の高さ約0.1 mが数mの奥行き誤差になる）。box下端では同じ選手のcamera間距離の中央値が0.2〜2.8 mだった。
統合した対応付けでも外観はCLIP-ReIDを使う（上の評価）。

## 2026-09-27のballだけのside判定（#932）

sideは`src/tasks/court_side`の幾何的な仮説検定でballだけから決め、`court_side`のimportは使わない（学習モデルなし）。
[合成ベンチマーク](nodes/court_side/000001-run-i932-synthetic-side-thresholds.md)で閾値を`min_frames=8, max_cost=0.8, min_support=0.2, min_margin=0.15`に決めた。
選定に使わないscene・seedのheld-outでも28条件の誤判定は0で、誤判定を防いでいるのはmarginである（誤った仮説が最良になった試行のmarginは最大0.10）。
静止した誤検出の反復が誤った仮説を支持する失敗を合成で観測し、同じ観測の繰り返しを証拠から除いた。
[Meiji 3camの全clip](nodes/court_side/000002-run-i932-meiji-clips-side-20260927.md)では、注釈ballで51 clip中50 clipが決まり（すべて[F,F,T]、margin 0.34以上、clip_000は人手確定と一致）、検出器ball（ft-e13）では7 clipだけが決まった（誤判定0、注釈と両方決まった6 clipはすべて一致）。
検出器ballの44 clipは理由付きで停止した。原因は閾値ではなく、検出器の見落としと、同じ別物体に長く張り付く誤検出である（注釈ballから20 px以内のprecisionはcam0 46%〜cam2 76%）。検出器ballの最良仮説が誤っていたclipが9あり（最大margin 0.086）、margin 0.15がそれらを止めている。
したがって本番でsideが決まる割合は、ball検出器の改善（#934、top-K出力）に依存する。court検出で失敗する6 clipは別の課題である。
既定`pipeline.yaml`で`court_side`をexecuteにして[Meiji clip_000を再実行](nodes/tennis_scene/000024-run-i932-component-side-meiji-clip000-20260927.md)すると、sideは[F,F,T]（margin 0.60）で、scene.npzの全配列がside importの000022とbit単位で一致した（ballは注釈のimport）。

## 2026-09-27の#915分割と既定設定での実clip確認

#915は#931でPR #937〜#940に分割した。PLCSの固定track Re-IDとCourtSideModelはmainに入れない（下記2026-09-24・25の記録は実験履歴として残す）。
`player_association`・`court_side`はtennis_scene所有のschemaだけを持つload専用nodeとなり、モデルが入るまで（#933・#932）は確認済みデータの`imports/`で埋める（`court_side`は#932でballから、`player_association`は#933で幾何＋外観から決めるcomponentになり、どちらのimportも削除した。上の節）。
分割後の既定`pipeline.yaml`（b863 Court＋region search、ROI 10m）で[Meiji clip_000を完走](nodes/tennis_scene/000022-run-i931-default-meiji-clip000-20260927.md)した。有効frame数とsideは#915の最終runと完全に一致した。
このrunでball・side・人物対応はimportしたものであり、side・対応モデルの精度評価ではない。
同時に、全repro.shのscript参照が再現可能であることを`kg_repro_paths.py`で検査するようにした。
v1 annotation layoutの廃止（readerはv2のみ）に伴い、v1の[Meiji 1clip datasetを新しい出力先へv2で再生成](nodes/tennis_scene/000023-run-i931-v2-dataset-meiji-one-clip-20260927.md)し、[DINO token](nodes/slcs/000139-run-i931-v2-dataset-meiji-dino-precompute-20260927.md)まで作った。validityは000022と一致し、SLCS window（DINO必須）として読める。残りのv1 dataset（Meijiの他clip、broadcast単眼）はv2で読めず、Meijiは#932・#933の後に再生成する。既存のv1 SLCS実験結果は履歴として有効だが、同じdatasetでの再学習はできない。

## 2026-09-26の宣言型clip pipeline検証

PR #915の宣言型component pipelineは、clip単位のimmutable store、各componentの型付き入出力とload/execute、先頭frameのCourt共同推定、外部ballの明示importでMeiji `clip_000`を処理した。初回の[設定](nodes/tennis_scene/000014-run-scene-component-meiji-fullclip-20260925.md)・[既定Court](nodes/tennis_scene/000015-run-scene-component-meiji-fullclip-r2-20260925.md)・[処理方針変更](nodes/tennis_scene/000016-run-scene-component-meiji-fullclip-b863-20260925.md)・[tracking人数上限](nodes/tennis_scene/000017-run-scene-component-meiji-firstframe-20260925.md)による停止を記録し、camera内ID分裂の一意な結合を経て[全段の初回完走](nodes/tennis_scene/000018-run-scene-component-meiji-tracklets-20260925.md)に到達した。ただしその完走ではcam2の重複IDと3D人物有効率の偏りが残った。

[重複ID修正後の実clip](nodes/tennis_scene/000019-run-scene-component-meiji-idstitch-20260925.md)では、学習済みRe-IDが異なる選手を結んでcamera alignmentに失敗した。生cosineとコート平面での足元距離は誤対応を支持し、合成100sceneのRe-ID評価を実動画精度へ外挿できない。ユーザー指定により、モデル予測・embeddingを別artifactで保持したまま、既存の人手人物対応を旧GVHMR bboxと現trackの一意照合後に明示loadした。[最初の確認済み対応run](nodes/tennis_scene/000020-run-scene-component-meiji-confirmed-reid-20260925.md)は下流を完走したが、対象外のraw人物trackを単独global IDとして3人目へ渡す誤りが残った。[最終2人軸run](nodes/tennis_scene/000021-run-scene-component-meiji-target2-20260925.md)では対象外trackを原検出・追跡に残して明示除外し、3camera×1010frame、対象2人の関節3D/SMPL配置、ball 3D、scene export、全段load-only再開を確認した。これは確認済み対応を使った処理・構造の検証であり、Re-IDモデルの実動画合格や独立3D精度保証ではない。次はモデル対応の実動画改善を独立に評価し、人手3D基準と可視化で配置・球軌道の品質を確認する。

## 2026-09-25のRe-ID補助head削除

入力された人物trackを対応付ける責務に確定し、Re-IDの補助人物判定head・補助損失・track棄却と学習時の偽track追加を削除した。[明示的なv2 checkpoint exportと100scene再評価](nodes/plcs/000123-run-plcs-headless-reid-export-eval-20260925.md)では、元のembeddingが全件bitwise一致し、現在の通常matchingでF1=0.9642・group完全一致77%となった。下記の診断値を新しい通常経路で再現した結果で、再学習や独立testの追加ではない。既存のembeddingは補助損失を含む旧学習に由来するため、pair lossのみで新規学習した精度は未確認である。次の検討は対応失敗の分析と独立sideの構造であり、side学習と実動画の採用判断は引き続き保留する。

## 2026-09-24時点の固定track Re-ID

PR #915はPLCS専用へ変更し、2D trackerが再登場も含め同一人物IDを維持する前提で、cameraごと累計4人・非再利用slotへ切り替えた。BLCS associationとtasks/base共通化を撤去し、PR #920の下流も単一2D球を直接三角測量する構成へ統合した。sideは独立境界に分離した暫定構成で、今回は学習しない。

[初回GPUスモーク](nodes/plcs/000114-run-plcs-track-reid-gpu-smoke-20260924.md)は完走したが、[compiled本学習](nodes/plcs/000115-run-plcs-fixed-track-reid-e60-s42-20260924.md)はvalidation NaNで停止した。[fresh推論](nodes/plcs/000116-run-plcs-reid-validation-numeric-probe-20260924.md)は有限で、[モード切替](nodes/plcs/000117-run-plcs-reid-mode-transition-probe-20260924.md)、[optimizer更新前](nodes/plcs/000118-run-plcs-reid-mode-cache-probe-20260924.md)、[反復評価とmask](nodes/plcs/000119-run-plcs-reid-sdpa-probe-20260924.md)を切り分けた。全無効行の自己参照化だけでは[実データGPU検証](nodes/plcs/000120-run-plcs-reid-safe-mask-gpu-smoke-20260924.md)の非有限値を解消できなかったため、compile=falseを明示し、自動fallbackは導入していない。[eagerの2epoch/全validation検証](nodes/plcs/000121-run-plcs-reid-eager-gpu-smoke-20260924.md)後に[新規60epoch学習](nodes/plcs/000122-run-plcs-fixed-track-reid-eager-e60-s42-20260924.md)を完了した。最低val/lossの42epoch目を固定した100scene testでは、cosineペアF1=0.9641、補助headを含む既定matchingはprecision=0.9974・recall=0.7471・F1=0.8543、group完全一致33%だった。同一embedding/閾値で補助headだけを使わない診断ではmatching F1=0.9642・group完全一致77%となり、真の人物trackの棄却が主な取りこぼし要因である。これは既定pipelineの成績ではなくpost-hoc診断で、testでcheckpointや閾値は選び直していない。FP追加と欠測augmentationの順序が補助headのshortcutになる可能性はあるが、直接検証は未実施。入力を対象人物trackだけとするか誤検出除外も担当するかを確定し、対応付けと選別を分けて扱うことを次の判断とする。生成は3〜4view・scene内1〜4人の固定scene splitで、未見motion・実動画・5view以上の精度や本番採用を主張しない。独立sideは未学習で、Re-ID評価を踏まえて構造を再検討する。

## 2026-09-23の自動scene統合確認

標準tennis_sceneを手動side・人物対応なしのassociation→三角測量→GVHMR経路へ接続した。実重みの[PLCS全clip CPU検証](nodes/tennis_scene/000012-run-tennis-scene-plcs-association-cpu-fullclip.md)と[BLCS全clip CPU検証](nodes/tennis_scene/000013-run-tennis-scene-blcs-association-cpu-fullclip.md)では、512前後/約1024 frame、3/5 viewが有限値で完走した。一方、side/ID品質は本番採用基準に未達であり、新association配布重みを承認した結果ではない。ID棄却を含む指標と従来の学習時指標を区別する。以下の3D baselineは履歴として維持し、現在の標準設定の変更を精度改善やdeploy認定と解釈しない。次は学習安定性の解決とvalidation評価、その後に実動画の校正・3D有効率まで含む受入を行う。

## 2026-09-22の追加確認

camera-local V2観測からsideとclip内IDを推論するモデルを既存task runnerへ接続し、Global MHA＋mHC、幅512・12 stage・8 headのGPUスモークを[PLCS](nodes/plcs/000113-run-plcs-association-refactor-512-smoke.md)・[BLCS](nodes/blcs/000042-run-blcs-association-refactor-512-smoke.md)で完了した。両方ともfit/validation/testとcheckpoint再読込が成功し、batch2の最大予約メモリは約5.45 GiBだった。4更新だけの診断ではside balanced accuracy=0.5であり、収束や既存deployへの優位性は確認していない。次は再生成済みV2の同じ800/100/100 split・seed42で両taskを新規60 epoch学習し、sideの両クラスrecallとID/FP指標、定常学習性能を確認する。以下の3D推定baseline・deploy判断は更新しない。

## 2026-09-23の統合パイプライン確認

[Meiji clip_000の既定Court再生成](nodes/tennis_scene/000003-run-tennis-scene-cleanup-meiji-court-model-20260922.md)では、3camera×1010 frameに共通の完全14点校正frameがなく、後段の3D推論前に停止した。実行時checkpoint SHAは事前記録と一致した。既定hybridへの移行を実動画E2Eの精度保証とみなさない判断を維持する。[画像だけによる領域探索](nodes/court_detection/000032-run-court-meiji-model-only-regions-20260922.md)では、既存b863…checkpointと固定grid選択で27/27のhybrid推論・3camera校正が通った。手動点と外注ballは推論入力にしていない。[全3030frame](nodes/tennis_scene/000004-run-tennis-scene-meiji-court-regions-dino-block-20260922.md)でも全camera・全frameのKP14と校正が成立した。後段は旧DINO拡張のdeprecated APIで停止し、[正規buildと32frame chain確認](nodes/tennis_scene/000005-run-tennis-scene-dino-extension-chain-20260922.md)で演算互換性を確認した。[再実行の人物照合](nodes/tennis_scene/000006-run-tennis-scene-meiji-auto-person-mismatch-20260922.md)で、同じtrack IDでも隣接コート人物を選ぶauto選択を検出し、公開前に中止した。[モデルCourtの5m filter](nodes/tennis_scene/000007-run-tennis-scene-meiji-court-filter-5m-20260922.md)では対象へ戻った一方、cam2遠方選手の境界欠損と、広いROIの投影反転が判明した。[可視ROI修正と10m margin](nodes/tennis_scene/000008-run-tennis-scene-meiji-visible-court-roi-20260922.md)では、cam2遠方選手の直接観測が298→925frameとなり、全cameraで既存対象への軌跡対応を確認した。補間区間は残るため、次は全段生成時の姿勢照合と3D/動画評価を行う。

## 2026-09-21の追加確認

技術報告に伴う5ノードを確認し、実写homography postprocessと合成データ源の幾何的不確実性を更新した。以下の知見は既存deploy checkpointの昇格根拠にはしない。

- **Court detection / 実写homography postprocess**: [指定4写真の保存予測](nodes/court_detection/000028-run-court-supplied-photos-paper-20260918.md)に対し、[PROSAC](nodes/court_detection/000029-run-court-prosac-paper-20260920.md)と[KP・LINE共同推定](nodes/court_detection/000030-run-court-kp-line-hybrid-20260920.md)はいずれも4枚でHを生成した。共同推定はKPのみより予測LINEとの双方向内部整合を高めたが、人手GTがなく、段階間で採用KPも異なるため、実コートへの精度向上率や4写真外への汎化は未確認である。写真CのLINE欠落・対応の曖昧さも残る。次は会場分離の人手GTと固定モデル・解像度で旧H／PROSAC／共同推定の誤差、失敗率、棄却率、処理時間、下流E2Eを同条件比較する。保存出力による再検証は可能だが、ニューラル再推論には[記録済みのcheckpointハッシュ不一致](../paper/court_robustness/README.md)の解消が必要である。
- **Synthetic data / SfM幾何**: [B00の保存トラック診断](nodes/synthetic_data_generation/000022-run-court-sfm-ground-drift-20260919.md)に続き、[B00〜B03の共通2区間診断](nodes/synthetic_data_generation/000023-run-court-sfm-all-scenes-drift-20260920.md)でも、時間分割した共通地面セルに高さ不整合を観測した。SfM driftと整合する兆候だが絶対ドリフト誤差ではなく、符号はシーンごとに異なり、B01の偏りは小さく、B03は支持セルが少ない。地面凹凸・特徴点誤差・三角測量誤差も分離できないため、合成教師の幾何的不確実性として扱う。次は再訪で十分に重なる地面観測と独立地面基準を用意し、長距離構造制約・loop closureの有無を同条件で比較して下流court精度への影響を測る。

[PLCS・BLCSの三角測量残差モデル初回検証](nodes/plcs/000105-group-geometric-residual-v1.md)では、合成testの幾何初期値に対してPLCSの平均3D誤差が約20%、BLCSが約1%減った。一方、Meijiの同一実clipでは両taskとも再投影誤差が増え、PLCSの大きな骨長異常とBLCSのほぼ一定の微小補正が残った。合成64sceneの追加診断でも3D誤差改善と入力観測への再投影悪化が同時に起きたため、実clipの再投影悪化を実3D悪化とは断定しない。独立3D正解はなく、実映像の精度改善・production置換を支持する根拠は得られていないため、追加profileは実験用とする。合成camera摂動とCourt14からの推定誤差構造の差は原因候補だが未分離であり、次は同一splitでcamera再推定と独立摂動を比較し、clean caseの不要補正・実clipの持続外れ値を別に確認する。従来PLCSのSMPL-root/yawとは出力契約が異なるため、既存deployの指標と直接順位付けしない。

[三角測量残差v2の比較](nodes/plcs/000111-group-geometric-residual-v2.md)では、Court14再推定・四隅＋正面2候補・持続誤検出を実装し、同一観測/初期3Dの6条件を評価した。validationで選んだPLCS legacy/rawはtest平均0.155524→0.119929m、BLCS balanced/rawは0.844004→0.812402m。asinhはBLCSのlegacy対照には有効だったが、PLCSでは悪化し最良構成も更新せず、入力感度と精度を区別する。PLCSの実clip骨長異常は減ったが長い前腕・再投影増大が残り、BLCSの実補正は約2mm。独立3D正解がなくproduction置換の根拠とはしない。BLCSの[native worker中断](nodes/blcs/000037-run-blcs-residual-v2-balanced-s42.md)は[checkpoint復旧](nodes/blcs/000040-run-blcs-residual-v2-balanced-s42-resume.md)で完走したが、後半のstochastic順序差が制限。次はcamera-onlyの識別性、低誤差点の不要補正、長い欠測、実測誤差分布と複数seed/会場を確認する。

2026-09-21の導入方針見直しでは、上記比較結果を再確認し、残差モデルをPLCSのみに限定した。Court14再校正＋従来損失＋raw入力を単一構成として採用し、BLCS導入と不採用分岐を撤去する。過去のBLCSを含む実験記録は保持し、再現は記録済みcommit/patchを使う。測定結果・実写精度の制限とproduction非昇格の判断は変わらない。

## 2026-09-20の追加確認

CIと登録SKILLの整合性を再確認した。保存形式・未完成の記録・summary本文の更新検出を強化した運用上の変更であり、実験結果や以下の研究判断には変更がない。

初回移行時点の全200ノードを7つの機能・研究トピックへ分割した。既存の実験IDと数値・再現bundleは保持している。論文の出典は[Papers](Papers/README.md)に一元化し、GVHMRを利用するPLCS記録に背景研究の参照を追加した。これは過去runが論文の手法を比較検証したという主張ではない。

9月に追加された知見のうち、次の判断を更新する。

- **Court detection**: [run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1](nodes/court_detection/000026-run-court-residual-wideline-pose-lora-b8-e20-s42-resume-e5-retry1.md)では親に対してKP平均誤差とline Diceが改善する一方、seg mIoUとpose再投影が悪化した。head構造とline幅を同時変更しており、単一要因の効果は分離できない。次は同一line schemaでheadだけを比較する。
- **PLCS / motion source**: [run-plcs-accad-gvhmr-meiji-1000-v1-train](nodes/plcs/000102-run-plcs-accad-gvhmr-meiji-1000-v1-train.md)は200 epochの学習と混合testを完走したが、向きとposeに改善余地がある。生成教師に対する評価で、実動画の独立3D正解への精度でもGVHMR追加の因果効果でもない。次は固定val/testでACCAD-only対混合train、未見収録holdoutを比較する。
- **Synthetic data / B00外観変換**: [50/100枚・7k/30k比較](nodes/synthetic_data_generation/000021-group-b00-clay-flare-images-steps-v1.md)では、共通評価8視点で100枚30kが今回の実験基準となった。50枚の学習を延ばすだけではSSIM低下と白線の薄れがあり、100枚版にも近いネットのぼけ・線の欠けが残る。[ホスト再起動による中断](nodes/synthetic_data_generation/000016-run-b00-clay-flare-nht-7k-interrupted-v1.md)は失敗として保存した。生成教師に対する単一シーン・単一seedの診断であり、実世界の幾何精度やproduction採用は未検証。次は元画像100枚の同条件対照で変換由来の誤差を分離し、編集範囲と視点間整合性を検討する。
- **統合・生成・UI検証**: 新しいtask区分によりデータ生成・smoke・統合診断を辿れる。これらの完走を推定精度の改善と混同しない。最新の個別条件・残課題はノード本文を正本とする。

以下のproduction/deploy表は**2026-09-04時点の調査記録**を保持している。今回の構造移行ではcheckpointの再評価・昇格をしておらず、9月20日の現行配備状態を保証する表には更新しない。新規結果を以前の異なるsplitと直接ランキングしない。

## 現在の全体像

現行pipelineでは、2D ball detection、court detection、single-person PLCS、single-ball BLCSにdeploy checkpointがあります。SLCSは全体版の収録分離学習・validation比較まで進みましたが、実世界での頑健性は未確立です。実RGB用の教師checkpointは専用生成profileで選択し、以下の従来pipelineのdeployとは区別します。

追加の[実RGB SLCSの実験群](nodes/slcs/000009-group-slcs-real-rgb.md)では、Meiji全体の生成・品質確認とbroadcastとの統合を完了しました。現在の課題は入力欠損・裾誤差・時間的スパイクであり、教師生成の未完了とは区別します。

従来の2026-08-30のBLCS観測ベース2D追跡比較（[#832のgroup](nodes/blcs/000031-group-i832-blcs-observation-tracking.md)、3 run）では、conservative設定が学習用associationの運用選択となりました。ただし単一seed・FP augmentation無効のfamily内比較であり、single-ball deployの置換や実動画での優位を示す結果ではありません。

| task | 現在の基準 | 主な固定値 | 現在の判断 |
|---|---|---|---|
| `ball_detection` | [`run-i618-convnext-v2-ft`](nodes/ball_detection/000009-run-i618-convnext-v2-ft.md) | test F1 `0.721789`、precision `0.735656`、recall `0.708436`、距離 `2.176208 px` | offline最高値ではなく、実clipのcoverageと軌道安定性を含めてdeploy継続 |
| `court_detection` | [`run-i621-court-kp512-resume-r4`](nodes/court_detection/000006-run-i621-court-kp512-resume-r4.md) | val best `2.23 px`、固定checkpoint再評価 `1.708886 px` | 旧KP14 / 512入力deployの比較値。現在のhybrid移行は別評価で、独立held-out testは未確立 |
| `plcs` | [`run-deploy-multiview-plcs-i590-courtkp14-v2`](nodes/plcs/000082-run-deploy-multiview-plcs-i590-courtkp14-v2.md) | position `0.175284 m`、yaw `6.443357°` | 3–6 camera・court KP14の現行single-person deploy |
| `blcs` | [`run-deploy-multiview-blcs-v3-simfix-c3-6-v2`](nodes/blcs/000011-run-deploy-multiview-blcs-v3-simfix-c3-6-v2.md) | position `1.064595 m`、endpoint `2.024551 m` | 3–6 camera・court KP14の現行single-ball deploy |
| `slcs` | [全体版のval5条件](nodes/slcs/000062-run-slcs-full-no-smooth-gap-rgb-val-v2.md) | 全61clipの固定split、60epoch、入力条件・train定数baseline比較 | ball低分散崩壊を脱しRGBの寄与を確認。欠損・裾・時間的スパイクは残り、頑健なdeployとは未認定 |

2026-09-21時点の旧pipelineが参照したcheckpointは次です。現在の標準設定はassociation契約へ移行しており、この表は過去の配置記録です。

| stage | checkpoint |
|---|---|
| court | `court_detection/hybrid/court-detection-epoch=17.ckpt` |
| ball | `ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt` |
| PLCS | `plcs/real-rgb-meiji-foot-e60-v1.ckpt` |
| BLCS | `blcs/real-rgb-meiji-e60-v1.ckpt` |

## 2026-09-21のKP＋LINE下流移行

[run-court-hybrid-downstream-migration](nodes/court_detection/000031-run-court-hybrid-downstream-migration.md)では、ユーザー指定の残差head checkpointへ共通KP＋LINE推論を接続し、下流もcamera_view_v2へ移行した。8画像のH採用は3例であり、推定完了率の改善や実動画E2E精度は未確立。既定の変更は入力契約の統一であって、旧モデルへの精度優位の証明ではない。B00〜B03の保存alignmentと生成データは再publicationしていない。

以下の既存baseline比較は元のas-of commitに基づく履歴として保持する。当時のpipeline checkpointはMeiji fine-tune版・window128・明示したreference-camera契約を使用した。他会場の汎化、独立正解Hでの誤採用率、下流E2E評価を次の課題とする。

## タスク別の主要な知見と判断保留事項

### Ball Detection

[#934の実clip契約検証](nodes/ball_detection/000019-run-i934-evidence-meiji-clip000.md)で、
Meiji 1 clipの全3cameraに対するnative heatmap・top-K・局所patchの保存とload-only再開が成立した。
単一点が非観測でも検出証拠を保持できる。これは精度比較ではなく、deploy選択は変更しない。
保存契約の検証と、以下のvideo単位holdoutによる検出精度評価を区別する。

[混合FTの初回](nodes/ball_detection/000020-run-i934-mixed-ft-s42-r5.md)は最初のvalidation後、
MDD描画へ正規化済み入力の宣言を渡していなかったため停止した。checkpointとvalidation指標は未保存で、
描画経路修正後の[同条件再実行](nodes/ball_detection/000021-run-i934-mixed-ft-s42-r6.md)は12 epochを完走した。
混合validation F1はepoch 0が最大で、そのcheckpointをholdout比較用に固定した。
後続epochでF1が改善せず、loss低下だけを追加学習の根拠にはできない。
[選択epoch 0のMeiji holdout比較](nodes/ball_detection/000022-run-i934-meiji-holdout-e0-r7.md)では、
observed 28,806 frameのrecallが44.49%から68.00%へ、閾値なしtop-K recallが71.59%から85.11%へ改善した。
一方、採用検出のp95は346.08から386.30 source pxへ悪化し、cam2では低scoreも含むraw p95も悪化した。
欠損低減と誤検出抑制は両立しておらず、2026-09-28時点のdeployはft-e13を維持する。
AI補助注釈・単一video/seed、手首距離既知36.90%という制約があり、3D品質や他sourceの忘却は未検証。
後続のvalidation候補選択は[Ball Refiner](#ball-refiner)へ引き継ぐ。score較正・誤検出抑制・独立testでの確認は残る。

現行deployはfine-tuning版を維持します。[`run-i618-convnext-v2-scratch`](nodes/ball_detection/000010-run-i618-convnext-v2-scratch.md) はTrackNet test F1 `0.7692`、距離 `2.01 px`でoffline評価では上ですが、実clip coverageが`92.0% → 91.1%`へ下がり、`179.9 px`のteleportを1件発生させました。したがって、単一のF1最高値より実動画上の安定性を優先しています。

3DGS augmentationでは、固定checkpoint・split・decodeによる比較基盤 [`run-i618-3dgs-blcs-real-baseline-v1`](nodes/ball_detection/000011-run-i618-3dgs-blcs-real-baseline-v1.md) が整備されています。simple-sphereを1/12混合したtreatmentは単一seedのgame9で`+0.018454 F1`でしたが、残りseedとgame10 final testが未完了のため、効果は確立していません。

[Meijiの全scene診断](nodes/tennis_scene/000009-run-tennis-scene-meiji-raw-ball-baseline-20260923.md)ではscene/7動画の構造・decodeは成立したが、Ball欠損と非物理的3D軌道が大きかった。[保存前処理の照合](nodes/ball_detection/000018-run-ball-checkpoint-normalization-meiji-20260923.md)で、公開RGB APIとcheckpointのImageNet正規化の接続漏れを確認した。修正はdataset前処理と実model入力が完全一致し、Meiji選定窓の大誤検出は減ったが、recall改善は一様でなくTrackNet 8frameの4px一致数は5→4だった。前処理復元と精度向上を同一視せず、次は修正後の全区間GPU・3D・動画を再評価する。

[修正版の単発scene](nodes/tennis_scene/000010-run-tennis-scene-meiji-corrected-pipeline-20260923.md)と[独立dataset生成](nodes/tennis_scene/000011-run-tennis-scene-meiji-corrected-dataset-20260923.md)は完了し、両sceneの構造と全14動画の全frame decode、既存SLCS reader受理を確認した。Courtは全区間で成立したが、Ball欠損は62.5/33.9/34.0%、3D ballの負高さ50frame・最大412m/s、PLCS/GVHMR整合残差が残る。窓境界不整合の証拠はなく、2D観測/pose mask急変が異常と同時にある。scene公開の成立を高品質教師や3D精度保証とみなさず、次は観測の同一性・可視性の安定性と独立3D評価を分けて検証する。

### Ball Refiner

[#935の教師・既存文脈監査](nodes/ball_refiner/000001-run-i935-data-audit-r2.md)で、
全storeのsplitを保持し、observed位置教師と明示的out_of_frameの存在負例を分けられた。
空frameと推定・unknownはamodal負例にしない。確定負例はchatに偏り、Meijiだけでは存在較正を判断できない。
既存Meiji pose/courtは一部しか揃っていないため、文脈なしpilotを先に準備し、full比較前に生成を完了させる。
未生成をmask欠損へ置き換えず、camera-local KP14と明示的なViTPose score変換を使う。
[凍結ft-e13証拠cache](nodes/ball_refiner/000002-run-i935-evidence-ft-e13-trainval-r3-20260928.md)はtrain/val全frameの生成・checksum/PTS/局所patch読込まで成功した。
[文脈なし時間MDN pilot](nodes/ball_refiner/000003-run-i935-detector-only-ft-e13-s42-r4-20260928.md)は12 epoch・3,000更新を完走した。
同じ選択用validationの観測frameでは、detector argmaxより平均・p95誤差が減る一方、中央値・20px recallが悪化した。
学習接続の成立と精度改善を区別し、detector deploy継続の判断は変えない。
[未較正分布の診断](nodes/ball_refiner/000004-run-i935-calibration-hdr-ft-e13-r5-20260928.md)では、
較正側の6時刻clip群でも平均誤差の改善と中央値/20px recallの退行が同時に見られた。
人工証拠欠損で領域は広がるが、90/95% HDRのcoverageは約81/86%に留まり、分布の裾の過信が残る。
この6群のbootstrapは探索的で、実RGB遮蔽や独立testへの一般化の証拠ではない。
次は補正を別run・較正側のみでfitし、同一母数の文脈生成・ablationと点精度の退行も検証する。
[元動画pipelineの接続監査](nodes/ball_refiner/000005-run-i935-pipeline-ft-e13-r7-20260928.md)では、
270 frameの全GMM保存と別プロセスのload-onlyが成立し、同じ検出証拠からのCPU再計算も小さな数値差で一致した。
専用recipeの接続証拠であり、未較正pilotのdeploy採用や、JPEG学習cacheとの精度同等性を示さない。
最終test・RGB遮蔽対照・full文脈/ablation・標準sceneの3D入力切替は未検証。存在較正はMeijiの正例だけから結論しない。
[3sourceの文脈pilot](nodes/ball_refiner/000006-run-i935-context-fullframe-pilot-r9-20260928.md)は、
import可能な古いDINO拡張のbackend dispatchで停止し、完了clipは0だった。
[run専用再ビルドの再試行](nodes/ball_refiner/000007-run-i935-context-fullframe-pilot-r10-20260928.md)では
3source・561frameのCUDA生成と別プロセス読込が成功した。
画像監査で観客・隣接court人物の混入とchatの視点変化・累計60trackを確認し、chatのcourtは実行済み欠損だった。
有効poseの存在をプレー中の人物のrecallや文脈の有効性と同一視しない。
[最長3sourceの分割probe](nodes/ball_refiner/000011-group-i935-context-shards-r12-probe.md)も全frameの生成・読込が成功し、18分以内で完走した。
Meijiのcourt有効点には目視のずれ・対象コートの曖昧さがあり、chatのcourt欠損も続く。保存成功を文脈品質の保証としない。
2026-09-29の[#964のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5889433860)で人物契約が変わるため、
旧全clip生成は停止した。r13 shardsは保持して学習には使わず、#964完了まではperson/pose生成・延長・学習を行わない。
[最初のvalidation比較](nodes/ball_refiner/000012-run-i935-val-candidate-recall-r14-20260929.md)はCUDA device index不足で推論前に失敗したが、
[修正版の3checkpoint比較](nodes/ball_refiner/000013-run-i935-val-candidate-recall-r15-20260929.md)は完了した。
Meiji video_000の候補recall@8はmixed-e11が最大で、閾値F1によるr6のepoch 0選択とは逆転した。
全camera・chat val・TrackNet game9の候補recallもe11が最大だが、候補内での順位誤りとTrackNet top-1の退行は残る。
[全epoch保存の混合FT再学習](nodes/ball_refiner/000014-run-i935-mixed-ft-val-recall-s42-r16-20260929.md)はepoch 10中にCUDA unknown errorで失敗した。
保存済みepoch 0–9のMeiji val recall@8を照合し、2026-09-29のユーザー判断どおり単独最大のepoch 9を選択した。
peak allocatedは8 GiB cap未満で、directiveに従いWSL2/driver層の障害として扱うが、根本原因を断定しない。
候補recallはepoch 4以降の上積みが小さく、epoch 10–11の再開・延長は行わない。threshold F1との順位逆転も再現した。
[epoch 9の新cache回収](nodes/ball_refiner/000015-run-i935-evidence-mixed-e9-trainval-r17-20260930.md)で全329 clip / 145,767 frameのhash・読込が一致し、Meiji候補recallはbf16 validationとcamera別でも0.13 pp未満の差だった。
[同条件pilot再学習の回収](nodes/ball_refiner/000016-run-i935-detector-only-mixed-e9-s42-r18-20260930.md)は12epoch/3,000更新、560 paired NPZのhash・母数・全既存集計が一致した。
新pilotはMeiji全cameraで旧pilotよりobservedの裾誤差を抑え、新detector単体に対してもp95を改善するが、中央値の精密定位は劣る。
TrackNetの観測位置は退行し、chatは中央値が悪化して裾だけ改善。Meiji較正側のgap HDR95 coverageも約86%に留まり、較正済みとは扱わない。
detectorの一様gap密度によるcoverage=1は全画面領域の自明な結果なので、coverageと面積を併記し、位置誤差を公平な比較とする。
存在/位置の教師がない層はN/A。
[典型frameの精度診断](nodes/ball_refiner/000017-run-i935-precision-variants-s42-r20-20260930.md)では、
正しいdetector候補からrefiner平均が系統的に右へずれ、同じ偏りが他sourceの中央値退行にも現れた。
絶対座標headの平均はほぼ候補peak上になく、epochで偏りの向きが反転するため、
格子解像度やsigma床だけよりも平均parameterizationと未収束/揺れる最適化が主要な候補となる。
run20 directiveに従いcourt-only先行案を保留し、同一recipeの長期化と候補を保持する平均の比較を先に行う。
run20のGPU比較は116秒で監視walkのFileNotFoundErrorにより停止し、checkpoint/val結果は得られなかった。
モデル精度による棄却とは扱わず、[消失競合を修正した同条件retry](nodes/ball_refiner/000018-run-i935-precision-variants-s42-r21-20260930.md)で
事前宣言した3案を比較し、全108checkpointと420val NPZを回収した。
候補残差12kはdetectorより各source/camera/halfの位置誤差を改善し、長期化だけより典型精度がよい。
一方、detector誤り件数で選んだ厳しい270frameでは20px成功率が退行し、個別の失敗は残る。
ただしcalibration halfの観測HDR90/95は0.80/0.85、人工gapでも0.84/0.88で過信が残る。
[2026-09-30のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5908081470)で候補残差headを基準設計に採用した。
[共分散だけのclip交差検証](nodes/ball_refiner/000019-run-i935-covariance-loco-s42-r23-20260930.md)では、
calibration halfのOOF observed HDR90/95が0.80/0.85から0.86/0.89へ改善しNLLも下がった。
ただし面積は約1.8倍、HDR50は過大被覆、人工gap/他sourceのNLLは悪化し、裾の過信も残る。
配布用倍率1.8125と全K4 residual bankを明示hashで保存し、#936の旧bankは対照として残す。
bank作成frameは配布倍率のfitと重複するため、OOF性能と区別する。
[seed44再試行](nodes/ball_refiner/000021-run-i935-seed44-retry-r24-20260930.md)は資源上限内で完了したが、[事前10比較](nodes/ball_refiner/000023-run-i935-seed-reproduction-r25-20260930.md)は9/10で不合格。seed44の人工gap NLLだけがabsolute_12kより悪い。位置分位点は両追加seedでe9 top-1を上回るが、これを全条件の再現成功とは扱わない。
[e9/anchored seed42/固定倍率の明示pipeline option](nodes/ball_refiner/000022-run-i935-pipeline-candidate-r24-20260930.md)の[元動画check](nodes/ball_refiner/000024-run-i935-source-check-retry-r25-20260930.md)では、3camera各270frameのexecuteとfresh-process loadが完了し、全保存配列はbit一致した。終了コード1は全phase後のstrict field診断であり、実行失敗ではない。
[固定BゲートのGT比較](nodes/ball_refiner/000025-run-i935-source-b-gate-r26-20261001.md)はpooled p90が+46.34 px悪化して許容+5 pxを超えたため不合格。中央値とNLLは許容内だが、run26時点では既定ft-e13＋旧refinerを維持した。[全810frameの切り分け](nodes/ball_refiner/000026-run-i935-source-tail-audit-r27-20261001.md)はframe/PTS・窓・正規化のbugを支持せず、中間720p縮小とJPEGによる入力差が候補・成分選択に増幅されることを支持する。同じCPU/pipelineでcam2を再encodeするとp90と最大成分選択がcacheへ戻った。pooled差は連続block bootstrapで0を除外できず、短い末尾区間に依存するため一般化は未確認。固定gateを変更せず、入力経路の整合・MP4証拠の再学習・既定維持の選択肢と費用を提示し、対策の選択は保留した。
[追加ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5912616143)どおり、seedの9/10 FAILを保持したまま再現は十分と扱う。今回Bを止める理由はsource精度のp90であり、seed失敗やstrict診断へ置き換えない。固定倍率の三seed診断にはgap/TrackNet NLLの悪化とcalibration halfの過信が残る。
[2026-10-01のユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5921216642)で、B FAILを保持したままe9＋anchored seed42＋固定倍率の既定化と、refiner後のconfidence選別を採用する方針へ進んだ。mp4直接入力を維持し再学習しない。#964完了前のcontext着手も許可された。[run28の積み直し・資源監査](nodes/ball_refiner/000027-run-i935-context-budget-r28-20261001.md)で#964の人物既定を取り込んだが、全329 clipの見積22–33時間が4時間枠を超えるためcache jobは登録しなかった。[run29](nodes/ball_refiner/000028-run-i935-confidence-r29-20261001.md)で既定切替・標準scene refinerを追加し、clip_000を除く保存済みMeiji valで存在確率と全GMMの90%包含楕円面積の規則を固定した。保持frameの誤差は低下したがcache入力での選定結果であり、mp4への一般化は未確認。consumer配線は完了し、同じ欠測maskをside・幾何・三角測量へ渡す。安全benchは未完了。証拠のない区間の改善と文脈ablation、test評価も未完了。

### Court Detection

[Meiji全frame処理の時間分解](nodes/court_detection/000033-run-court-meiji-hybrid-cpu-profile-20260923.md)では、3030frameのCourt工程が約116分だったのに対し、3cameraの各1frameでもCPU hybrid geometry単体が1.77–2.58秒を要した。GPU推論だけの所要時間とは扱わない。精度評価と並行して、同じframeごとの推定契約を保つCPU後処理並列化・GPU batch化を検証する価値がある。静止frameの複製や間引きによる結果変更とは区別する。

KP14 detectorは実pipelineで利用可能な水準ですが、`1.708886 px`は`test_dataloader`がvalidation dataを読む条件の再評価値であり、独立testではありません。次の品質更新にはrecording-disjoint test、geometry valid率、line support、処理時間、PLCS / BLCSへのE2E影響が必要です。

court segmentationはKP14とは別契約です。[`group-i524-dinov3-ssl-court`](nodes/court_detection/000005-group-i524-dinov3-ssl-court.md) では凍結backbone条件で非SSL `0.517 mIoU`からSSL `0.800 mIoU`へ改善しましたが、KP14 deploy modelの置換根拠にはなりません。点・線共同postprocessは固定した保存予測4枚に対するdiagnosticまで成立しましたが、held-out人手GT付きの正式なpostprocess-only benchmarkはまだありません。

### PLCS

従来deploy（上記as-of commit）は、split trunk、H=0/S=6、position weight 8、補助pose loss無効、court KP14というpipeline互換recipeです。過去のablationでは、positionにはS6/H0、rotationには別の容量配分が有利であり、単一構成が全目的を同時に最適化しないPareto構造が確認されています。[`group-i545-loss-head-tuning`](nodes/plcs/000067-group-i545-loss-head-tuning.md) のposition frontier `0.166 m`は有力ですが、現行KP14 deploy以前の契約なので直接置換には使いません。

canonical poseでは、[`run-plcs-canonical-temporal-decomp-beta01-noaug`](nodes/plcs/000087-run-plcs-canonical-temporal-decomp-beta01-noaug.md) が平均pose固定から入力依存motionへの移行を確認しました。canonical MPJPEは`0.091136 m`、motion amplitude ratioは`1.174967`、centered Pearsonは`0.795146`です。一方、high-frequency fractionは予測`0.391067`に対してGT `0.068930`であり、motionを復元する代わりにjitterを過剰生成しています。position / rotation headは未学習なのでdeploy精度との比較には使いません。

reprojection lossは一方向な改善ではありません。[`group-plcs-multiview-axial-reprojection-loss-w1-v4-t128`](nodes/plcs/000090-group-plcs-multiview-axial-reprojection-loss-w1-v4-t128.md) のV=4/T=128条件では、position `1.386235 → 1.352761 m`、angle `66.968224 → 63.600704°`へ改善しましたが、0.5 m以内率は`0.118984 → 0.088828`へ悪化し、X誤差と分散も増えました。複数seedとweight sweep前にdefaultへ採用しません。

camera-view v2のreference selectorも決着していません。PLCSではreferenceがpositionとID switchesで良い一方、selector-zeroがY-sign、heading、presenceで良く、指標ごとに優位が逆転しました。このselector比較単独ではv1からの移行根拠になりません。2026-09-21のpipelineは上記の入力契約移行によりv2へ変更しています。

### BLCS

single-ballではmultiview deployが単眼親よりposition `1.845 → 1.065 m`、endpoint `3.408 → 2.025 m`へ改善しました。ただしcamera presetも変わるため、改善全量をview数へ帰属できません。

physics priorはaccuracyとsmoothnessのtrade-offです。[`group-i593-physics-prior`](nodes/blcs/000004-group-i593-physics-prior.md) のftCは実clip jerkを`0.280 → 0.106`へ改善しましたが、in-distribution positionを`1.845 → 1.947 m`へ悪化させました。そのため機能は残してもdefault checkpointは置き換えていません。

track-query architectureの[`group-i786-normv2-large-cuda-ablation-eb32`](nodes/blcs/000019-group-i786-normv2-large-cuda-ablation-eb32.md) では、positionはA `3.522323 m`、identity continuityはB `17.04 ID switches`、presence / lifecycleはD `0.972115 F1`・birth/death `4.43 / 5.36 frames`が最良でした。単一の総合勝者はありません。また、この群は旧versioned-v2 runtimeで、現行mainとloss beta・artifact schemaが異なるためhistorical family evidenceとして扱います。

[`group-i801-reference-selector-ablation`](nodes/blcs/000025-group-i801-reference-selector-ablation.md) のmatched BLCS比較では、selector-zeroがreferenceよりposition `3.721058 < 3.813505 m`、Y-sign accuracy `0.877656 > 0.871094`でした。第三RoPE軸によるreference明示の追加効果は確認できず、production v1を維持します。

観測ベース2D追跡の[`group-i832-blcs-observation-tracking`](nodes/blcs/000031-group-i832-blcs-observation-tracking.md)では、GT lifecycleに依存する旧random-slot入力を、noise後の2D観測からdeterministicに対応付ける入力へ変更して比較しました。3 runは同一の`blcs/multi_object` split、`blcs_track_query`（Q=4）、seed `832`、100 epoch、FP augmentation無効で揃えています。

| run / association | position error (m) ↓ | presence F1 ↑ | ID switches ↓ | duplicate active tracks ↓ | missed GT frames ↓ |
|---|---:|---:|---:|---:|---:|
| [legacy random slot](nodes/blcs/000028-run-i832-blcs-legacy-slot-baseline.md) | 5.434213 | 0.878670 | 0.64 | 80.22 | 26.76 |
| [conservative（距離閾値0.04・保持2 frames）](nodes/blcs/000029-run-i832-blcs-tracker-conservative.md) | 5.430877 | 0.881151 | 0.66 | 86.19 | 24.63 |
| [permissive（距離閾値0.10・保持8 frames）](nodes/blcs/000030-run-i832-blcs-tracker-permissive.md) | 5.435566 | 0.878761 | 0.60 | 83.53 | 25.32 |

表は各runの`metrics.json` / `diagnostic_metrics.json`に基づくtest実測値です。conservativeはposition / F1 / missed GT framesで最良ですが、baseline比のposition改善は約`0.00334 m`にとどまり、ID switchesとduplicate active tracksは増えています。permissiveはID switchesが最良で、F1もbaselineを僅かに上回りますが、positionとprecisionは悪化しています。全runが100 epochを完走し、曲線に発散やNaNは報告されていません。低いID switchesだけでtracking failureが解消したとは判断しません。

#832ではconservativeを運用選択として採用し、[現行association設定](../src/tasks/blcs/configs/data/_observation_tracking.yaml)も`max_distance=0.04`、`max_missed_frames=2`です。ただし、指標間の許容差・重みと複数seed評価は未確定で、因果的・統計的な優位は未確立です。次はこの条件を基準に重複trackとID切替の増加を検証し、FP augmentationを有効にした条件と実検出入力での評価を分けて確認します。

#832のID switchesはpost-#824の定義です。#643 / #648 / #650などのpre-#824の保存値とは直接比較できません。再現時は各bundleのcommitと`uncommitted.patch`を確認します。特にlegacy baselineとconservativeには未commit差分が保存されており、現行mainの設定だけで同じ学習条件になるとは限りません。

学習性能では、[`group-blcs-compile-training-abba-v4`](nodes/blcs/000014-group-blcs-compile-training-abba-v4.md) によりcompiled実行がsteady-stateで`1.90×`高速、peak CUDA allocatedが`19.6%`減る一方、cold-start込み3 epochでは`2.98×`遅く、break-evenは約18 epochと分かっています。これはtrajectory精度ではなくruntime baselineです。

multi-ballはsingle-ballと別契約です。短clip diagnosticと、[`run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep`](nodes/blcs/000013-run-i648-blcs-lifecycle-v4-large-pointattn32-rope2d-t512-b1-100ep.md) の512-frame lifecycle baselineも、sequence length・data・training budgetが違うため相互に直接順位付けしません。

### SLCS

従来のshared/split DINO比較はtrain / val / testが同じ13 windowを共有するmemorization実験でした。[収録分離パイロット](nodes/slcs/000009-group-slcs-real-rgb.md)で残ったball低分散出力は、[全体版60epoch](nodes/slcs/000072-run-slcs-full-real-rgb-no-ball-smooth-e60-v3.md)では改善し、[同じvalの5入力条件](nodes/slcs/000062-run-slcs-full-no-smooth-gap-rgb-val-v2.md)でtrain平均位置定数との差とRGBの寄与を確認しました。pilotと全体版は更新数・評価clip構成が異なるため、両者の差を単一施策の効果とは呼びません。

[Meiji全件監査](nodes/slcs/000108-run-slcs-meiji-v9-full-qc-v2.md)と[broadcastとの統合](nodes/slcs/000134-run-slcs-real-rgb-full-assembly-v1.md)は完了し、全体版testにはMeijiの別収録もあります。既存基準runの自動testは終端last重みの記録であり、候補選定根拠ではありません。追加探索では自動testを無効にし、validationで採否を決めます。[gap48比較](nodes/slcs/000069-run-slcs-full-real-rgb-gap48-val-v1.md)は全体平均を改善してもbroadcast full/gapが悪化したため基準置換を見送りました。教師は独立実測3D正解ではなく、欠損区間の大誤差と時間的スパイクは未解決です。最新の施策・選定・限界は[実験群](nodes/slcs/000009-group-slcs-real-rgb.md)を参照してください。

なお、[速度整合候補の初回val評価](nodes/slcs/000081-run-slcs-full-real-rgb-velocity-val-interrupted-v1.md)は再起動後に空出力が見つかり採用不可。
その後のユーザーのgoal優先指示でローカル作業を再開し、[新しいval5条件評価](nodes/slcs/000082-run-slcs-full-real-rgb-velocity-val-v2.md)は完走しましたが、位置平均と欠損境界の退行により基準置換を見送りました。Windowsクラッシュの原因は未確定です。

## 結果を解釈するための規則

| 区分 | 用途 |
|---|---|
| **production / deploy** | 現行pipelineが参照するcheckpoint。単一metricの最高値だけでは変更しない |
| **benchmark** | 固定split・decode・metricで新施策を比較する起点 |
| **family** | 特定architecture、loss、data contract内の比較。タスク全体へ一般化しない |
| **diagnostic** | overfit、smoke、canonical-onlyなど、経路や仮説の成立だけを確認する実験 |
| **missing** | held-out評価、複数seed、正式runなどが不足し、基準として使えない状態 |

比較時は次を守ります。

1. view数、dataset、target frame、single/multi-object、metricが異なるrunを直接順位付けしない。
2. deploy判断ではbest checkpoint、実clip安定性、下流E2E、処理時間を単一metricより優先する。
3. 単一seedの小差は確立した効果とみなさない。
4. no-op、作業tree取り違え、failed qualification、holdout rejectを正の証拠へ昇格しない。
5. 現行mainとnormalization、loss beta、runtime、artifact schemaが異なるrunはhistorical evidenceと明記する。
6. associationの生成方法、FP augmentation、ID switchesの定義も比較条件に含める。学習入力の運用設定の採用と、production checkpointの更新は別の判断として記録する。

## 優先して解くべき課題

| 優先度 | 領域 | 不足している証拠 | 完了条件 |
|---|---|---|---|
| S | Court Detection | recording-disjoint test | 固定モデル・解像度で旧H／PROSAC／共同推定のKP距離、geometry valid、line support、失敗・棄却率、wall time、下流E2Eを比較 |
| S | PLCS canonical motion | jitter抑制とmulti-task再導入 | motion相関を保ち、high-frequency fractionを低下させ、position / rotation併用でもmean collapseしない |
| S | BLCS観測ベースtracking | #832が単一seed・FP augmentation無効、重複track増加 | 許容差・重みを先に固定し、同一条件で3 seed以上を比較。position / presence / ID / duplicate / missedを併記し、FP有効条件と実検出入力でも評価 |
| A | BLCS track-query architecture | #786が旧runtime・入力契約 | associationとpost-#824 metricを固定し、A/B/Dをcurrent loss・schema、3 seed以上で再実行してposition / ID / lifecycleのParetoを確認 |
| A | PLCS reprojection | 単一seed・weight 1のみ | weight `0.1/0.3/1.0`を複数seedで比較し、meanだけでなく0.5 m率・軸別誤差・tailを改善 |
| A | SLCS | 入力欠損・位置の裾・時間的スパイクと擬似教師の限界 | 固定valのdomain別・高速区間別で施策を比較し、選定後testと複数seedで再現性を確認 |
| A | Ball 3DGS augmentation | campaign未完了 | 残りseedとgame10 final testを固定protocolで完了 |
| B | multi-person / multi-ball | deploy互換E2E評価が不足 | single-object契約と分離したまま、lifecycle・presence・identityを長sequenceで評価 |

## このknowledge directoryの読み方

- [`summary.md`](./summary.md): 現在の到達点と未解決課題を俯瞰する入口。
- [`nodes/`](./nodes): 各runの設定・metric・考察、および関連runをまとめるgroup nodeの正本。
- [`runs/`](./runs): `repro.sh`、`metrics.json`、`pred_test.npz`、収束曲線などの再現性bundle。
- [`README.md`](./README.md): knowledge graphのschema、登録方法、検証手順。
- [`webui/`](./webui): node間の関係と実験結果をグラフとして閲覧するUI。

このsummaryは、pipeline checkpointが変わったとき、同一契約で再現された重要な結果が追加されたとき、評価契約が変わったとき、またはdiagnostic領域に初めてheld-out baselineができたときに更新します。新runが1件追加されるたびに追記するのではなく、研究上の結論または優先順位が変わった場合に更新します。
