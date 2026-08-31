# B01–B03 3DGS学習・コートアライメント改善 実施報告書

## 1. 文書情報

| 項目 | 内容 |
| --- | --- |
| 対象PR | #836 `Fix complete multi-court alignment selection` |
| 旧PR | #829（#836へ統合したためclose） |
| 対象ブランチ | `feat/alignment-b01-b03-complete` |
| Colab実行ID | `20260829T150257Z-f72f860df71e` |
| Colab入口 | `scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh` |
| 対象シーン | B01、B02、B03 |
| 実施日 | 2026-08-29〜2026-08-30 UTC |
| 最終競合解消commit | `e6415e37`（`main`の`72d01d11`をmerge） |
| Drive成果物root | `MyDrive/tennis_lab/outputs/synthetic_data_generation/alignment-runs/20260829T150257Z-f72f860df71e/` |

本書は、最初の自動コート数推定実装から、B01で正解の3面を受理するまでに発生した失敗、原因分析、アルゴリズム変更、Colab実行の並列化、推論キャッシュ、Google Driveへの保存、PR競合解消および検証結果をまとめたものである。数値は、上記Drive runに保存された最終成果物および同run内の試行履歴から採取した。

## 2. 結論

- B01は正解どおり3面、B02とB03は各1面を推定し、全候補がfitとholdoutの双方で受理された。
- B01の失敗原因は、単純にfitカメラが32台で少なかったことではない。画面・再構成領域の境界にある3面目が部分観測となり、完全な1面を単独で同定するための線分が欠ける一方、従来探索が先に見つけた1面または2面の状態を採用して終了していたことが主因だった。
- 解決策は、より大きい完全状態を優先して検証する探索、共通スケール再fit、候補ごとのfit信頼性検査、そして「独立に十分見える2面が作る等間隔latticeから、ちょうど1面の部分観測候補だけを復元する」限定的な境界補助である。
- 境界補助はコート数を3に固定して捏造する処理ではない。fit/holdoutの双方で意味線、等間隔、共通スケール、非重複topologyを確認し、条件が一つでも欠ければfail-closedする。
- court detectionモデルの推論はCPUではなく`cuda:0`で行った。推論後の地面投影、候補探索、剛体変換fit、fit/holdout検証はCPU中心で行った。
- GPUを物理的・論理的に固定比率で分割してはいない。次シーンの3DGS学習と直前シーンのライン推論を別processとして同じGPU上で重ね、ライン推論終了後のCPU処理をGPU学習の背後に隠す方式を採用した。
- 72視点分の生確率マップをシーンごとに永続キャッシュした。後段アルゴリズムだけを変更した再試行ではcourt detectorを再実行していない。
- reconstruction、最終alignment、全視点heatmap、weighted projection、診断JSON、過去試行、推論キャッシュはDriveへ保存済みで、最終`alignment.json`はローカル検証コピーとDriveコピーのdigestが一致した。

## 3. 要求と実施範囲

今回の作業では、次の要求を同時に満たす必要があった。

1. B01〜B03の3DGS reconstructionとalignmentを完了する。
2. CPU側の幾何処理をGPU学習の裏で実行し、wall-clock上のCPUボトルネックを隠す。
3. VRAMに余裕があるため、ライン推論と3DGS学習を可能な範囲で同一GPU上に重ねる。
4. B01の正解コート数は3面である。正解値だけを強制するのではなく、失敗理由を解析して一般化可能なアルゴリズム修正を行う。
5. court detectionの生推論結果はalignmentロジックを変更しても不変なので、再生成しない。
6. reconstruction、alignment、heatmap、試行ログ、推論キャッシュを作成・更新の都度Driveへ保存する。
7. 実行時間の長い候補探索を並列化する。
8. Colab実行シェルをtimestamp付きの`train`配下に置く。
9. #836のmain競合を解消し、旧PR #829をcloseする。
10. `.spin`の一時変更をPR差分に残さない。

dataset生成（court/BLCS/PLCS）と最終report stageは今回のColab runの対象外とした。pipelineは`alignment`で終了し、`datasets/`および`report/`が生成されていないことをシェル内validatorで検査する。

## 4. 最終実行構成

### 4.1 ステージ構成

各シーンは次の2回のpipeline呼び出しで処理した。

1. `request.from_stage=ingest request.through_stage=reconstruction`
2. `request.from_stage=alignment request.through_stage=alignment`

alignment-only呼び出しを可能にするため、dataset handlerを起動時に即時構築せず、該当stageへ到達したときだけ構築する`DeferredStageHandler`を導入した。これによりalignmentだけの再試行では、BLCS/PLCSなどの重い依存関係やassetを読み込まない。

### 4.2 GPU/CPUの重ね合わせ

シェルの実行順序は概念的に次のとおりである。

```text
時間 ───────────────────────────────────────────────────────────────>

GPU主処理:
  B01 reconstruction
        ├─ B01 line inference ─┐
        └─ B02 reconstruction ─┴─ overlap
                              ├─ B02 line inference ─┐
                              └─ B03 reconstruction ─┴─ overlap
                                                    └─ B03 line inference

CPU主処理:
        B01 projection/search/fit/validation
                              B02 projection/search/fit/validation
                                                    B03 projection/search/fit/validation

Drive:
  B01 reconstruction確定保存 → B01 cache/attempt/alignment保存
  B02 reconstruction確定保存 → B02 cache/attempt/alignment保存
  B03 reconstruction確定保存 → B03 cache/attempt/alignment保存
```

実装上は、現シーンのreconstruction完了直後にそのreconstructionをDriveへ保存し、alignmentをbackground processで開始する。そのまま次シーンのreconstructionをforegroundで開始し、次のalignmentを始める前に直前alignmentの終了を待つ。このため、同時に無制限のGPU jobを起動することはなく、最大でも「次シーンの3DGS学習」と「直前シーンのライン推論」が競合する。

GPU memoryの固定partition、CUDA MPSのquota、MIGは使用していない。Colabの単一L4上で2 processが通常のCUDA allocatorを介して共有する。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`を設定し、断片化を抑えた。したがって「GPU分割」は専有率を保証する分割ではなく、空いている演算資源とVRAMをprocess間で共有する並行実行である。

### 4.3 実測環境と決定性

B01最終診断に記録された環境は次のとおりである。

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA L4 |
| court-line device | `cuda:0` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0` |
| seed | 42 |
| model mode | eval + inference mode |
| deterministic algorithms | enabled、warn-only=false |
| cuDNN benchmark | false |
| cuDNN deterministic | true |
| TF32 | matmul/cuDNNともfalse |
| CUBLAS workspace | `:4096:8` |

同じsoftware/hardware条件での決定性を要求する一方、異なるGPU世代間のbit identityは主張していない。

## 5. 入力と再現性契約

シェルはDriveから入力をローカルColab VMへstageする前後でdigestを検査する。不一致は学習開始前にfail-closedする。

| 入力 | SHA-256 |
| --- | --- |
| DINOv3 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c` |
| court line `court-detection-epoch19.ckpt` | `81914bc58ba08824061b4509f54fcb2637a99b5c505cd5c28780cd4c1e88bfd4` |
| B01 video | `c9608e911f86274a862a289927ff9d0cc587543f836ffbdcad127f8ce61b5d56` |
| B02 video | `035a3e79637583d0794e598808fcdd46aac9d3f8e374599f453718a3d6c8615a` |
| B03 video | `80ec1676b420b05f22fc9c4ed5db9257e1c35b9e9bb9596dd1be3f479c7287ac` |

依存関係は`uv sync --locked`、NHTは公開CLIとtrainer runtimeを検査し、不足時だけ`spin setup-nht --with-sfm-learned`で構築する。DINOv3 submoduleも固定checkoutで初期化する。

## 6. heatmapとfitカメラ数

### 6.1 `weighted-projection.png`は何に使われるか

`weighted-projection.png`は、各fit視点のcourt-line確率を推定地面平面へray projectionし、距離重み付きで集約した2次元可視化である。表示専用の画像から再fitするのではなく、同じ投影処理で得た数値配列と点群を候補探索・fitに使う。画像は、その数値evidenceを人間が確認できるようにrenderした診断成果物である。

B01最終heatmap manifestの主要条件は次のとおりである。

| 項目 | 値 |
| --- | --- |
| 生推論・視点別heatmap | 72視点 |
| aggregateに含める視点 | fit 32視点 |
| holdout | 40視点、aggregateには不使用 |
| probability map | 各視点 `256 x 448` |
| grid spacing | 0.0025 NHT scene units |
| raster shape | `814 x 668` |
| 距離重み | `1 / (1 + (camera_range / 0.35)^2)` |
| reducer | 各視点cell max後、距離重み付きglobal sum |
| aggregate encoding | 正値に`log1p`、99.5 percentile基準のTurbo表示 |

### 6.2 なぜfitは32視点なのか

設定済みprefix 48視点は、4つの固定unitに分かれ、各unitをfit 8 / holdout 4に固定する。したがってfit 32 / holdout 16となる。追加した24視点は既存fitの再選択や再fitには使わず、全てholdout tailに割り当てる。最終的にfit 32 / holdout 40となる。

この構成には次の目的がある。

- 既存48視点キャッシュとcamera ownershipを保持する。
- 追加視点を見てからfit集合を選び直す評価漏洩を防ぐ。
- 候補選択はfitで1回、holdout評価は選択後に1回だけ行う。
- 追加24視点を完全に未使用データとして、時間方向・視点方向の一般化を厳しく確認する。

32視点が原因かを疑い、72視点へ拡張して観測範囲を増やしたこと自体は有効だった。しかし、fit数を単純に48または72へ増やすだけでは、B01の境界にある3面目の欠けた外周線を生成できない。さらにholdoutをfitへ流用すると、候補探索と評価の独立性を失う。最終結果では32 fit視点だけで3候補を選択し、未使用40 holdout視点でも全3候補が閾値を通過したため、今回のB01について「32視点では統計的に不足していた」という仮説は採用しなかった。

## 7. B01失敗の根本原因

### 7.1 観測上の原因

B01には実際に3面あるが、`weighted-projection.png`では3面目が投影領域の境界に位置する。中央2面は外周・service line・center service lineを比較的完全に観測できる一方、端の1面は片側sidelineやfar baselineなどが弱い、または範囲外になる。

従来のstrict identifiabilityは、1候補を他候補から独立に完全なコートとして説明できることを要求していた。3面目については、観測されている対応点のfit residualは良好でも、コート全体の各semantic segmentを単独で十分に覆うことができず、`anchor_level_eligible`などの意味線同定条件で棄却された。

これは「線モデルが3面目を全く検出しなかった」という失敗ではない。3面目のservice/center/片側境界に相当する部分線は存在したが、完全な1面を単独証明するための観測が欠けていた。

### 7.2 探索上の原因

初期の自動コート数探索には、次の問題が重なっていた。

1. 高scoreの1面を選んだ後、その周辺evidenceを抑制するgreedy性が強く、後続コートの線を共有境界ごと消費しやすかった。
2. 先に受理可能な1面または2面状態が見つかると、より大きい完全状態を十分に探索・検証しない経路があった。
3. 共通境界線を持つ隣接コートで、evidence ownershipを単純排他的に扱うと後続候補が不利になった。
4. 3面状態を生成できても、境界の部分観測候補を「独立に完全な1面ではない」として棄却し、最終的に2面状態へbacktrackした。
5. 一時的な50% unexplained-evidence許容は、コート数を増やす代わりに誤候補も通し得るため、安全な根本解決ではなかった。

### 7.3 「既知の3面」を設定するだけでは解決しなかった理由

B01が3面であるというground truthは最終validatorの期待値として使ったが、探索結果に3面を無条件で注入してはいない。既知数を3にしても候補の幾何と意味線が不正ならvalidatorは失敗する。実際、3面を期待する途中試行でも出力は1面または2面となり、strict validationで失敗した。

## 8. 改善アルゴリズム

### 8.1 weighted residual探索

各fit視点の確率mapを地面へ投影し、line probabilityとcamera proximityを掛けた正の重みを持つ点群を作る。orientation bandとcenter tileごとにcourt templateを最適化し、次を満たすproposalだけを保持する。

- template scoreが最低値以上。
- scaleが物理範囲内で境界へ飽和していない。
- 既選択候補とminimum center separationおよびfootprint非重複を満たす。
- 共通scaleへ再fit可能。
- 元evidence総量の5%以上を新たに説明する。

各depthで単一greedy状態だけを残さず、最大128状態のfrontierを保持する。B01最終実行では2 orientation band、42 center tileを使い、9,450 tile-stateを探索し、5,418を幾何的に不可能として早期除外した。396 proposal、176 complete stateを得た。

### 8.2 最大の検証済みcomplete stateを優先

候補数の多いcomplete stateから順に、共通scale再fit、候補信頼性、whole-court validationを行う。単にscore最大の1状態を採用せず、ranked stateを順にrefineし、失敗理由を保持して次へ進む。

B01では18状態をrefineし、先行17状態を棄却した後、rank 17の3候補状態を採用した。5面・4面などの過剰状態は複数候補がfit-unreliableであり、2面状態より候補数が多いという理由だけでは受理されない。

refinementは`ThreadPoolExecutor`で最大12 workerを使用する。proposal探索で使う大きなNumPy/SciPy配列をprocess間copyせず共有でき、最終結果は元rankで決定するため、thread完了順に依存しない。

### 8.3 共通scaleとtopology

同一施設内のコートは同じ物理寸法を持つため、候補ごとのnative scaleを共通scaleへ再fitする。B01の最終common scaleは`0.05667475124371754 NHT scene units / metre`、候補間の最大相対偏差は約1.05%で、許容約7.29%を十分下回った。

B01の候補中心間距離は次のとおりで、全pairのfootprint overlapは0だった。

| pair | 中心間距離 |
| --- | ---: |
| court-000 – court-001 | 11.1527 m |
| court-001 – court-002 | 11.1154 m |
| court-000 – court-002 | 22.2681 m |

隣接間隔が約11.1 mで揃い、外側2面の距離がその約2倍である。

### 8.4 boundary-lattice-assisted identifiability

3面状態で2面だけがstrictに独立同定でき、残る1面が境界部分観測の場合に限り、次の手順で補助する。

1. strictに信頼できる2候補が存在することを確認する。
2. outlierがちょうど1候補であることを確認する。複数の不完全候補は補助しない。
3. 2候補の中心・向き・共通scaleから等間隔lattice上の欠けた中心を算出する。
4. 補助中心が投影bounds内にあり、既存候補と重複せず、物理的な中心間隔を満たすことを確認する。
5. 補助後の候補に対しfitとholdoutを別々に評価する。
6. 両partitionで次の意味線条件を確認する。
   - longitudinal/transverseそれぞれのminimum camera count
   - center service line
   - singles/doubles sidelineのnested構造
   - baseline
   - service line
   - 他2面からのequal spacing
7. 共通scale、全pair topology、全候補のresidual閾値を再検査する。

B01では`candidate-002`だけがこの補助対象となった。候補中心はおよそ`(-0.293, 1.019)`から`(0.00694, 1.03196)`へ補正された。診断JSONには`lattice_assisted_candidate_ids=["candidate-002"]`と、acceptance mode `boundary_lattice_assisted`が明示される。

この候補のwhole-template coverage診断は、境界欠損のため単独では低い。一方で、選択対応点のresidual、意味線の部分構造、2面からの等間隔、fit/holdoutの再現性は全て合格した。補助を通常のstrict acceptanceと区別して永続診断へ残すことで、「完全に観測された」と誤表示しない。

### 8.5 最終B01探索統計

| 指標 | 値 |
| --- | ---: |
| 利用可能camera | 266 |
| 選択camera | 72 |
| fit / holdout | 32 / 40 |
| 元fit投影点 | 19,936 |
| 説明点 / residual点 | 10,766 / 9,170 |
| 説明weighted evidence | 60.91% |
| 候補別説明率 | 24.07%、14.81%、22.04% |
| complete state | 176 |
| refinement試行 / 棄却 | 18 / 17 |
| 選択rank | 17 |
| 最終候補数 | 3 |
| stopping reason | `no_additional_complete_court` |

過剰な4面・5面状態は候補信頼性で棄却され、最終3面の後も追加の完全な1面を説明できなかったため停止した。コート数は期待値の強制ではなく、この探索結果から3となった。

## 9. 試行経緯

Driveには成功出力だけでなく、置換前のalignmentと試行ログも`B01/alignment-attempts/`へ保存した。主要な時系列は次のとおりである。

| UTC | 試行 | 結果・判明事項 |
| --- | --- | --- |
| 16:22 | cached | 既存推論を使ったstrict whole-court validationが失敗。再推論以外の問題であることを確認。 |
| 16:38 | prefix72 | 72視点へ拡張したが1面。視点総数だけでは解決しないことを確認。 |
| 17:04 | holdout-tail | fit 32を固定し追加24をholdoutへ移動。1面のまま。評価漏洩を防ぐpartition契約を確立。 |
| 19:14–19:15 | known-three / strict-residual | B01=3を期待して検証したが出力は2面、またはstrict rejection。数を指定するだけでは不足。 |
| 19:53 | monotone refinement | refinementを単調化したが1面へ退行。greedy basin依存を確認。 |
| 20:26 | fit backtracking | fitで棄却された状態から次rankへbacktrackする経路を追加。長時間化が顕在化。 |
| 20:41 | parallel fit backtracking | 候補検証を並列化。まだ3面状態のsemantic gateを通せず失敗。 |
| 20:59 | shared lines | 隣接コートの共有線を考慮。3面proposalは改善したが独立同定gateで失敗。 |
| 21:17 | identifiability diagnostics | longitudinal/transverse offset levelとrejection reasonを詳細化。3面目の部分観測を特定。 |
| 21:36 | candidate fit filter | 明らかにfit-unreliableな過剰候補を早期除外。 |
| 21:41 | diagnostic roundtrip | rejection診断のserialize/deserializeを検証。 |
| 21:47 | root filter | 2面を安定受理。3面目だけが残ることを再確認。 |
| 21:52 | residual refinement | residual上の再探索を追加したが3面目のstrict identifiabilityを満たせず。 |
| 22:13 | 12-worker試行 | module pathの入力ミスで即時失敗。成果物は変更していない。 |
| 22:14 | largest complete set / 12 workers | 正しい入口で再実行。探索時間を短縮したが出力は2面。 |
| 22:26 | rejection diagnostics | 3面状態が`reliable=2/3`であることを永続診断に記録。 |
| 22:38 | lattice repair v1 | lattice補助を導入したが補助後候補のgateが厳密すぎ、2面。 |
| 22:46 | lattice gate diagnostics | `center_out_of_bounds`と`candidate_fit_unreliable`を分離し、必要な意味線条件を特定。 |
| 23:03 | boundary lattice | 3面を初めて受理。全fit/holdout・scale・topologyを通過。 |
| 23:16 | final | 同条件で再実行し3面を再現。最終成果物としてDriveへ確定保存。 |

最終B01 alignmentは、既存72-view推論キャッシュを使った状態で約3分40秒だった。途中試行の長時間化に対して、12 worker refinementと推論再利用が効いている。

## 10. court-line推論キャッシュ

### 10.1 キャッシュ単位

生のcourt-line probability mapは、alignment後段のprojection、コート数探索、acceptance閾値から切り離した。cache identityは次だけから構成する。

- line checkpoint内容
- DINOv3 backbone checkpoint内容
- model architecture設定
- expected short side
- inference device
- seed

後段の幾何アルゴリズムやexpected court countを変更してもidentityは変化しないため、NPYを再利用する。画像自体のdigestも視点entryに保存し、別画像を同じcamera IDで誤再利用しない。

### 10.2 最終キャッシュ

| 項目 | 値 |
| --- | --- |
| schema | `court_line_inference_cache_v1` |
| cache key | `eea97d5cf0e9453ee15512d091e08e7673c3e752fb70d7a6eaf38a436c03446e` |
| expected / completed | 各シーン72 / 72 |
| probability dtype/shape | float32、各`256 x 448` |
| 保存物 | NPY、閲覧用Turbo PNG、進捗manifest |
| local root | `<scene>/court-line-inference/<cache-key>/` |
| mirror root | `<Drive run>/<scene>/court-line-inference/<cache-key>/` |

各viewは推論直後にローカルへatomic saveし、その後Driveへ`.uploading`経由で置換する。manifestはviewごとに更新するため、途中でColabが停止しても完了済みviewまで復旧できる。Drive側に同一size・同一digestのfileがあれば再copyしない。

main取り込み後のrepository-wide architecture testは、旧dataset ArtifactRefを禁止する目的で`fingerprint`/`sha256`というtokenを一律禁止していた。この検査が今回の「生推論を再生成しないための計算メモ化」も検出したため、line inference cacheの当該2 tokenだけを明示的に例外化した。他のactive module、他の禁止token、固定pose architectureは引き続き検査対象である。

## 11. 最終結果

### 11.1 シーン単位

| Scene | 推定/受理コート数 | fit/holdout | NHT units/m | 元投影点 | 説明evidence | lattice補助 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B01 | 3 / 3 | 32 / 40 | 0.0566747512 | 19,936 | 60.91% | candidate-002 |
| B02 | 1 / 1 | 32 / 40 | 0.0648909977 | 26,539 | 96.49% | なし |
| B03 | 1 / 1 | 32 / 40 | 0.0629350506 | 21,005 | 97.36% | なし |

### 11.2 B01候補別residual

全候補でinlier fractionはfit/holdoutとも1.0だった。これは選択対応点が0.3 m inlier gate内にあることを示し、境界候補の全template線が画面内に見えていることを意味しない。後者は別のwhole-template/semantic diagnosticsで評価する。

| Court | Partition | Camera | Correspondence | RMS (m) | Q95 (m) | Max (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 000 | fit | 17 | 2,652 | 0.1824 | 0.2599 | 0.2878 |
| 000 | holdout | 26 | 4,810 | 0.1864 | 0.2599 | 0.2858 |
| 001 | fit | 29 | 3,021 | 0.1746 | 0.2410 | 0.2645 |
| 001 | holdout | 26 | 3,031 | 0.1712 | 0.2436 | 0.2664 |
| 002 | fit | 19 | 3,016 | 0.1542 | 0.2423 | 0.2558 |
| 002 | holdout | 16 | 2,737 | 0.1657 | 0.2417 | 0.2568 |

### 11.3 B02/B03候補別residual

| Scene | Partition | Camera | Correspondence | RMS (m) | Q95 (m) | Max (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B02 | fit | 32 | 6,400 | 0.1145 | 0.2096 | 0.2632 |
| B02 | holdout | 40 | 8,000 | 0.1106 | 0.2058 | 0.2612 |
| B03 | fit | 32 | 6,400 | 0.1128 | 0.2035 | 0.2584 |
| B03 | holdout | 40 | 8,000 | 0.1210 | 0.2137 | 0.2646 |

### 11.4 acceptance閾値

| 指標 | fit | holdout |
| --- | ---: | ---: |
| minimum camera | 6 | 3 |
| minimum correspondence | 100 | 80 |
| inlier distance | 0.3 m | 0.3 m |
| minimum inlier fraction | 0.9 | 0.9 |
| maximum RMS | 0.3 m | 0.3 m |
| maximum Q95 | 0.3 m | 0.3 m |

3シーンの全候補・全partitionが全閾値を通過した。

## 12. Drive成果物

最終成果物は次に保存した。

```text
MyDrive/tennis_lab/outputs/synthetic_data_generation/alignment-runs/
  20260829T150257Z-f72f860df71e/
    B01/
      reconstruction/
      court-line-inference/<cache-key>/
      alignment-attempts/
      alignment/
    B02/
      reconstruction/
      court-line-inference/<cache-key>/
      alignment-attempts/
      alignment/
    B03/
      reconstruction/
      court-line-inference/<cache-key>/
      alignment-attempts/
      alignment/
```

各`alignment/`には次がある。

- `alignment.json`: machine-readableな最終変換、partition、metrics、layout。
- `court-geometry.json`: 受理コートの幾何。
- `ground-line-map.npz`: 地面平面上のline evidence。
- `diagnostics/evidence.json`: camera selection、探索統計、決定性、cache、scale。
- `diagnostics/candidate-metrics.json`: 候補ごとのfit/holdout、semantic、topology。
- `diagnostics/summary.txt`: 人間向け短縮結果。
- `line-heatmaps/heatmaps.npz`: 数値heatmap archive。
- `line-heatmaps/manifest.json`: view、bounds、encoding、weightの契約。
- `line-heatmaps/views/`: 72視点のraw/weighted PNG。
- `line-heatmaps/weighted-projection.png`: fit 32視点の集約投影。

Driveコピーの主要digestは次のとおりである。

| Scene | `alignment.json` | `weighted-projection.png` |
| --- | --- | --- |
| B01 | `3790373784aac071e26ade00aa667702e5c827b5a1ae4ae7a200c04d0aff70d0` | `8aa2d20cfa4ea9863629de705460cdb90210cde6d09b23a5552976afdde4318a` |
| B02 | `d5d39f905c838c511e4ff8f60119d021b7e4f584f12377141826d2cfeb52d08b` | `b6adad30b4fd65c38c59dfdf692a6e3b020db03e204559849a7b260ff6de60bc` |
| B03 | `c062b28040ba87879ceb15ed3f7dd5fa21ac6ffcc8b5b93437bf248e3f42b005` | `8fe365caeb0b13daff2eb5980e9b7241b350f1dfd5eb95a006c669c19fc510e8` |

3つの`alignment.json`について、Driveとローカル検証コピーのSHA-256一致を確認した。

publicationはColab VMのlocal filesystem上でatomic stage replacementを完了してからDriveへcopyする。Google Drive FUSEをcanonical pipeline workspaceに直接使わないのは、atomic directory replacementのfilesystem semanticsを保証できないためである。Driveへの保存も`.uploading` stagingを使い、未完了directoryを最終成果物として見せない。

## 13. Colab入口と運用

通常実行は次の1コマンドである。

```bash
bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh
```

Drive/GPUを使わず、repository path、入力契約、sceneごとのcommandだけを確認する場合は次を使う。

```bash
bash scripts/colab/train/20260829T150257Z/run_b01_b03_alignment.sh --dry-run
```

シェルは次を検査する。

- Colab/Drive mountと期待root。
- NVIDIA GPUの存在。
- repositoryとcanonical configの存在。
- 全入力のSHA-256。
- locked Python環境とNHT runtime。
- ingest/reconstruction/alignmentの完了status。
- B01の受理数が3であること。
- 72 raw heatmap、72 weighted heatmap、weighted projectionの整合性。
- dataset/reportが生成されていないこと。
- reconstruction保存後でなければalignmentをDriveへpublishしないこと。

既存destinationを無条件上書きしない。新規runはUTC時刻とcommitから一意な`RUN_ID`を作る。alignment再試行時は旧出力を`alignment-attempts/`へarchiveしてから新出力を置く。

## 14. commit単位の変更経緯

| Commit | 変更 |
| --- | --- |
| `7bb553dd` | scene evidenceから正のコート数を推定する初期実装。weighted projectionとsemantic contractsを導入。 |
| `b56d5281` | 上位proposalのrefinement失敗時に次rankを試す。 |
| `6b54d409` | proximity/line probabilityによるweighted search、共通scale refinementを強化。 |
| `f16d7468` | alignment-only Colab run、deferred dataset stage、B01〜B03 shellを追加。 |
| `f72f860d` | deferred handlerのcompositionを公開し、重いstageを本当に遅延構築。 |
| `8bb472c5` | Colab入力・NHT・GPU・checkpoint契約をfail-closed化。 |
| `71d1b4ae` | alignmentを待つ前にreconstructionをDriveへcheckpoint。 |
| `2e679290` | 次シーン学習と直前シーンline inference/CPU alignmentをoverlap。 |
| `17ab2a86` | view単位line inference cacheとDrive mirrorを追加。 |
| `151cec96` | camera prefixを72へ拡張し、追加視点をholdoutへ。 |
| `b0320da1` | 48視点のfit ownershipを固定し、24視点tailで再fitしない契約を追加。 |
| `7b59cbcc` | 最大complete state優先、12 worker refinement、境界lattice補助、詳細診断を追加しB01=3を達成。 |
| `e6415e37` | 最新mainをmergeし競合解消。timestamp shellへ移動し、`.spin`をmainへ復元。 |

## 15. main競合とPR整理

### 15.1 #836競合解消

`main`の`72d01d11`をmergeした際、次の3fileが競合した。

- `src/synthetic_data_generation/pipeline/application.py`
- `src/synthetic_data_generation/pipeline/runner.py`
- `tests/unit/synthetic_data_generation/pipeline/test_runner.py`

解消方針は次のとおりである。

- #836側の`DeferredStageHandler`を維持し、alignment-only runでdataset依存をimportしない。
- main側のBLCS mesh rendererを失わず、遅延`_build_blcs_handler`内でGaussian/meshを設定に応じて選択する。
- runtime requestのauthorityは`from_stage`と`through_stage`を両方必須とし、任意の`source_video`削除もmain側仕様と統合する。
- mainから追加されたportable manifest/court rerun testは、runtime契約を弱めず明示的に`through_stage=report`を与える。
- `.spin/cmds.py`はmainの`cu130`版へ戻し、PRのbase差分から除外する。
- Colab shellはrepository root直下の旧位置から、要求された`train/20260829T150257Z/`へ移動する。

push後のGitHub判定で#836は`MERGEABLE`となり、競合がないことを確認した。

### 15.2 #829

#829は同じ自動コート数対応の旧PRで、#836が完成版とmain競合解消を含むため、#836へsupersedeした旨をcommentしてcloseした。

## 16. 検証

### 16.1 ローカル検証

- alignment、pipeline、NHT subprocess、Colab E2E、scene pipeline E2E: 226 tests passed。
- boundary lattice、partial court、measured sourceの重点回帰: 5 tests passed。
- BLCS unit、NHT alignment、BLCS dataset、BLCS mesh integration: resourceを必要としない35 tests passed。
- 長時間のreal-domain scene pipeline CPU integrationも完了。
- Ruff lint passed。
- Ruff format check passed。
- mypy: changed 38 source files、0 issues。
- shell `bash -n` passed。
- Colab shell `--dry-run` passed。
- `git diff --check` passed。
- PR差分に`.spin/cmds.py`がないことを確認。

ローカルworktreeに`data/synthetic_data_generation/raw/B00.mp4`がなかったため、このfixtureを前提にするconfiguration integration 27件は`FileNotFoundError`で実行不能だった。これはB01〜B03の実装失敗ではないが、未実行を成功扱いにはしていない。

### 16.2 GitHub Actionsで発見した追加問題

main merge後の最初のCIでは、long-tail、scene-pipeline、knowledge-webui、labelは成功した。remainder laneは4,132 passed / 81 skippedの後、repository-wide removed-architecture test 1件だけが失敗した。

失敗理由は、旧ArtifactRef/fixed-pose architectureを禁止するtext scanが、今回必要な`line_inference_cache.py`のcache identity tokenも検出したためである。推論cacheは公開dataset artifact identityではなく、モデルまたは画像が変わったときだけ再推論するための計算メモ化である。そこで当該fileの2 tokenだけをexact exemptionとし、他file・他tokenへの検査を維持した。

## 17. 安全性とfail-closed条件

今回の改善では、B01=3を得るために閾値を一律緩和していない。特に次を禁止・拒否する。

- digestが異なる入力、checkpoint、画像cacheの再利用。
- fitとholdoutのcamera重複。
- holdoutを見た後の候補再選択、再fit。
- 物理scale範囲外またはscale boundへ飽和した候補。
- 共通scale偏差が大きい候補集合。
- 最低5%の新規evidenceを説明しない追加候補。
- minimum center separation違反、court footprint重複。
- 2個以上の部分観測候補をlatticeで同時補完すること。
- 信頼できる2面、等間隔、意味線、fit/holdoutのいずれかを欠くlattice補助。
- B01で3面以外を成功とするColab最終validator。
- alignment-only runでdataset/reportを意図せず生成すること。
- Driveの未完了`.uploading`を最終成果物としてpublishすること。

過去に検討した「unexplained evidenceを50%まで許す」scene固有overrideは削除した。最終設定の最低追加説明率は全scene共通の5%であり、B01の各候補は14.81%以上を説明している。

## 18. 残課題と推奨事項

1. 現在のGPU共有はbest-effortであり、VRAMやSMの割合を保証しない。将来、同時実行時のOOMまたは学習速度低下が問題になる場合は、line inference batch size、CUDA stream優先度、またはMPSを別途測定する。
2. boundary lattice補助は「隣接等間隔のコート」という施設構造を利用する。非等間隔配置や孤立した部分コートには適用せず、strict modeで失敗させる。
3. B01最終3面はfit/holdout residualを通過しているが、境界候補のwhole-template可視率自体が増えたわけではない。将来の撮影・再構成では、端のbaseline/sidelineを含むcameraを追加するとlattice補助への依存を減らせる。
4. fit 32 / holdout 40は今回の評価独立性を優先した設計である。fit数のablationを行う場合は、holdoutを流用せず別の完全未使用test splitを用意する。
5. Drive runは再現・監査用に保持し、`alignment-attempts/`とcacheを削除しない。特に失敗履歴は、将来の閾値変更で過去の誤状態が再受理されないか比較するために有用である。
6. 異なるGPUでのbit identityは保証していないため、hardwareを変えた再実行では数値許容差によるsemantic validationを使い、byte一致だけを成功条件にしない。

## 19. レビュー時の確認箇所

- 実行入口: [`run_b01_b03_alignment.sh`](./run_b01_b03_alignment.sh)
- Colab全体説明: [`../../README.md`](../../README.md)
- camera selection、weighted search、lattice補助: `src/synthetic_data_generation/alignment/evidence_source.py`
- inference cache: `src/synthetic_data_generation/alignment/line_inference_cache.py`
- fit/holdoutとboundary acceptance: `src/synthetic_data_generation/alignment/fitting.py`
- strict contracts: `src/synthetic_data_generation/alignment/contracts.py`
- production閾値: `src/synthetic_data_generation/configs/alignment/production.yaml`
- deferred stage composition: `src/synthetic_data_generation/pipeline/application.py`
- Colab command E2E: `tests/e2e/colab/test_b01_b03_alignment_script.py`
- B01回帰: `tests/unit/synthetic_data_generation/alignment/test_evidence_source.py`

以上。
