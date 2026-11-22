# PR #2182 第3回レビューのmajor指摘への対応（親 #1734 から #1751 へ）

第3回の再レビュー（gpt-6.2-astra / xhigh）で、前回の5件の修正は受け入れられた。新たにmajor指摘が5件出た。親の判断は各項の末尾に書いた。

## 1. deviceを変更してもBallと人物検出が以前の成果物を再利用する

実行検証。標準設定から同じclipのCPU計画とCUDA計画を作り、資産を固定したメモリ内storeで `ExecutionPlan.decide()` を実行しました。Ballと人物検出はいずれも、モデル宣言のdeviceが変わったのに、CPUで `execute`、CUDAで `cached` になりました。`definition.py` の `ball_detection` と `person_detection` の成果物設定に有効なdeviceが入らず、cache hitでは `borrow()` の共有キー照合まで到達しません。両nodeの成果物キーに有効なモデル設定を反映し、同じstoreでdeviceを変える回帰テストが必要です。

親の判断: **直す。** BallとPerson detectionの成果物キーにも、有効なモデル設定（deviceと精度など）を入れる。同じstoreでdeviceを変える回帰テストを書く。cache identityが変わる範囲を、PRに明記する。

## 2. 外観encoderでSHAを検証したcheckpointと実際に読み込むcheckpointが一致しない

実行検証。`PipelineRuntimeConfig.association_encoder_weights` に別ファイルを指定し、標準計画のfactoryからencoderコンストラクタに渡るpathを捕捉しました。`ModelSpec.weights` は指定ファイルを指しましたが、コンストラクタには既定の `person_vit_clip_reid.pth` が渡りました。`definition.py` は宣言に使ったpathをfactoryへ渡さず、`build_encoder()` がrootと名前から解決し直します。tracking側も同じ構造です。このままでは別の重みで生成した特徴を、指定した重みのSHAで共有・保存できます。宣言と読込で同じ解決済みpathを使うか、不一致を読込前に拒否してください。

親の判断: **直す。** 宣言と読み込みで、同じ解決済みのpathを使う（factoryには、宣言に使ったpathを渡す）。trackingの側も同様にする。

## 3. 共有Courtを使うと外観DINOv3の実装変更を完成markerが見逃す

実行検証。`video_shared` とDINOv3外観encoderを選んだ標準計画で、固定の共有Courtと内容が異なる2つの外観実装資産を使って比較しました。`player_association` の資産SHAは変わりましたが、`TennisSceneOrchestrator.publication_identity()` のdigestは同一でした。`enabled_model_assets()` が外観encoderについて列挙するのは重みだけです。既存の `annotation.json` があると `completed_annotation()` が先にclipをskipするため、node側の検査は働きません。完成markerにも外観モデルが宣言した実装資産を含める必要があります。

親の判断: **直す。** 完成markerの資産の列挙に、外観encoderが宣言した実装資産（DINOv3の実装）を入れる。

## 4. dataset実行中のcheckpoint置換で、実際の推論と完成markerに別のSHAが付く

実行検証。資産読み取りと公開先をテスト用に置き換え、`generate_pseudo_annotations()` に渡すBallのSHAをA、clipの標準計画作成時に読むSHAをBにしました。実行計画はBを採用し、公開処理にはAが渡り、clipの結果は `generated` になりました。dataset入口は全clipの前に一度だけpublication identityを取得し、各clipで束縛した資産との一致を公開時に検査しません。後で重みをAへ戻すと、Bの成果物をAの完成品として再利用できます。公開するmarkerをそのclipの実行計画に束縛し、両者が食い違う場合は公開前に停止してください。

親の判断: **今回は直さない（対象外）。** 実行中にcheckpointを置き換える操作は、前回の4（実行中のPythonソースの追加）と同じ種類の、実行中に資産を編集する運用である。PRの「既知の制限」に、1行で追記する。

## 5. DINOv3のpackageがディレクトリsymlinkの場合、異なる実装を同じモデルとして共有する

静的推論。`src/utils/models/loading/dinov3.py` の `dinov3_source_assets()` は `repository.rglob("*.py")` で実装を列挙します。Python 3.11のこの走査は配下のディレクトリsymlinkを辿りません。一方、`hubconf.py` の `from dinov3.hub.backbones import ...` はリンク先を読みます。同じ `hubconf.py` と重みを持つ2つのrepositoryで、`dinov3/` が異なる実装へのリンクなら、共有キーも `_guarded_hub_load()` の実装digestも一致し、後のnodeが先の実装を使えます。実行中のソース編集は不要です。実際のimport対象をidentityへ含めるか、検証できない配置をモデル生成前に明示拒否してください。

親の判断: **直す（明示的な拒否でよい）。** repositoryの中のディレクトリsymlinkなど、走査で辿れない配置を、モデルの生成前に明示的に拒否する。

## レビューの往復を止めるための横断確認
これまでの指摘は、次の3種類に集中している。修正する前に、各種類を、推論モデルを持つ全node（court・ball・person detection・tracking・pose・association encoder・gvhmrなど）について洗い出す。結果を、種類×nodeの表にしてPR本文に載せ、漏れをまとめて塞ぐ。
- A. 成果物キー（cache identity）に、有効なモデル設定（device、精度、checkpoint、実装資産）が入っているか。
- B. 宣言した資産（SHAを検証したpath）と、実際に読み込む資産が、同じ解決済みのpathか。
- C. 完成marker（publication identity）の資産の列挙が、nodeが宣言した資産（重みと実装）と一致するか。
実行中に資産を編集する運用（ソースの追加・書き換え、checkpointの置き換え）は、範囲外として既知の制限に記載する。

## 共通の条件
- 静かなフォールバックを作らない。GPUの再受入は不要。
- CPUの関連テスト、pre-commit、CIを通し、PR本文に追記する。merge・closeは親が行う。

