# PR #2182 第4回レビューのmajor指摘への対応（親 #1734 から #1751 へ）

第4回の再レビュー（gpt-6.2-astra / xhigh）で、前回のA/B/Cの横断修正は受け入れられた。新たにmajor指摘が4件出た。4件とも直してください。

## 1. 選手選別が同名の別 checkpoint で作った外観特徴を再利用する

CPU実行検証。player_selection.py の process_with_models は、追跡済み特徴を使うかどうかを encoder 名だけで決めます。同名で重みの SHA が異なる2つの ModelSpec と追跡済み特徴を与えると、選別側の factory 呼出しは0回で、結果には tracked_detection_features と記録されました。tracking_encoder_weights と association_encoder_weights を別々に指定すると、選別は前者の特徴を使い、人物対応は後者の重みを使います。追跡特徴と選別側のモデル identity が一致するときだけ再利用し、不一致時の処理をテストで固定してください。

親の判断: **直す。** 追跡済みの特徴は、追跡側と選別側のモデルidentity（重みのSHAを含む）が一致するときだけ再利用する。一致しなければ明示的に再計算するか拒否する（どちらにするかは、選別の仕様に合わせて決め、PRに明記する）。両方の場合をテストで固定する。

## 2. component を差し替えると重みを変えても旧成果物が cache hit する

CPU実行検証。definition.py の standard_definition.add は、overrides で渡した component の settings にクラス名だけを保存します。同じクラスで異なる SHA を宣言する推論 component を順に渡し、同じメモリ上の store で実際の ExecutionPlan.decide を呼ぶと、初回は execute、2回目は cached でした。publication_identity() も標準構成から作るため、差し替えを識別できません。実際に選んだ component の ModelSpec を成果物と完成 marker の両方に反映するか、識別できない差し替えでは再利用を拒否してください。

親の判断: **直す。** overridesで差し替えた推論componentでは、実際に選んだcomponentのModelSpecを、成果物のidentityと完成markerの両方に反映する。反映できない差し替えは、再利用を明示的に拒否する。

## 3. DINOv3 が宣言した repository の外から実装を読み込める

CPU実行検証。dinov3.py の _guarded_hub_load は、初回 import 後に dinov3 の読込元を照合しません。hubconf.py だけがある repository をメモリ上で模擬し、別の DINOv3 を sys.path に置くと、実際の torch.hub.load は外部の package を読み込みました。固定された identity に含まれたのは hubconf.py だけです。この状態では Court と外観 encoder の共有キー・完成 marker が実際の推論コードを識別できません。読み込んだ dinov3 モジュールを宣言済み資産と照合し、外部解決を拒否してください。

親の判断: **直す。** DINOv3を読み込んだ後に、モジュールの読込元（__file__）が宣言したrepositoryの中にあることを照合する。外部で解決された場合は拒否する。

## 4. 計測 hook が初期化・解放の失敗を抑制すると node が成功扱いになる

CPU実行検証。inference_runtime.py の borrow と _release は、モデル操作を model_event の context manager 内で実行します。ValueError を抑制する hook を渡すと、initialize が失敗しても borrow は未初期化のモデルを返しました。release が失敗するケースでも runtime.node は正常終了しました。後者では PlanExecutor が成果物を公開できます。モデル操作の失敗を hook の例外抑制と独立に伝播させ、初期化失敗後の借用と解放失敗後の公開を防いでください。

親の判断: **直す。** モデルの初期化と解放が失敗したら、計測hookの例外抑制とは独立に、その失敗を伝播させる。初期化に失敗した後の借用と、解放に失敗した後の公開を防ぐ。

## 横断確認（前回のA/B/Cに追加する）
- D. 注入経路: overrides、外部からのcomponent差し替え、hub・sys.pathによる外部解決のように、宣言の外から実装や重みが入る経路を洗い出す。各経路でidentityに反映するか拒否するかを、表にする。
- E. 失敗の伝播: hook、context manager、callbackが例外を抑制した場合でも、モデルのライフサイクルと公開の判定が誤らないかを確かめる。
- 前回のA/B/Cの表に、D/Eを追加してPR本文を更新する。

## 共通の条件
- 静かなフォールバックを作らない。GPUの再受入は不要。cache identityが変わる範囲を、PRに明記する。
- CPUの関連テスト、pre-commit、CIを通す。merge・closeは親が行う。

