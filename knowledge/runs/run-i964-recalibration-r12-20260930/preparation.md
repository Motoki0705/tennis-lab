# run 12 準備の実測記録

[事前protocol](protocol.md)は8e8face1、[fit前投稿](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5910654476)。
特徴生成・fit・新dev採点・予約未見評価は、この準備記録の作成時点では未実行。

## 選定結果と不足特徴

旧#933のclip keyとdataset metadataから固定規則で選んだ結果:
|clip|source frames|3camera frames|
|---|---:|---:|
|video_000/clip_003|1052|3156|
|video_000/clip_011|1297|3891|
|video_001/clip_020|1074|3222|
|video_001/clip_000|1316|3948|
|video_002/clip_002|250|750|
|video_002/clip_009|658|1974|

合計6clip / 18camera / **16,941 camera-frame**。
選定全候補、dev/予約除外と元cameraのsource区間非重複、動画・重みhashは
[feature-plan.json](feature-plan.json)。
短いclipの選択は結果で変えていない。未見はmetadataのkey/hash/区間だけで、映像・特徴・人物ラベルを開いていない。

#964の完了済み特徴manifestはrun 3 FT smoke、run 7 CLIP/SOLIDER dev、run 8 KPR dev。
今回の無ラベル集合との一致は**0camera**。旧#933のROI/旧trackerのpose・appearanceは
COCO全画面.30＋新既定trackのidentityを満たさない。したがって18cameraを新規抽出する。
他laneのcache/worktreeは借用していない。

## CPU準備

- AFLinkは[配置記録](aflink-placement.json)の通常ファイル、4,348,705 bytes、公開weightのSHA一致。
  checkpoint root外symlinkではない。重みはgit/GitHubへ配布しない。
- [全pipeline CPU検査](preflight.json): 15資産・28node、全execute、AFLink stateのCPU loadが成功。
  [解決済みCUDA候補設定](qualification.pipeline_config.yaml)は旧association尺度のままであり、
  較正済みYAMLのcommit後にpreflightを再実行する。GPU数値成果物を検証したとは称さない。
- [次runの合格条件](qualification-plan.md)を固定。今回のGPU枠をfull pipelineに流用しない。
- crop maskの差は[別記録](identity-audit.md)。旧結果の「保存特徴からの再現」と
  productionの画像入口の一致を区別する。既定を変更する暫定案は採用していない。

## 通常検証と制限

split選択の結果非依存、dev/予約/人物labelの除外、元source区間の重複拒否、
実extractorとのmask投影一致、古いchecksum拒否、空frame保持、
監視超過時の起動拒否/監視失敗時の自分の子processの終了、既存特徴再生の
**10 tests pass（pytest -n4）**。ruff/mypy/commit hooks、shell構文検査もpass。
最初の再利用監査ではcourt出力class名の誤記で停止し、CourtKPResultへ修正した。
これはfit/推論の失敗ではなく、未公開CPU監査ツールの検証中の誤り。
部分的な自分の出力は別directoryに保持し、成功runと混ぜない。

GPU見積りは実測ではなく60–115分 / peak8–10GB / 出力1–2GB。
1jobだけ、resource=all、外側timeoutはkill猶予込み7200秒、torch allocator7GiB、
GPU全体9.5GBの監視停止（2秒間隔）。host available<6GiBまたはrun出力5GBで停止する。
成功を偽装する小batch再試行や別jobは登録しない。実行はqueue待ちへ渡す。


## 確定した再利用一覧（CPU監査成功）

[reuse.json](reuse.json)に各入力archive/元track/court/sideのpathとhashを固定した。
各modelのhash、DINO全画面.30/800・1333/merge off、pose float32・batch4、
CLIP-ReID、run 11のtracking profile・AFLink・selection/associationの実装hashを確認した。

|clip|cam0|cam1|cam2|
|---|---|---|---|
|video_000/clip_000|元特徴/track再利用|1 rowをmask投影、CPU再追跡が必要|元特徴/track再利用|
|video_000/clip_007|元特徴/track再利用|元特徴/track再利用|元特徴/track再利用|
|video_001/clip_001|8 rowをmask投影、CPU再追跡が必要|元特徴/track再利用|元特徴/track再利用|
|video_002/clip_013|元特徴/track再利用|元特徴/track再利用|元特徴/track再利用|

ViTPoseは全40,531 row、CLIPは有効性が一致する40,522 rowで旧値を使用する。
9 rowは追加推論をせず欠測へ変換し、新archiveの全値roundtripを確認した。
元archive/trackは変更していない。2cameraの再追跡は次runであり、精度はまだ採点していない。

無ラベル6clipのcourt calibration v1 / court observations v2は全3cameraで
元動画hash・設定・checkpoint一致、descriptorと配列の検証/CPU loadが成功した。
固定sideは旧#933と同じ注釈ballの決定が6/6にあり、人物labelを使用していない。
これらを新規人物特徴と組み合わせる。full pipelineでのfresh court/ball/sideとは区別する。

最終feature planの全コード/入口/入力/重み/予約hashとlabel非存在のCPU再検証も
[plan-validation.json](plan-validation.json)でpass。fit/新dev採点は0回。
