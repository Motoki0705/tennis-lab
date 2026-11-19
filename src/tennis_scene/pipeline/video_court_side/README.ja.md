# 固定カメラのcourt sideを動画単位で決める

[English](README.md) · [简体中文](README.zh-CN.md)

`court_side.mode=video_shared` は、同じ動画に属する固定カメラのhalf-turnを一度決めます。校正には `court_kp.mode=video_shared` が必要です。dataset IDとvideo IDは共有コートから引き継ぎます。既定は `court_side.mode=clip` です。
`court_side.evidence=player` を指定すると、[選手の足元観測](../../../tasks/court_side/README.md#プレーヤーから判定する)を使います。buildとloadには同じ証拠種別を指定します。既存の動画判定は既定のballで読み込みます。

court_sideに従来の6閾値だけを保存した設定はclipモードで読みます。現行形式では `mode` と `video_shared` を対で指定します。

各clipを従来の格子でサンプリングし、選択した証拠で仮説を評価します。frameはclip IDとclip内の元frame番号で識別します。同じhalf-turn仮説のcostとsupportを各clipの有効な多視点frame数で加重平均し、カメラ対の共通frame数を加算して、その証拠の閾値で動画を判定します。動画成果物から別の内容hashで参照する単独診断に、各clipの使用frame番号、全仮説の順位、対ごとのframe数、margin、停止理由を保存します。有効frameが0件なら順位は空、marginはnullです。

## 単一clipから使う

[共有コートの設定](../video_court/README.ja.md) と対象の `clip_id` を指定します。保存済みの判定を使うには、`src.tennis_scene.scripts.run_pipeline` に次の設定を追加します。

```bash
court_side.mode=video_shared \
court_side.video_shared.action=load \
court_side.video_shared.directory=tennis_scene/court_side_shared/match \
court_side.video_shared.artifact_digest="$SIDE_DIGEST"
```

作成時は `action=build`、`directory`、`request` と `request_sha256` を指定します。両パスはARTIFACT root相対です。requestには共有コートに含まれる**全clip**のstoreを列挙します。各storeに `court_calibration` と全cameraの観測が必要です。

| 証拠 | cameraごとの入力 | `inputs_digest` の取得 |
|---|---|---|
| `ball` | `ball_points/<camera>` | `PinnedSideInputs.from_store(directory).digest` |
| `player` | `player_selection/<camera>` | `PinnedSideInputs.from_store(directory, evidence="player").digest` |

sourceと入力の版を固定するため、後続sceneを同じstoreに公開してもpinは有効です。playerのpinでは、保存済みの選別v2とv3から観測bboxとmaskを読みます。1 clipの動画の例です。

```json
{
  "schema": "video_court_side_inputs_v1",
  "clips": {
    "video_000/clip_000": {
      "role": "data",
      "directory": "match/dataset/videos/video_000/clips/clip_000/annotations/tennis_scene",
      "inputs_digest": "<revision digest>"
    }
  },
  "evidence_clip_ids": null
}
```

`role` は `data` または `artifact`。既定のnullでは全clipを使います。`evidence_clip_ids` に一部のclipを指定しても、全clipの単独診断を保存します。同じ入力のbuildでは成果物を再利用し、`rebuild=true` で再計算・破損修復を明示できます。loadでは固定した判定と診断の内容hashを照合します。

## clipの採否と複数clipからの利用

build/loadは校正に使ったコートのdrift診断の版も照合します。clip nodeは、設定で選んだ観測・校正行列・reference cameraを固定した入力と照合します。単独で確定したhalf-turnが共有結果と矛盾したclipは `court_side_video_shared_conflict` で失敗します。単独で未決なら共有結果を使い、停止理由をwarningに残します。`court_side.video_shared.unobservable_policy=fail` を選ぶと単独未決も失敗します。動画全体が未決ならどちらの方針でも失敗します。`run.json` に両方の順位・margin、適用結果と両成果物のdigestを保存します。共有判定のidentityは証拠種別を識別し、そのdigestと実行方針をside node・dataset markerのidentityに含めます。

一括処理の前段は同じ共有コートと `court_side.mode=clip` で組み、全clipで校正と上表の入力を用意します。clip IDから `PinnedSideInputs` への辞書を `build_video_court_side` に渡すか、設定に従う `prepare_video_court_side` を使います。各clipの `standard_definition` に共有コートと `video_court_side` 成果物を渡します。観測や校正、数値設定が変わった場合は対応する成果物が必要です。clipごとの診断は `artifact.clip_report(clip_id, policy)` で取得できます。

buildの再利用と準備済みbuildの計画作成では、requestのSHA・証拠集合・全clipの現行入力版を照合し、校正と指定した観測の全配列のchecksumを確認します。単独診断にだけ使うclipも対象です。動画判定の明示loadは、指定digestの判定と診断を読みます。

build・loadともに、証拠に選んだ全clipの全cameraに現在の `court_kp.video_shared.unobservable_policy` を適用します。棄却された観測があればclip・camera・drift理由を返して停止します。直接APIでは `court_policy` にこの設定を渡します（既定 `warn`）。証拠集合から外したclipも、自身の実行・公開時には同じコート採否検査を受けます。

clip側の自動cacheと明示loadは、共有digest・コートとsideの両方針・入力依存が一致する共有componentの出力を採用します。共有モードのcourt_sideへのimportとoverrideは拒否します。datasetの完成markerを作成・再利用する際は、clipの採否、scene exportから共有componentへの依存、共有判定で使った証拠の配列checksumを確認します。
