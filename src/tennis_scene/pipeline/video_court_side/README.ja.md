# 固定カメラのcourt sideを動画単位で決める

[English](README.md) · [简体中文](README.zh-CN.md)

`court_side.mode=video_shared` は、同じ動画に属する固定カメラのhalf-turnを一度決めます。校正には `court_kp.mode=video_shared` が必要です。dataset IDとvideo IDは共有コートから引き継ぎます。既定は `court_side.mode=clip` です。

各clipに従来のサンプリング、画素閾値の解像度補正、静止観測の除外、ballだけの仮説検定を適用します。frameはclip IDとclip内の元frame番号で識別します。同じhalf-turn仮説のcostとsupportを各clipの有効な多視点frame数で加重平均し、カメラ対の共通frame数を加算して、従来の閾値で動画を判定します。動画成果物から別の内容hashで参照する単独診断に、各clipの使用frame番号、全仮説の順位、対ごとのframe数、margin、停止理由を保存します。有効frameが0件なら順位は空、marginはnullです。

## 単一clipから使う

[共有コートの設定](../video_court/README.ja.md) と対象の `clip_id` を指定します。保存済みの判定を使うには、`src.tennis_scene.scripts.run_pipeline` に次の設定を追加します。

```bash
court_side.mode=video_shared \
court_side.video_shared.action=load \
court_side.video_shared.directory=tennis_scene/court_side_shared/match \
court_side.video_shared.artifact_digest="$SIDE_DIGEST"
```

作成時は `action=build`、`directory`、`request` と `request_sha256` を指定します。両パスはARTIFACT root相対です。requestには、共有コートに含まれる**全clip**について、`court_calibration` と全cameraの `ball_points/<camera>` が完成したstoreを列挙します。各 `inputs_digest` は `PinnedSideInputs.from_store(directory).digest` で得ます。sourceと入力の版を固定するため、後続sceneを同じstoreに公開してもpinは有効です。1 clipの動画の例です。

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

clip nodeは、現在のball観測・校正行列・reference cameraを固定した入力と照合します。単独で確定したhalf-turnが共有結果と矛盾したclipは `court_side_video_shared_conflict` で失敗します。単独で未決なら共有結果を使い、停止理由をwarningに残します。`court_side.video_shared.unobservable_policy=fail` を選ぶと単独未決も失敗します。動画全体が未決ならどちらの方針でも失敗します。`run.json` に両方の順位・margin、適用結果と両成果物のdigestを保存し、共有判定のdigestと実行方針をside node・dataset markerのidentityに含めます。

一括処理側は、全clipの入力段をそろえてから最終計画を組みます。clip IDから `PinnedSideInputs` への辞書を `build_video_court_side` に渡すか、設定に従う `prepare_video_court_side` を使います。各clipの `standard_definition` に共有コートと `video_court_side` 成果物を渡します。観測や校正、数値設定が変わった場合は対応する成果物が必要です。clipごとの診断は `artifact.clip_report(clip_id, policy)` で取得できます。
