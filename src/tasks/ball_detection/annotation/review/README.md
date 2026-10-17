# ball のレビューと教師採用

`BallReviewAdapter` を共通の [`RevisionStore`](../../../base/annotation/editing/README.md) と
`build_handoff()` へ渡す。`BallLabels` は元の prediction receipt、方針、全 native frame/PTS、
注釈対象区間を保持する。初期状態にはモデル座標を教師としてコピーせず、全行を未確認・存在不明にする。
元予測は不変の別 artifact のまま、画像の描画や一括確認に使う。

hybridの不在予測も、初期ラベルは `reviewed=false, presence=unknown` のままにする。
handoffの `prediction=absent_unreviewed` と `review_rows.predictions` は元予測であり、
同じ行の確認済みlabelとは別に表示する。存在logit・heatmap peakの意味は
[予測契約](../README.md#hybridの予測案)を参照する。
不在予測は画像に点を描かず、workerは全画面と前後の文脈を確認する。球が見えれば
`replace` で追加し、判断できなければunknownや未解決の状態を保存する。
点がないことやモデルの低scoreだけで `confirm_absence` を実行しない。

## 方針と状態

`BallReviewPolicy` は一括予測確認の可否、対象frameの確認/解決率、画像の最低倍率、補間の最大秒数を
**すべて明示**する。数値の既定値や本番用の方針ファイルは置かない。production plan の
`ReviewPolicyRef` はpilotの根拠と決定を必須にする。本実装は、未確認frameを教師から除外する
`unreviewed_teacher=exclude` と既存の `observed_only` 教師を扱う。
全frameレビューは既定条件ではなく、確認率の採用値によって決まる。
将来、未確認予測を教師に含める方針を採る場合は、由来とreader契約を含む別の実装が必要になる。

- `reviewed` はworkerによる画像確認の申告。採用時に、そのworkerの実画像入力と照合する。
  画像生成や画像提示だけでこの値を立てない。
- `presence=unknown` は球の存在自体が未確定。`present` は球があると判断した状態で、
  位置が不明なら `unresolved` 観測を保持する。`absent` は確認済みの対象なし。
- 観測はtrackごとに `visible / occluded / interpolated / out_of_frame / unresolved` と、
  native pixel座標・`manual / model_confirmed / interpolation` の由来を持つ。
  occludedの推定位置は任意であり、見えた点には読み替えない。
- 複数球、遮蔽、画面外、notes、cut/hit/bounce/track_change/unknown_breakを保存する。
  旧breakをhit/bounceと推測しない。blocking noteが残るframeは解決済みと数えない。

| 状態（注釈対象frame） | heatmap教師 | MDD座標教師 |
|---|---|---|
| 確認済みの単一visible球 | positive | 対象track |
| 複数visible球 | 複数positive | 除外 |
| 確認済みの対象なし / out_of_frameのみ | negative | 除外 |
| 未確認・存在不明・位置未解決・blocking note | 除外 | 除外 |
| 補間・遮蔽推定を含む | 除外 | 除外 |
| 前後の文脈frame | 除外 | 除外 |

教師maskは `teacher_rows()` から導出し、workerに自由記述させない。全frameを残したまま、
実際の確認範囲、未解決範囲、利用可能な教師を分ける。全対象が除外のclipは採用せず保留する。
heatmap負例だけのclipがMDDの追加座標教師になるわけではない。

## 編集

`BallEdits` のoperationsを `EditCommand.changes` へ渡す。actorは、同じcontextを担当する
`ball_review` workerの `Launch` 参照。batch全体の検証後に共通storeがCASで公開する。

- `replace`: 対象frameの現在のhashを明示して、座標修正、観測の追加/除去、状態、notes、境界を
  更新する。frame/PTS/対象区間は編集できない。解消したnoteも以前のrevisionに残る。
- `confirm_predictions`: 明示した範囲の元予測をvisible球として確認する。方針による許可と、
  各frameを含む画像参照が必要。確認済みframeを一括で上書きしない。
  不在予測を含むbatchは全体を拒否する。確認後に範囲を分け、対象なしには
  `confirm_absence`、誤った不在予測の修正には `replace` を使う。
- `confirm_absence`: 未確認の範囲を対象なし、または指定trackの画面外として確認する。
  各frameの全画面画像が必要で、局所cropから不在としない。
  hybridの不在予測にも同じ画像・worker証拠を要求するため、`ball.review-policy.v1` の
  方針・確認率・倍率・採用条件は変わらない。方針を変える場合は定性確認とユーザー判断を別途行う。
- `interpolate`: 同一trackの可視端点と実PTS比を使う。端点・内部の全frameの確認、方針の時間上限を
  要求し、イベント/ID境界・blocking noteを端点も含めて跨がない。内部の実観測・確定不在を
  上書きしない。端点変更で古くなった補間は拒否し、必要なら同じbatchで再計算する。

画像は共通rendererの `RenderedImage` を使う。clip/元予測/同じrevision履歴、描画点と元ラベル、
native crop・倍率を照合する。確認したvisible点は根拠crop内に必要。引継ぎは最大64frameの
要約・確認/未解決範囲・境界・次操作を返し、見たが未解決のframeは両方の範囲に残る。

## 採用

workerの終了後、`prepare_adoption(..., telemetry, reason=...)` に実行記録を渡す。
参照画像のbytesが応答IDへ結び付いたことと、その画像を見たworkerが各frameの申告者であることを
照合する。別workerの画像確認、生成だけの画像、別版/別clipの画像は根拠にできない。
採用方針の確認率・解決率と教師の有効性を検証し、proof・ReviewEvidence・BallTeacherMask・
AdoptionReceiptを不変保存する。`RevisionStore.seal()` で同じ現行revisionを採用し、以後の編集を止める。
独立QAや追加の画像承認工程はない。

## Worker接続と再試行

`BallReviewWorker` は共通poolのTaskAdapter、`BallReviewServices` は親側の操作adapter。
固定したReviewContextをpayloadとする `ball_review / v1 / review` のWorkRequestを扱う。
操作の共通プロトコル・上限は[review services](../../../base/annotation/workers/review/README.md)に置く。
手順の正本は[worker instructions](worker-instructions.md)で、promptへ固定policy・native寸法・
初期handoffとともに取り込む。workerが指定した描画点を採用せず、元予測またはrevisionから描く。

handoffと少数frameの詳細は `image_proof` を返す。過去の実画像入力まで照合済みなら
`verified`、現workerの終了後に検証するものは `pending_current_attempt`、過去attemptの
証拠欠測は `missing`。欠測frameは保存済み画像の再提示と明示的な置換で再確認する。
モデル記録前のcancelや応答ID競合のreportは履歴に残す。handoffには利用不能attemptの件数と
理由ページの先頭参照を表示し、詳細は最大8attemptずつ`next_page`で辿る。ページはpayloadと
actorを固定した不変artifactで、判断のないattemptも含む。参照の読込み手順はworker instructionsに置く。
行詳細の`image_proof_reasons`にも対応する理由を表示する。そのattemptの判断は再確認まで採用できない。
参照・hashの破損は再試行でも拒否する。
ラベルが既にreviewedでも、単に別workerのactorへ付け替えて採用しない。

終了後は全登録attemptの証拠を集め、保留・cancel前の有効な確認を維持したまま採用を検証する。
欠測・画像未提示・古いheadへの完了申告は採用を拒否する。seal後、結果receipt保存前に中断しても
同じ採用を再利用する。workerのmutableな変更案は親側へ不変コピーしてからrevisionへ参照する。

このAPIの検証は合成VFR MP4・合成rollout・ローカルの合成CLI processを使う。
実Codexの画像入力互換性、実動画の品質、本番の方針値は未検証。
実行可能なcampaignの範囲と起動手順は[CLI](../../../annotation/README.md)を参照する。
