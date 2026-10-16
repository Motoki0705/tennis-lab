# クリップ選別

取得済み source を全時間から調べ、通常速度の実プレーが **15秒の過半**に
なる非重複窓を選ぶ。前後1秒を文脈として残し、中央13秒を注釈対象にする。
7.5秒ちょうど・スロー・リプレイは採用しない。対象区間の全フレームが
実プレーであるという意味ではない。必要数を満たせない場合は理由を残す。

選別は [Collection](../collection/README.md) の取得と準備の間に入る。
`ScreeningPolicy` を campaign plan の `selection` に固定し、
`Screening(collection).prepare(source_id)` をCPU工程で実行し、全体overviewと
不変のsource入力を作る。`request(prepared_input, worker_attempt_directory)` は
共通workerの受付時に小さいリクエスト・schema・指示だけを作る。workerがその指示を実行し、最後に
`finalize(request, decision, presentation, previews)` を呼ぶ。
このモジュールは Codex の起動や追加のQAレビューを実行しない。

- リクエストは source のbytes hash・実PTSの原点/time base・全長、policy、
  plan、最大clip数に結び付く。15秒未満・設定上限超過・非対応の映像形式は
  preflightの理由付き選外として台帳に記録する。
- `python -m src.tasks.base.annotation.screening --request PATH
  --request-sha256 HASH images --start S --stop E --step STEP` はworker自身のworkに
  画像要求を書き、親controllerへ返す。親は資源管理下のCPU工程で逐次decodeし、
  最大24タイルのJPEGページを1ページずつ生成する。時刻はsourceの実PTSから
  計算する。前の採用PTSからstep以内の最後の実frameを選び、窓末尾も覆う。
  疎な映像の実際の間隔はそのまま記録し、frameの複製やpaddingはしない。
  同じ入力・区間・サンプリング/表示仕様のページは
  hash検証後に再利用する。画像一式とreceiptはdirectory単位でatomicに公開する。
- workerの選別手順の正本は [WORKER.md](WORKER.md)。`check --decision PATH` は
  schema・時刻・件数・選別条件だけを検証する。画像閲覧の証明にはならない。
- `PresentedImages` は **信頼するlauncherが実際のimage tool出力から作る**。
  workerの自己申告・JPEG生成記録・簡易 `events.jsonl` のコマンド実行だけを
  presentationとして扱わない。全overviewページ、採用する各窓を0.5秒以下の
  間隔で覆うdetailの全ページが提示済みでなければfinalizeは拒否する。
  間隔は要求したstepだけでなく、実PTSから窓先頭→最初の画像、画像間、
  最後の画像→窓末尾を検証する。低fps・VFRなどで実際の間隔がpolicyの
  detail上限を超える場合は、全ページ提示済みでも採用を拒否する。
  不明瞭な境界の0.1秒確認はworkerが判断し、追加previewも根拠として残す。
- finalizeは結果のimmutableコピー、提示記録、preview群を根拠receiptに結び、
  `SelectionReceipt` を公開して台帳へ登録する。receipt公開直後に中断しても
  同じ参照で再開できる。未選択や不足を準備済み・公開済みと扱わない。
- 画像要求の未応答・失敗や、時間切れで確認を終えられない場合は `deferred` とする。
  `insufficient` は確認済みの素材で適格区間が足りない場合だけを表す。親adapterも
  未完了の画像要求を照合し、部分選択を確定・準備せず元動画を保持する。
  理由付きの明示再試行で選別をやり直す。遅れて画像が生成されても閲覧済みとはしない。

全体overviewとsource入力は `campaigns/<campaign>/screening/<source>/`、
workerの指示・結果は共通の `campaigns/<campaign>/workers/<work-id>/<attempt>/` に置く。
workerの自由な作業は `work/`、親が描画する追加画像はattemptの `previews/` に置く。
画像要求・応答の対応は `work/service-requests/<hash>.json` と `services/<hash>/reply.json`。
親はrequestのsource/plan/attempt・区間・画像数を照合し、1要求600frame、
policyで宣言した0.1秒以上の間隔に制限する。描画済み画像は閲覧済みとは扱わない。
正式receiptと提示記録は親が作り、workerが編集する入力にしない。
機械検証は提示された画像と宣言された区間の整合性を確認する。実プレーや
速度の意味判断はworkerの選別であり、機械的に正しさを保証するものではない。

#1123 の単体テストは合成動画で境界・画像提示・hash・原子的公開・台帳再開を
検証する。合成workerと実CPU描画stageの接続でも、rolloutのない結果が採用されないことを
確認する。実動画・Codex実ログによる接続検証はpilotで行う。
