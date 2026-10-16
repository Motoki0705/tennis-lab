# クリップ選別

取得済み source を全時間から調べ、通常速度の実プレーが **15秒の過半**に
なる非重複窓を選ぶ。前後1秒を文脈として残し、中央13秒を注釈対象にする。
7.5秒ちょうど・スロー・リプレイは採用しない。対象区間の全フレームが
実プレーであるという意味ではない。必要数を満たせない場合は理由を残す。

選別は [Collection](../collection/README.md) の取得と準備の間に入る。
`ScreeningPolicy` を campaign plan の `selection` に固定し、
`Screening(collection).request(source_id, attempt_id)` でリクエスト・schema・
全体overview・worker指示を作る。共通workerがその指示を実行し、最後に
`finalize(request, decision, presentation, previews)` を呼ぶ。
このモジュールは Codex の起動や追加のQAレビューを実行しない。

- リクエストは source のbytes hash・実PTSの原点/time base・全長、policy、
  plan、最大clip数に結び付く。15秒未満・設定上限超過・非対応の映像形式は
  preflightの理由付き選外として台帳に記録する。
- `python -m src.tasks.base.annotation.screening --request PATH
  --request-sha256 HASH images --start S --stop E --step STEP` は CPUで逐次decodeし、
  最大24タイルのJPEGページを1ページずつ生成する。時刻はsourceの実PTSから
  計算し、frameの複製やpaddingはしない。同じ入力・区間・表示仕様のページは
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

保存先は固定rootの `campaigns/<campaign>/screening/<source>/<attempt>/`。
workerの自由な作業は `work/`、共通画像コマンドの生成先は `previews/`。
正式receiptと提示記録は親が作り、workerが編集する入力にしない。
機械検証は提示された画像と宣言された区間の整合性を確認する。実プレーや
速度の意味判断はworkerの選別であり、機械的に正しさを保証するものではない。

#1123 の単体テストは合成動画で境界・画像提示・hash・原子的公開・台帳再開を
検証する。実動画・Codex実ログによる接続検証は #1125 / #1140 / #1141 で行う。
