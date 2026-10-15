# 応答・画像提示の記録

`capture()` は終了したworkerの `Launch` / `ExitReceipt` と、そのthread固有の
Codex rolloutを照合する。別threadの最新ログは選ばず、session ID・attempt CWD・
起動時の参照を検証する。`WorkerPool` は実Codexの成功をadapterへ渡す前にこれを実行し、
計画と異なるmodel・矛盾する応答ID・取得不能なログを採用検証へ進めない。
失敗したworkerも可能なら記録し、取得不能の理由をattempt内に残す。

- `token_usage_record.response_id` を重複排除する。`token_count` 通知件数を
  API回数として使わない。input、cached input、cache write、output、reasoning、totalを別々に
  記録する。reasoningをoutputへ重ねて足さない。欠測・IDなし・内容が違う重複は
  明示し、不完全な合計をnull、確認できた小計を別欄に残す。記録されたthread/turn累積と
  当該応答の `input + output` によるcontext近似は別値である。
- call IDでtool入力と出力を結び、出力を次の一意な応答へ対応付ける。
  境界のない出力、ID欠測直前の出力、compactionで置換されたcontext内の未対応出力は
  orphanとして保持し、後続応答への画像提示を主張しない。分類はversion付きtask adapterが
  所有する。共通層はtask名や編集コマンドを推測しない。
- 画像はtool出力のdata URLに記録された**その時点のbytes**を保存する。
  更新後の作業画像を読み直して代用しない。rendererの不変画像とhash/sizeが完全一致した
  ときだけ、frame/PTS・crop・倍率・tile・描画条件・注釈版を含むmetadataへ結ぶ。
  同じbytesの候補が複数ある場合は一意なImageView pathも必要で、batch順で推測しない。
  resizeや再圧縮で対応が不明な画像はunboundとし、教師採用の確認範囲に使わない。
- `presented_images()` は応答へ対応した、元画像との同一性を確認できる参照だけ返す。
  これは画像の入力証跡であり、ラベルの確認・採用を証明するものではない。
  task adapterが別途レビュー範囲・判断・機械検証を照合する。

保存先は `campaigns/<campaign>/reports/telemetry/<attempt>/`。
JSON索引は100応答ごと、記録とHTMLは応答ごとに分ける。HTML索引は画像を埋め込まず、
応答ページで必要な画像だけ読む。一意に対応した不変renderer画像は、ログ内のbytesと
保存ファイルを再照合して直接参照し、画像を二重保存しない。対応が不明な画像はログの
bytesを `cache/presented-images/<sha>.*` に保存する。reportから参照される画像・
renderer metadataはGCの保護対象である。参照共有は独立したバックアップではなく、
元画像の改変・消失は検証エラーになる。既存reportやcacheは書き換えない。
raw rolloutは移動せず、path/hash/sizeを記録する。共有レポートには推論文やtool入力/出力の
全文を複製せず、入力/outputのhash、文字数、時刻、画像を残す。

パーサはraw JSONLを1行ずつ読み、最大64 MiB/行、画像32 Mpixel、未対応tool出力1024件の
上限を持つ。上限超過・未知の画像形式・改変は明示的なエラーになる。全画像をメモリに
集めず、応答の小さい索引だけを保持する。画像単独のtoken料金を区間差分から算出しない。

合成rolloutとローカルの偽CLIによる試験は、欠測・重複・画像すり替え・曖昧な対応・
model/CWD/threadの不一致・再開時の改変検出・ページングを検証する。
実Codexの収集/レビューと注釈差分の接続は各task adapterおよび #1131 / #1141 の実行で確認する。
