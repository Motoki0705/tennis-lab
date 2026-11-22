# レビュー用の取得と隔離CPUテスト

正式レビューと全体調査が共有する実装。全体調査の `opsreport` は互換入口と台帳adapterを持ち、
正式レビューの `review_server` は複数の固定commitとrunごとのjournalを使う。

- `retrieval*`・`mcp_server`: 固定commitのgit読取、GitHub GET、ページ境界・切詰め・取得証拠。
- `reproduction`: 許可された既存CPUテストだけを使い捨てcloneで実行する。
- `repro_isolation`・`repro_linux`・`repro_sandbox`: Linux namespace/cgroupによる強制と実機probe。
- `records`: 記録先のProtocolと台帳に依存しないjournal。
- `review_server`: requestで固定したcommitごとの取得入口とテスト許可リスト。

## GitHub取得ツール

正式レビュー・全体調査のGitHub取得には **gh 2.97.0以上の安定版**が必要。
`--allow-escape-sequences` が定義された
[公式v2.97.0のAPI実装](https://github.com/cli/cli/blob/v2.97.0/pkg/cmd/api/api.go) を最小版とする。
各GETの前に `gh --version` を検査し、旧版・版不明・prereleaseは
`UNSUPPORTED_GH_VERSION` と必要最小版を返し、APIを呼ばない。gitだけの取得にはghは不要。
`gh api --method GET --allow-escape-sequences` の出力を受信してから、以下の方式で無害化する。

- 適用範囲は `ci_log`、`pr_diff`、issue/PR/コメント/reviewの本文と一覧、CIの一覧に含める表示用メタデータ、
  GitHubコマンドのエラー詳細。JSONは解析後の選択した文字列フィールドに適用する。git取得は対象外。
- 完全なANSI列（7-bit/8-bitのCSI、OSC、DCS、SOS、PM、APC、その他のESC列）を除去する。
  残るC0・DEL・C1制御文字はtabとLF以外を小文字16進の `\u00xx` 表記へ置き換える。
  CR、BS、NULも可視化し、不完全なANSI列はESC等を可視化して本文を保持する。
- 応答の各GitHub `sources[]` に `text_sanitization` を付け、`policy=terminal-controls-v1`、
  `input=gh_output`、`changed`、`ansi_sequences_removed`（列数）、
  `control_characters_escaped`（除去後に可視化した文字数）を残す。変更がなくても0件を記す。
  件数はそのsourceから表示に使うフィールド全体を範囲指定前に処理した数で、要求したページだけの件数ではない。
  応答全文を保存する `retrieval.jsonl` にも同じ印と件数が残る。
- gh自身がJSON内の制御文字を先にcaret表記等へ変換する場合がある。
  件数はghから受信した文字列への追加処理数であり、GitHubの原文の制御文字総数ではない。
  既に可視化された文字列やリテラルの `\u001b` から、元の制御文字を推測して除去しない。
- ID、head SHA、PR差分の完全性は無害化前の値で検証する。本文の行数・文字offset・継続用
  `content_sha256` は無害化後の全文が基準。ANSIだけが変わり表示結果が同じ場合、このdigestは変わらない。

## 隔離CPUテスト

隔離実行は最大15分・memory 4GiB・swap 0、GPU・外部network・credentialsを隠す。
サービス内は同時1件で、通常pytestには共通ロックを課さない。
OSの実機probeが有効でなければ `cpu_tests` を公開せず理由を返す。
node IDは親が固定commitに存在するCPUテストを選び、任意コマンド・テスト生成・依存追加は受け付けない。
probeと許可リストの準備手順は [全体調査の隔離実行](../../ops-report/README.md) を参照。
