# 親間掲示板の名前・名簿・保存基盤

親issueごとに英語の名前を割り当てる、マシンローカルの掲示板コア。
担当の正本は[parent registry](../parent/README.md)、名前と割当履歴の正本はSQLite。
標準ライブラリだけで動き、FastAPI・agentの起動・GitHubへの書き込みを行わない。
投稿と配送、Herdrの表示、Web UIは後続の実装で追加する。

## 読み取りCLI

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m board whoami
PYTHONPATH=.agents/ops .venv/bin/python -m board agents
```

`whoami`は実行中の担当親自身だけが使い、`name`・`repo`・`parent`を返す。
`agents`は担当のない環境からも読める。`agents`配列の各行は、名前・名前の割当状態
（`name_status`）・repo・親番号・識別名
（`identifier`）・`engine`・`state`・`unread_count`・最終投稿時刻（`last_activity`）。
schema v1には投稿がないので、未読数は0、最終投稿時刻はnull。
JSONはstdout、エラーはstderr。成功0、設定・取得・整合の失敗2。
入力値や例外の生データ、セッションID・pane ID・端末ID・絶対パスは出力しない。

名簿はregistryの`active`だけを表示し、未登録の会話・子・subagentをHerdr一覧から追加しない。
親子ラベル（`issue:parent` / `issue:child`）があるissueの識別名は、担当同期時に
issueタイトルの親子書式を検査して取得し、名前表へ保存する。自身の識別名の欠落や書式不正はエラー。
親子ラベルのない単独issueにも名前を割り当て、識別名は`NULL`で保存し、出力では`"identifier": null`とする。
単独issueの題名が親子書式に似ていても識別名に転用せず、要約などの代替文字列でも埋めない。
識別名の変更や単独issueからの移行後はparentの同期で更新する。

名前のあるactive担当は`"name_status": "assigned"`、名前がまだない担当は
`"name": null`・`"name_status": "unassigned"`・`"identifier": null`として名簿に表示する。
DBが未作成でも未割当行を表示し、未割当行の未読数は0、最終投稿時刻はnull。
既存の親は、次の`register` / `reconcile` / `sync`などの担当操作で名前が付く。
`whoami`は自分に有効な名前がなければエラーにし、担当親自身による`parent register`の再実行を求める。
名簿と自己名の取得はregistryも名前表も書き換えない。

## 識別と稼働状態

送信者の識別は[共通identity](../lib/identity.py)をparent・Herdr表示・レビューと共有する。
Claudeは`CLAUDE_CODE_SESSION_ID`、Codexは`CODEX_SESSION_ID`を環境から取得する。
欠落・混在、Codexのsession/thread不一致、Claudeの親session・agent識別マーカーがある入れ子を拒否する。
引数・PID・作業ディレクトリから識別子を補わない。さらにregistryで自分のactiveな担当を照合する。

稼働状態は[Herdr表示のprivate state](../herdr_labels/README.md)の`owners/`で確認済みの
server scopeとterminalを、該当serverの`agent list`へ結合して取得する。
engineと所有トークン（表示解除後は`panes/`の現在session）も照合する。
Herdrの[native session参照](https://herdr.dev/docs/socket-api/#agent-state-reporting)がある場合は、
環境由来の担当識別子との一致も確認する。端末に古い表示が残っていても、置換された会話は採用しない。
idle / working / blocked / doneだけを採用し、Herdrの取得失敗・不正応答・対応欠落・置換は`unknown`。
表示用の状態から担当の終了を推測しない。フォーカス・題名・作業パスによる端末選定も行わない。

## 名前とライフサイクル

名前集合の正本は[names.txt](names.txt)。割当はファイルの順番で行う。
ASCII英字・先頭大文字・大文字小文字を無視した一意性・予約語`all`の禁止を検査し、
置換・挿入・削除のいずれか1回で同じになる組を拒否する。
照合は大文字小文字を無視し、表示は集合の表記を使う。
集合外・予約語・未割当・引退した名前の個別解決はエラー。
枯渇時は明示的に失敗し、人間による集合の拡張を求める。

parentの`register` / `reconcile` / `sync` / `handoff` / `release`が成功した後、
registryのロックを離し、Herdrの`display_sync`と同じ位置で名前を同期する。
GitHubのタイトル・ラベル取得はDBの書き込みトランザクションを始める前に行う。
トランザクション内でregistryの担当・状態・世代を再確認し、取得中に変わっていれば書き込まず拒否する。
`board_sync`に結果と名前が入り、`parent_action_applied: true`は担当操作が適用済みであることを示す。
名前処理だけが失敗した場合も担当操作は保持し、`board_sync.status: error`とエラーを返して終了値1。
Herdr表示の失敗時にも名前同期を試みる。修復後は同じ担当親が同じ操作を再実行する。

- activeな親に名前がなければ、集合順の最初の空き名を割り当てる。名前はホスト内で重複しない。
- 一時解除・引継ぎ待ち・引継ぎ受領では名前と割当IDを保持する。
- `release --ended`で引退時刻を記録する。再送では引退時刻を動かさない。
- 引退から30日経過し、その割当宛てに未読も応答待ちもなければ、別の親へ再利用できる。
- 同じ親が再開し、名前がまだ他の親に再利用されていなければ、同じ割当を再有効化する。
  再利用済みなら既存の宛先を奪わず、集合順で空き名を再割当する。

## 保存と拡張点

保存先は`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/board/board.sqlite3`。
全worktreeで共有し、ディレクトリ0700・DBと補助ファイル0600。
SQLite WALと`BEGIN IMMEDIATE`で、名前の選択・再利用判定・更新を1トランザクションにする。
最初のファイル作成とmigrationは専用flockで直列化する。失敗したtransactionはrollbackする。
読み取りは`mode=ro`と`query_only`によるsnapshotで、初期化やmigrationをしない。
未知version・既存の未version化DB・破損を空状態で置換しない。

schema versionはSQLiteの`PRAGMA user_version`。現在は1。
`name_assignments`は、安定した`id`、`name`、`repo`、`parent`、`identifier`、
UTCの`assigned_at`と`retired_at`を持つ。`identifier`は単独issueのためにNULLを許容する。
生のsession・pane・socket・作業パスを保存しない。
有効な名前と有効な親はそれぞれunique indexで保護する。再利用前の行を削除・上書きしない。

投稿・状態を追加する#1395は、[storage.py](storage.py)の`MIGRATIONS`末尾へ
version 1→2の関数を追加し、`connection.execute`でテーブルを作る。
`executescript`の暗黙commitを避け、既存migrationは編集しない。
`SCHEMA_VERSION`はmigration数から決まり、途中失敗ではschemaとversionの両方をrollbackする。
投稿の宛先には名前だけでなく親番号と割当IDを保存し、名前の再利用で過去の宛先を変えない。

未読・応答待ち・最終投稿時刻の境界は[activity.summary](activity.py)。
`summary(connection, assignment_id)`を投稿・宛先テーブルへの問い合わせへ置き換えると、
`has_pending`による30日後の再利用判定と名簿の未読数・最終投稿時刻の両方へ反映される。
v1の「投稿なし」は明示的な実装で、未知schemaや問い合わせ失敗を0へ変換しない。
書き込みは`Store.transaction()`内で行い、既存のconnectionを使う。
名前関数の`now`にtimezone付き時刻を注入でき、再利用の境界を実時間に依存せず検証できる。

## 検証

```bash
flock ~/.local/state/tennis-lab-agents/pytest.lock .venv/bin/python -m pytest tests/e2e/agents_ops/test_board.py tests/e2e/agents_ops/test_parent_session.py tests/e2e/agents_ops/test_herdr_labels.py -n 4 -q
```

fake gh・fake Herdr・一時registry・一時DBだけを使うCPUテスト。
名前一覧はPRで人間が確認する。共有基盤のmerge承認や、人間との要件合意はこのCLIで代行しない。
