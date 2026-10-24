# ci_failures: main CIの不具合引渡しと回復確認

`ci_watch` の分類済み記録（#1450）を読み、main CIの失敗署名を既存のIssue・PRへ結び付ける（#1452）。
Issueの作成とコメント以外のGitHub書込み、agentの起動、割当、レビュー、merge、CIの再実行、
workflow・timer・保護設定の変更は行わない。起動周期と重大障害時の優先割当・merge停止は人間の判断事項。

## 実行

Pythonのimport rootは `.agents/ops`。認証済みの `gh` を呼出元で準備する。
`sync --apply` は実際にIssueを作成・コメントするため、fixture以外での実行は運用側が判断する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures sync \
  --repo Motoki0705/tennis-lab --source-dir <ci_watchのstate-dir> --apply
```

`--source-dir` は初期化済みの [ci_watch状態](../ci_watch/README.md#保存形式と後続からの取得) を指す。
その公開Python APIで `records` を100件ずつ読み、同じread transactionの最新のSHA観測版から処理する。
読取中だけ既存 `collect.lock` を共有取得し、collector稼働中は未処理エラーにする。
GitHubの照合・投稿前にread transactionと共有lockを解放し、collectorの書込みを長く塞がない。
観測元への書込みは行わない。全記録を読むため、取得量は観測開始後の履歴に比例する。
`collection` の取得エラーを成功に変換せず、既知の失敗の記録・引渡しを続けても回復判定は保留する。

既定の保存先は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ci-failures/`。
別の保存先はサブコマンドの前に `--state-dir <path>` で指定する。
repository・観測元の絶対パス・観測開始SHA・workflow集合・schemaを固定し、不一致はエラーにする。
前回読んだsequenceとその内容を照合し、入力履歴の欠落・変更を空状態からの再開に読み替えない。

同じホスト・XDG状態root・repositoryの同期は共通のrepository lockで排他する。
出力state-dirやcheckoutが異なる起動も同じlockを使い、競合時は待たず非ゼロ終了する。
複数ホスト・異なるXDG root・外部の手動起票との分散排他は提供しない。

stdoutの `processed` は取得済み入力の処理を終えた意味で、修正完了を表さない。
`unresolved` が未回復署名数、`next_after` が入力の最終sequence。
取得エラー・不正な根拠・出典競合・原本欠落/改変は非ゼロ終了する。
書込応答喪失時も、同じコマンドを再実行してGitHubの全状態を照合する。

## 起票・既存経路への引渡し

- `code_regression` の根拠付き仮説だけを、新規の `bug` / `ai-proposed` 単独Issueにする。
  具体的な目的・範囲・対象外・依存・状況・Acceptanceは `issues.render_standalone_body` で生成する。
  親子関係や要件合意・共有基盤承認を生成しない。
- 本文1行目の出典は [issues共通API](../issues/README.md#ci失敗の出典) で生成・解析・索引化する。
  タイトルやラベルでは照合せず、全open/closed Issueをページ末尾まで調べる。
  同じ署名の複数Issue、不正な位置・版・形式のメタデータはエラー。
- 同じ出典を持つPR、または対応Issueをclosing句で参照するPRも全状態から調べる。
  PRだけが存在する場合も新Issueを作らない。複数の修正PRは履歴として保持する。
  同じ失敗を扱う既存作業には、この先頭行の出典を明示して接続する。自然言語の類似性では統合しない。
- 既存Issue・PRの本文・タイトル・state・担当・ラベルを変更せず、失敗発生と分類の根拠をコメントする。
  署名・SHA/run/attempt/job・分類内容からコメントを識別し、同じ内容の再投稿を防ぐ。
  投稿応答喪失後も、publisherとmarkerと本文全体を照合する。
- `flaky` / `runner_infrastructure` / `unclassified` は理由・原本参照付きのローカル記録に留める。
  対応済みの作業があれば追記するが、新しい修正Issueを作らない。
- 生成Issueは通常のopen Issueで、`ops-report` 容器や提案の出典を持たない。
  [dispatcherの収集](../dispatcher/README.md) が予定回に拾い、既存の着手判断・割当へ渡す。
  このCLIはdispatcherやrunnerを呼ばない。

## 回復の判定

Issueのclose、PRのclose、PRのmergeだけでは `recovered` にしない。
次をすべて満たす場合にのみ、修正PR・merge SHA・観測SHA・run/job URL・原本を保存し、対応作業へ回復コメントを残す。

1. 出典またはclosing句で結び付いた修正PRがmainへmerge済み。
2. `merge_commit_sha` が観測したmainのfirst-parent連鎖に含まれる。
   対象SHAはそのmerge以降、かつ同じ署名の最後の失敗SHAより後。
3. 対象SHAと元のworkflowの取得が完了し、そのworkflowの全runの最新attemptが成功。
   各runで元と同じjobが1つあり、jobと元の全失敗stepが成功。
4. テスト単位の署名では、成功jobログに同じpytest node IDの明示的な `PASSED` 行がある。
   集約の `1 passed`、別テスト、skipped/neutral、同じjob名だけでは実行を確認した扱いにしない。

成功jobログはGETで取得し、内容を検査できるローカル原本とjob IDの対応を保存する。
完了jobの保存原本は再取得せず検査して使う。取得失敗はエラー、保存原本の欠落・改変もエラーで、
APIから黙って補わない。テスト実行の記載がないログは理由付き未回復に留める。
job単位の署名は元のjob/失敗stepの成功を証拠とし、個別テストを実行したとは主張しない。

状態は `held`、`publication_pending`、`awaiting_fix`、`awaiting_main_ci`、`observation_incomplete`、`regressed`、`recovered`。
根拠保存後の投稿失敗は `publication_pending` に残り、分類による起票保留と区別する。
回復後に後続SHAで再失敗した場合は同じIssueを再利用し、回復確認を現在の状態から外す。
過去の回復記録は追記履歴に残る。closed Issueや放棄されたPRの再開・別作業への割当は担当親の判断に戻す。

収集全体の記録だけでなく、各SHAの最新取得エラーも回復判定を止める。
SHAの保存後・収集全体の保存前に中断しても、取得不能になったSHAの既知の失敗と原本参照を保持する。
同じSHAで後日の再CIが成功した場合、run・attempt・jobを含む回復根拠ごとに記録し、同じ根拠の再送だけを重複抑止する。

## 範囲取得と調査台帳への根拠

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures list --unresolved --limit 20
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures records --after 0 --limit 100
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures show --signature <ci-failure-v1:...>
# show/list/recordsが返すrefをJSONファイルへ保存して参照時点の原本を読む
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures original --reference <ref.json>
PYTHONPATH=.agents/ops .venv/bin/python -m ci_failures evidence --object <object-ID> --start-line 1 --limit 20
```

`list` はoffset/limitと総数・has_more、短い理由と原本refを返す。
`records` はsequence順の追記履歴とnext_after/has_more、`show` は署名の最新版を返す。
各記録のrefは `path / entity / id / sequence / sha256` で識別し、`original` は保存時点の内容を検査する。
ローカルの絶対パスを含むrefを公開Issueへそのまま載せない。

`record.occurrences[].evidence` と `record.recovery.evidence` は、
[調査台帳](../ops-report/README.md#調査台帳ライブラリ) の `record_evidence` にそのまま渡せる
`id / source_kind / reference / acquired_at / target_version / pr_head` を持つ。
referenceには元の `ci_watch` のsequence・対象SHA・内容検査値が入る。
入力のsequenceは `ci_watch records --after <sequence-1> --limit 1` で原本へ辿れる。
回復ログは `record.recovery.jobs[].test_log_object` を `evidence` で取得する。
台帳自体への登録・次の問いの生成・レポートへの掲載は全体調査側（#1447）が行う。

## 検証

```bash
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_ci_failures.py -q
```

実ci_watchの観測とrepair CLIをfake ghへ接続し、全Issue状態・PR・ページ境界・競合・応答喪失、
非回帰分類、修正merge前後の成功/失敗・原本参照・dispatcher収集と台帳の根拠互換を検証する。
実agentや実GitHub書込みは使わない。
