# ops-report: 運用レポートと逆提案（#1057）

プロジェクト本体とAI制御（AGENTS.md・`.agents/`・skill・memory・GitHub運用・この自動化自体）を
共同で改善し続けるため、3日おきに「運用レポート」issueを作り、人間が方向性を採用した提案をtriageで作業issueへ整理する。
着手判断・親割当は半日ごとの外部ループが行う。設計原則は [../../README.md](../../README.md)、
全体方針は親issue #1329、既存の手動受領経路の背景は #1216 を参照する。

## 構成

| パス | 役割 |
|---|---|
| `ops_report.py` | CLI入口（`report` / `triage` / `schedule init` / `schedule status`） |
| `opsreport/collect.py` | 決定的な収集。gh（open PR/issue、merge済みPR、CI run、過去レポート）、git（期間内commit、行数の多いファイル、TODO系マーカー、worktree/branch数）、掃除モジュール出力、共有5層memory、`knowledge/`、AI制御ファイル |
| `opsreport/retrieval.py` / `retrieval_git.py` / `retrieval_github.py` / `mcp_server.py` | 固定版・GitHubの範囲取得、取得記録の追記、GET専用stdio MCP。profileへの接続は統合側が行う |
| `opsreport/agent.py` | プロンプト組み立て、`../lib/agent_exec.sh` による親Codexのread-only実行、集約JSONの検証 |
| `opsreport/audit.py` | 固定SHAの全対象・担当計画、調査worktree、native scoutの返却と統合の検証、再開入力の照合 |
| `opsreport/ledger.py` | 可変の調査タスク・発見・根拠・引渡しの追記履歴、再読込・整合検証・親向け要約（統合用ライブラリ） |
| `opsreport/native.py` | 親runとCodex transcriptからnative起動・完了・実効設定を列挙し、台帳のタスクとnative IDを照合する読み取り専用API |
| `opsreport/reproduction.py` / `repro_isolation.py` / `repro_linux.py` / `repro_sandbox.py` / `reproduction_server.py` | CPUテスト要求の制限、使い捨てclone・台帳根拠・ログ、実機proofで許可するOS隔離backend |
| `opsreport/proposals.py` | 提案のID付与・描画・本文からのチェック状態の解析・子issue番号の追記 |
| `opsreport/schedule.py` | 初回基準・3日周期の予定計算と、成功/失敗・スキップのマシンローカル状態 |
| `opsreport/memory.py` | main版の専用memory入力、構造化更新の検証、エントリ/要約/索引の生成、採否の入力 |
| `opsreport/memory_pr.py` | 公開確認のjournal、専用worktreeでのcommit/PR、必須CI・パス・HEADを照合する専用マージ処理 |
| `required_checks.txt` | 専用memory PRの自動mergeに限って必須とするCI check名 |
| `opsreport/report.py` | issue本文（日本語）の組み立て |
| `opsreport/triage.py` | 採用チェック・既存受領の記録、既存issue照合、採否のmemory入力 |
| `opsreport/organization.py` / `organize_prompt.md` | 全open/closed issueとコードの固定入力、read-only整理agent、構造化計画の検証 |
| `opsreport/organization_apply.py` | 計画journal、作業issue生成・再利用、正式親子・親一覧・出典リンクの復旧 |
| `prompt.md` | agentへの依頼文（分析観点・品質基準・出力JSON形式） |
| `systemd/` | `report --if-due`（毎日 08:00 JST に予定判定）と `triage`（毎日 09:00 JST）の service / timer |

分担: 収集はスクリプト（同じrepo状態なら同じ事実）、コードベース読解と提案文の生成はagent（read-only）、
issue作成・編集はスクリプト。agentはJSONだけを返し、IDとMarkdownはスクリプトが付ける。
read-only のagentは git / gh / ネットワークを使えないため、GitHub・git履歴の事実（放置PRの差分規模・mergeable・CI、
全open issue、checkoutと基準refで内容が異なるファイル一覧など）はすべて `collect.py` が集めてプロンプトに埋め込む。
親のmodel / effortは [`../lib/agents.conf`](../lib/agents.conf) の `ops_report` profile、
子は実行環境のnative `scout` 定義に従う。子の設定を複製・上書きしない。

## 並列調査と部分公開

`audit_plan` は `base=` の固定commit SHAの全追跡パスを4担当（コード、基盤、AI運用/文書、issue/PR/研究）へ
一意に割り当て、収集データの担当範囲・除外・終了条件と一緒に親へ渡す。親と全scoutは、元repoの
`.claude/worktrees/ops-report-audit-<run-id>` に作る同じdetached checkoutをread-only sandboxで読む。
外部submoduleはgitlinkの固定版と利用箇所を対象とし、外部内部ソースは対象外・未調査と本文に明記する。
担当対象の未確認範囲は子が返す。実行後はcleanな調査worktreeだけを削除し、
HEAD変更・dirty状態は強制resetせずエラーにする。呼び出し元checkoutの変更は調査の事実に混ぜない。

親はnative `spawn_agent` の `scout` を並列起動し、枠が空いたら残りを進める。
並列数はprofileの `OPS_REPORT_MAX_THREADS`。wrapperは `multi_agent` を明示的に有効化し、
JSONL形式のCLIログと通常のセッション保存を維持する。native scoutを使わない旧Claude/alternate経路は削除し、
指定されたらエラーにする。別モデルの独立CLIセッションへの置き換えはしない。

返却形式の正本は [prompt.md](prompt.md)。担当ごとのnative ID、基準SHA、開始/終了時刻、状態、結論・根拠・反証・
未確認範囲・findingを親がJSONで返す。スクリプトは担当の欠落・重複・担当外・基準版/role違い・不正形式、
2担当以上の時刻重複、全findingと統合判断・提案の対応を検証する。
同じtopicは1判断へ統合し、重複は1提案へ、反対のpositionは根拠付き解決または未解決の矛盾にする。
topicや根拠が意味的に同一か・正しいかは親が現物を照合する責務で、スクリプトだけでは証明しない。
提案には担当findingと基準SHAを付け、カバレッジ・失敗理由・再開方法を本文に載せる。

失敗・中断・不正返却・未解決の矛盾があれば、確認済みの担当結論と検証できた提案だけを**部分公開**する。
本文冒頭に「未完了」、メタデータに `status=incomplete` を記録し、終了コードは1、予定状態はfailedにする。
不正な親の統合に含まれる提案は掲載せず、有効な子の結論は残す。専用memoryの更新は全体完了まで全件保留する。
triageは未完了レポートを処理しない。部分公開中は提案にチェックを付けない。

固定入力・全対象一覧・検証済み担当結果はstate dirの `audit/<slot>.json` にatomic replaceで保存する。
次回の起動または `report --if-due` は未完了回を優先し、不足担当だけを補って**同じissueを更新**する。
全担当受領済みなら親の再統合だけを行う。並列の証拠が不足した場合は2担当を再起動する。
基準SHA・集計境界・入力ハッシュは再開時も固定し、後で新しいslotを作る際に前回完了回の境界から集計する。
issueのclose、提案の選択/子リンク、収集中の本文変更、checkpoint欠落/破損/入力不一致は書き込み前に停止する。
checkpointを失ったら `collected.json` のaudit_runtimeを除いた入力と `audit.json.results` を照合して明示復元する。
入力ハッシュは本文の `audit=` とも一致しなければならず、新しいcheckoutへ黙って切り替えない。

親+子の全体調査に経過時間の上限はない。`OPS_REPORT_AGENT_TIMEOUT` は削除し、残っていれば明示エラー。
serviceの `TimeoutStartSec=infinity` とwrapperの `timeout=None` を使う。個別ツール・fetch・掃除・ロック待ちの
制限は別で、失敗を全体調査の成功へ読み替えない。agent失敗やSIGINT/SIGTERMはプロセスグループを回収して未完了公開する。
強制終了・停電等で公開できない中断はrunning/failed状態とcheckpointから再開する。

## オンデマンドの読み取り経路

`opsreport.mcp_server` はagentのsandbox外で起動するstdio MCP。調査親とnative子のsandboxはread-onlyのままにする。
起動時にホスト側でrepo・調査回・完全な基準commit SHA・GitHubの `owner/name`・外部のprivateな記録先を固定する。
`--ops-memory-sha` は専用memoryの別の固定版を指定する入口で、省略時は基準SHAを使う。

```bash
PYTHONPATH=.agents/ops/ops-report .venv/bin/python -B -m opsreport.mcp_server \
  --repo "$PWD" --base-sha "$base_sha" --repository "$repository" \
  --round-id "$round_id" --journal-dir "$journal_dir"
```

MCPのcommandにはこのcheckoutのPython、argsには `-B -m opsreport.mcp_server` と上記引数、
envの `PYTHONPATH` にはこのcheckoutの `.agents/ops/ops-report` の絶対パスを渡す。
Codex CLI 0.163.0の `-c mcp_servers.*` は **`exec` の後**に置く。
`exec` より前では登録が見えてもサーバーが起動されなかった。共通ランナー・profileへの恒久接続は統合側で行う。
Codexの `tools/list` は標準の `_meta` を伴うため受理する。cursorは省略または `null` のみ受理し、
空文字を含むcursor値と未知の引数は拒否する。tool一覧は1回で全件返すため、cursorを黙って無視しない。

| tool | 取得する範囲 |
|---|---|
| `code_paths` / `code_search` / `code_file` | 基準commitのパス、リテラル検索、UTF-8ファイルの行・文字範囲。symlinkを辿らず、gitlinkは境界として列挙する |
| `git_history` / `commit_diff` | 基準commitから辿れる履歴とcommit差分。shallow履歴、基準版の後のcommit、外部diff/textconvは許可しない |
| `record_paths` / `record_file` | `knowledge` / 共有 `memory` / 専用 `ops_memory`。ファイルのpathはscopeからの相対指定 |
| `github_items` / `github_item` | issue/PRの検索・一覧と本文。リテラル句、state・label・author・更新日で絞る |
| `github_comments` / `github_comment` | discussion・review comment・reviewの一覧と選択した本文。一覧には本文を入れない |
| `pr_diff` | 期待する完全なhead SHAを必須にして差分を範囲取得。取得前後のhead/base・変更量と本文のhunk範囲を検証し、ファイル数・追加/削除行数をPRメタデータと照合する |
| `ci_runs` / `ci_jobs` / `ci_log` | run一覧、取得時に固定したattemptのjob一覧、jobログの範囲。run/job IDとコードのhead SHAを保持する |

`page` / `per_page` で一覧を、`start_line` / `line_count` と `offset` / `max_chars` で本文を選ぶ。
文字offsetは選択した行範囲内のUnicode文字位置。`pagination.next` は元の絞り込みを含む次回のargumentsで、
同じtoolへそのまま渡す。本文の続きには `expected_sha256` が付き、取得途中の本文更新は `SOURCE_CHANGED` にする。
GitHub一覧は取得時点ごとに変化するため、複数ページを単一時点のsnapshotとみなさない。

全取得応答に出典、取得時刻、基準SHA、該当するPR/CIのhead、ページ/省略情報、実行権限を変更しない調査資料である旨を付ける。
1応答64 KiB・1回のbackend取得64 MiB・個別コマンド45秒を上限とし、到達時は構造化エラーを返す。
GitHub検索の1000件上限・run一覧の安全上限・incomplete応答、403/5xx、欠落ログ、PR更新、部分差分も成功に変換しない。
PR差分はhunk境界で切れた場合も独立した追加/削除行数で検出する。hunkを持たない変更は、binary marker、
完全一致rename/copy、mode変更、空ファイルの作成/削除を区別し、本文の完全性を照合できなければ `INCOMPLETE_DIFF` にする。
応答上限には小さいページ/範囲で対応し、backend上限やGitHub上限で取得できない範囲は未確認として残す。
任意のコマンド・HTTP endpoint・書込みmethodはtool引数に存在せず、GitHubには `gh api --method GET` だけを使う。

外部記録先は0700のディレクトリとし、`retrieval.jsonl` は0600・flock・追記・fsyncで親子のサーバー間で共有する。
要求と応答をreceiptで結び、応答全文と内容のSHA-256を保存する。記録失敗時は取得前に止めるか、未記録の本文を返さずエラーにする。
別の調査回/基準版の混在、途中で切れた末尾、記録の消失も停止対象。調査台帳への書込みtoolは提供しない。

自動検証は `test_ops_report_retrieval.py`（fake gh・使い捨てGit repo・実stdioプロセス。実agent/GitHub書込みなし）。
2026-10-24の接続probeでは、既存 `agent_exec.sh --agent codex --profile ops_report --mode read-only` と一時PATH shimを使い、
親と `spawn_agent(task_name=..., fork_turns="none", message=...)` のnative子が同じ追加MCPを呼び、両者のreceiptをサーバー側記録と照合した。
完成版では親が固定版の `AGENTS.md`、子がGitHubの #1447 本文を範囲取得した。両者の実記録でread-only / approval never / Astra / maxを確認した。
role選択引数・model/effortの上書き・設定ファイルの変更は行っていない。実効service tierは記録がなく未確認。
privateな実行記録と `evidence-summary.json` は指定campaignのprobe領域に保存する。

## 隔離されたCPUテスト要求

`opsreport.reproduction_server serve` は上の取得サーバーに `cpu_test_status` と `cpu_test_log` を追加する別入口。
同じ `--repo` / `--base-sha` / `--repository` / `--round-id` / `--journal-dir` / `--ops-memory-sha` に加え、
既存台帳の `--ledger` と、ホスト側でCPU用と確認したJSONの `--cpu-tests-file` を指定する。
調査回は台帳と同じ `YYYYMMDD`。JSONは `{"base_sha":"完全な基準SHA","nodeids":["tests/test_example.py::test_example"]}`。
allowlistの変更権限はagentに渡さない。agentから受けるのは既存の `finding_id` と、このリストに一致する `nodeids` だけ。
任意コード、shell、pytestオプション、ネットワーク・環境変数・版・上限の指定、パス脱出、symlinkは拒否する。

`cpu_tests` は既定では非公開・実行拒否。ホスト担当者が次の固定probeを実行し、全challengeが成功した
privateな `probe.json` を `serve --isolation-proof PATH` に渡した場合だけ公開する。環境変数による迂回はない。
proofは同じboot・UID・backendコード・Python環境に限り24時間有効で、各要求でも照合する。
欠落・期限切れ・不一致・検査失敗は理由付きの未完了とし、実行しない。

```bash
PYTHONPATH=.agents/ops/ops-report .venv/bin/python -B -m opsreport.reproduction_server \
  probe --directory tmp/reproduction-probe
```

probeは本番と同じscope・namespace・chrootで、ホストの一時localhost listenerへの接続拒否、
資格情報sentinelとHOME配下の不可視、GPUデバイスの不可視、権限除去・runtimeのread-onlyを検査する。
192MiB scopeでは単独なら入る128MiBの子を2つ起動し、合計でOOMになることを確認する。
離脱した子孫を残す通常終了・1秒のscope期限・stdout閉鎖後のwatchdogも検査し、scopeの無人化を照合する。
ログと `probe.json` を保存し、全件成功だけ終了値0。GPU・Blender・任意の再現コードは起動しない。

無効な `cpu_tests` の直接呼出しも結果と「要再現」を記録する。独立cloneは基準SHAに固定し、
元repoへworktreeを登録せず、objectsもhardlink・alternatesを共有しない。基準版と配備版のwrapperが違えば停止する。
共有 `pytest_locked.sh` はsandboxの**外**で実行する。一時的な `.venv/bin/python` launcherで共有lock取得後に
scopeを作り、sandbox内の実Pythonへpytest引数を渡す。worker数はwrapperの固定値を使い、launcherは終了後に照合・除去する。
隔離CPUテストの同時1件制限は専用の `isolated-cpu-tests.lock` で守る。通常のpytestはこの制限を受けない。実行全体は15分・4GiB・swapなしの一時user scopeで制限し、
終了時にはそのscopeだけを停止して空であることを確認する。永続systemd設定やtimerは変更しない。
user・mount・net・PID namespaceと最小chrootを使い、ホストHOME・データ・サービスsocketを公開せず、
cloneとread-onlyのPython runtime/venvだけを渡す。`/dev` はCPU用の4デバイスだけ、WSL GPUライブラリも隠す。
capabilityを全て除去してno-new-privilegesを設定し、実行前にも制約を検査する。tmpfsは64MiB、process数は256が上限。
実行後はHEAD不変かつignoredを含めcleanなcloneだけを削除し、dirty・回収未確認なら保持してエラーにする。
要求・結果・ログはprivateな記録先へ保存し、内容の検査値付き参照を台帳の不変根拠と既存発見へ追加する。
拒否・失敗で発見を確定へ進めない。台帳の競合・書込失敗、根拠やログの欠落・改変は明示エラー。

`test_ops_report_reproduction.py` はfake runnerと実gitで管理経路・proof拒否・scope条件を検証する。
fakeの成功はOS隔離の証明ではない。profile・現行report CLIへの接続と、CPU allowlist・proofの配布は統合担当の責務。

## 調査台帳ライブラリ

`opsreport.ledger.Ledger` は調査回の永続記録だけを扱う。現行の `audit.py` / CLIとはまだ接続しておらず、
旧checkpointの移行、情報取得、agent起動、レポート描画、GitHubへの引渡しは呼出元の責務。
native agent IDの記録だけで実起動や正式レビューを証明したことにはならない。

```python
from pathlib import Path
from opsreport.ledger import Ledger

book = Ledger.create(
    Path("audit-ledger.json"), slot="20261024", base_sha=base_sha,
    goal="潜在的な問題を根拠付きで発見する", at=now,
)
# 再開時はcreateせず、元の契約を明示して読む。
book = Ledger.load(Path("audit-ledger.json"), slot="20261024", base_sha=base_sha)
page = book.summary(limit=20, offset=0)
record = book.original(page["run"]["ref"])
```

`slot` は `YYYYMMDD`、SHAは完全な小文字のcommit SHA、時刻はtimezone付きISO日時。
`create(..., report_issue=番号)` または `bind_report(番号, at=now)` で公開先を1回だけ結び付ける。
公開が後になる場合は、調査終了後にも結び付けられる。`snapshot` は再構成した全状態のコピーを返す。

各 `record_*(record, at=now)` は下表の全フィールドを持つ辞書を受け取る。
差分のpatchではない。未設定の値も `None`、空文字列、空配列として明示する。根拠参照は台帳内の根拠ID。
IDは種類ごとに一意とし、更新時は同じIDで新しい記録を追記する。根拠は不変で、再取得には新しいIDを使う。

| API | recordのフィールド |
|---|---|
| `record_task` | `id`, `question`, `scope`（文字列配列）, `origin`（`kind`, `id`）, `agent_id`, `status`, `started_at`, `finished_at`, `conclusion`, `evidence`, `next_questions`（`question`, `scope` の配列）, `unconfirmed`（`scope`, `reason`, `resume` の配列） |
| `record_finding` | `id`, `task_id`, `kind`, `status`, `conclusion`, `evidence`, `counterevidence`, `details` |
| `record_evidence` | `id`, `source_kind`, `reference`, `acquired_at`, `target_version`, `pr_head`（版とPR headは少なくとも一方を記録） |
| `record_handoff` | `id`, `finding_id`, `kind`（`issue` / `pr` / `proposal` / `unassigned`）, `reference`（未割当のみ `None`）, `related_parents`（issue番号配列）, `reason` |
| `end` | `status`（`completed` / `incomplete`）, `reason`, `checked_scope`, `excluded_scope`（範囲の文字列配列）, `unverified_risks`（`scope`, `reason`, `resume` の配列）, `further_investigation`（理由の文字列配列） |

タスクの起点 `origin.kind` は `goal` / `task` / `finding`。全体目的のIDはslotとし、他は既存のIDを参照する。
タスクは `planned` で登録して `running` へ進み、`completed` / `failed` / `interrupted` / `cancelled` で終了する。
起動前の失敗・中断・取消も記録できる。`failed` / `interrupted` は同じIDで `running` へ戻し、
新しい開始時刻・native IDを記録する。成功には根拠と結論、失敗・中断・取消には未確認範囲・理由・再開方法が必須。
問いと範囲は登録後に変えず、起点は起動後に変えない。NFKC・空白統一・casefoldした問いと、
順序を除いた正規化scopeが同じ別タスクは登録できない。scopeの大文字・小文字は区別する。
完了・取消したタスクも重複判定に残り、別IDで同じ問いを再起動できない。

発見は `suspected` / `unverified` で登録する。`confirmed` には根拠、`refuted` には反証が必須。
`design` / `duplicate` / `stale-dependency` / `ops` 等の `details` は空辞書。
以下の2種類は専用の構造を持つ。

- `unused-code`: `details` に `decision`（`delete-candidate` / `keep` / `unverified`）と `usage_paths` を記録する。
  利用経路の種類はコードの `USAGE_PATHS` が正本で、動的import・registry・文字列・設定/CLI・生成コード・
  学習/評価/再現スクリプト・公開入口を区別する。各経路は `kind`, `outcome`
  （`used` / `unused` / `not-applicable` / `unverified`）, `evidence`, `reason` を持つ。
  `unverified` 以外の経路判定には根拠が必須。削除候補の確定には全経路で未使用または対象外の確認が必要で、
  維持の確定には少なくとも1つの利用経路が必要。未確認の判定は確定へ昇格できない。
- `bug` / `vulnerability`: `details` に `target_version`, `conditions`（文字列配列）, `reachability`, `impact`,
  `remediation`, `verification`, `usage_paths` を区別する。確定には全項目、到達できる利用経路の根拠、
  対象版と一致する根拠の `target_version` または `pr_head` が必要。反証は共通の `counterevidence` に残す。

`source_kind="code"` の根拠は調査基準SHAと一致させる。PR上のコードは別の出典種別でPR headを明記する。
台帳は参照先を取得しないため、根拠の内容・意味的な妥当性や保存先の存在確認は取得側と親が担う。

未完了タスク・未検証の発見・重要な未検証リスクが残る回は `completed` にできない。
取消したタスクのscopeも、完了回では `excluded_scope` に明記する。
`end` 後の追記は `resume(reason, at=now)` で未完了回を明示的に再開してから行う。
過去の終了理由も履歴に残り、完了回は再開できない。総数・段階数・同時枠の上限は台帳に持たない。

`summary(limit=20, offset=0)` は未完了タスク・未検証仮説・次の候補を各カテゴリでページングし、
件数と残数を返す。長文は240文字の抜粋にし、原本参照である `ref` を添える。
候補の `kind` は `planned-task` / `resume-task` / `followup`。登録済みの問いを追加調査として重複提示しない。
`reference(種類, ID)` / `original(ref)` で要約に載らない記録も取得でき、後続の更新後も参照時点の原本を返す。
`ref` は保存先・種類・ID・履歴番号・その時点までの履歴の検査値を含むローカル参照であり、公開本文へそのまま掲載しない。

保存は隣接する `.lock` で直列化し、追記した履歴を一時ファイルのfsync・atomic replace・親directoryのfsyncで確定する。
別writerの更新後は `LedgerConflictError` とし、再読込と判断のやり直しを要求する。
途中切れ・重複JSONキー・未知version・不正な参照や遷移は `LedgerCorruptError` で停止する。
元ファイルを保存し、検証済みbackupと原本記録を照合して明示復旧する。空状態への置換や末尾イベントの自動切捨ては行わない。

## native起動の実行証拠

2026-10-24、Codex CLI **0.163.0**（`multi_agent` stable）を既存の
`agent_exec.sh --agent codex --profile ops_report --mode read-only` で実行した。
設定の変更・子の定義追加は行わず、profileの `agents.max_threads=3`、`agents.max_depth=1` を使った。
この確認は以下の実行環境の契約であり、上の固定scout経路を切り替える実装は後続の統合で行う。

- 実際に使えた起動引数は `spawn_agent(task_name=..., message=..., fork_turns="none")`。
  `agent_type` や固定役割名は指定しない。戻り値の `task_name` は `/root/<名前>` というnativeの名前空間で、
  子session IDそのものではない。`model` / `reasoning_effort` を省略した子4件すべての `turn_context` で、
  親と同じ `gpt-6-astra` / `max`、read-only sandbox、`approval_policy=never`、作業ディレクトリを確認した。
  子のmetadataのdepthは1。実効service tierは記録にないため未確認。
- 親のほかに子3件を重ねて起動でき、4件目は `collab spawn failed: agent thread limit reached` で拒否された。
  `wait_agent(timeout_ms=...)` の後、1件の完了を受領すると、残る子の完了前に後続を起動できた（成功総数4）。
  同時枠は完了で解放され、総生成数の上限ではない。`list_agents()` で完了状態も確認した。
  この環境に `close_agent` はなく、明示closeによる回収は未検証。完了した子を閉じる操作は後続起動に不要だった。

再確認は短いprobe用promptを用意し、既存runnerだけを使う。promptでは子2〜3件に短いsleepと定型文の返却を依頼し、
枠不足を1回確認後、完了受領と後続起動を行う。probeのためにprofileやCODEX_HOME設定を変更しない。
run dirは新規のものを指定する（#1448の確認は以下の `initial/` に保存した）。

```bash
probe_root="${XDG_STATE_HOME:-$HOME/.local/state}/tennis-lab-agents/campaign-1349/probe-1448"
bash .agents/ops/lib/agent_exec.sh --agent codex --profile ops_report --mode read-only \
  --task-id probe-1448 --cwd "$PWD" --branch codex/audit-native-evidence \
  --prompt-file "$probe_root/prompt.md" --run-dir "$probe_root/<新しいrun名>"
```

`native.inspect_run(Path(".../agent-runtime"))` はrunnerの `result.json` にある
`contract.runtime_home`・`runtime.session_id`・`runtime.turn_id` を起点とする。現在の環境変数や最新turnへ切り替えず、
`sessions/` と `archived_sessions/` の実transcriptを読む。親の `response_item` 内の
`function_call`（namespace `collaboration`、name `spawn_agent`）と `function_call_output` を `call_id` で対応させる。
子の `session_meta.source.subagent.thread_spawn` と `parent_thread_id`、`agent_path`、
`task_started.turn_attribution.parent_turn_id` から親runとの対応を確定し、`session_meta.id` をnative IDとして返す。
子の `session_meta.session_id` は親のIDであり、子threadのIDとは区別する。
`task_complete` と最後のassistant本文を照合した結果、開始・完了時刻、実効設定、transcriptのパス・内容hashも返す。
親の最終JSONは実行証拠として読まない。起動拒否は `rejections` に分け、起動成功へ読み替えない。

`native.reconcile(run_dir, {"台帳タスクID": "native ID", ...})` は、そのrunで実際に起動した子と
台帳の対応が一対一で全件完了している場合だけ、タスクIDをキーに実行証拠を返す。
統合側は `inspect_run` の `NativeAgent.task_name` でspawn戻り値を照合して `agent_id` を取得・保存する。
nativeの名前空間をsession IDの代わりに渡さない。複数runの台帳はrunごとに分けて照合する。
このAPIは1タスクにつき新規spawnを扱う。同じ親turnでの `followup_task` は、最初の完了の流用を防ぐため明示エラーにする。
タスクの問いと子の調査内容の意味的な一致は、このAPIでは証明しない。

未完了・中断した子は `inspect_run` の `status=incomplete|aborted` として明示し、`reconcile` は拒否する。
記録欠落、不正/重複JSON、spawn応答の欠落、親子/turn/ID/実効設定の不一致、重複transcript、
対応しない子は `NativeEvidenceError`。未確認のfork形式や起動エラーの形式も推測で補わない。
CODEX_HOME内のmetadata headerも厳格にparseし、破損があれば停止する。
既存の `agent.py` / `audit.py` への接続は後続の統合が行う。

自動テストは `tests/e2e/agents_ops/test_ops_report_native.py`。
fixtureの `fixtures/native_transcripts.json` は上記実行の必要なrecordを抽出し、実session ID・tool call ID・
filesystemパス・prompt本文を記号に置換したもの。`/root/...` はfilesystemパスではなくnativeの名前空間である。
実agentやGitHub書き込みは起動しない。

## 提案ID・チェック・triage

- レポートのタイトルは `運用レポート YYYY-MM-DD`（実行日ではなく予定回の日付）。提案IDは
  `R-<予定回YYYYMMDD>-<連番>`（例 `R-20261019-03`）。同じISO週にも複数回発行できる。
  重複判定は本文1行目の `<!-- ops-report:v2 slot=YYYYMMDD since=… until=… proposals=… status=complete|incomplete … -->` にある
  `slot` が正本。タイトルの変更・close・再試行でも同じslotのissueを再公開しない。
- **旧形式の自動移行・後方互換はない。** `ops-report:v1 week=` とstatusのないv2本文はreport/triageとも明示エラー。
  reportはopen/closedの履歴全件を検査し、triageは旧形式のopenレポートが1件でもあれば新形式を含め書き込み前に停止する。
  切替時点の実データは0件。旧形式が発見された場合は、勝手に新IDへ読み替えず、レポートと子issueの対応を人間が確認して
  `ops-report` 対象から外すまで新周期との併用を禁止する。
- 本文のうちtriageが解析するのは `<!-- ops-report:proposals:begin -->` 〜 `end -->` の間だけで、
  そこにあるIDもメタデータのslot・提案数の範囲内でなければエラーになる。agentが書いた文章（概要・提案本文・選択肢など）は
  チェックボックス記法とHTMLコメントを無害化してから埋め込むので、選択可能な項目やマーカーを偽造できない。
- 通常の提案は行頭のチェックボックスで選ぶ。選択肢つきの提案（放置PRの「継続 / rebaseして仕上げ / close」など）は
  **選択肢を1つだけ**チェックする。2つ以上、または見出しだけのチェックはエラーとしてtriageが非ゼロ終了する（他の提案は処理する）。
- 提案には目的・根拠・変更範囲・期待効果・受入条件・関連親issue・関連既存issueを描画する。
  agent出力の必須フィールドと型は `prompt.md` と検証器が定義する。関連親issueは指定できるが、
  親未指定・親セッション未登録でもIssue化する。**チェックは方向性の採用であり、要件合意・共有基盤承認ではない。**
- 日次triageはopenの完了レポートだけを対象にする。既存の提案メタデータを全open/closed issueから
  `index_proposals` で照合し、対応済みの提案はタイトル改名・close・正式親変更後も再利用する。
  新たな整理対象がない回は整理agentを起動しない。
- 未対応の採用提案を [Issue整理agent](organize_prompt.md) に渡す。単独作業、既存作業の再利用、
  既存親への子追加、新規親と複数子、理由付き保留を表現する。単独作業に不要な階層を作らない。
  open作業の既存担当・assignee、open PRの競合、重要な未決事項、明示停止は保留する。
  完了済み作業の過去のassigneeは保持して再利用できるが、open PR・明示停止はclosedでも保留する。親の強制付替え・issue削除はしない。
- 既に明示指定された親のregistry行や過去の受領記録がある提案は、その親の受領経路を維持する。
  activeなら `proposal_pending` を記録し、親自身が `receive-proposal` で受領する。停止・引継ぎ待ちでは
  状態を表示する。既存の人間合意経路を自律根拠へ書き換えず、triageが担当を引き受けることもない。
- 未チェックかつ `見送り理由: 未記入` は未判断。人間が理由を書いたものだけ明示的な見送りとする。
  採用と見送り理由の併記はエラー。採否と参照の専用memoryへの入力は下記「専用memoryの保存」を参照。
- レポートをcloseすると日次triage対象外。完了レポートの採否はclose後も次回調査の入力に含める。
- 同時実行: reportの同slot公開とtriageの引渡し・本文追記を共通issue lockで排他し、
  親stateの読取とコメントはparent CLIのregistry lockで直列化する。ロックは消して回避しない。
- `→ #番号` はtriageが検証して付ける。PRタイトル等の同じ形は `→ \#番号` にエスケープし、
  本文メタデータと一致しないリンクはエラーにする。

## 予定と状態

- `schedule init` を導入完了時に1回だけ実行する。その時刻より**後**に初めて迎える08:00 JSTがanchor。
  08:00ちょうどの初期化は翌日が初回となる。anchorから72時間（3日）ずつの08:00が予定回（slot）となり、月・年・閏日をまたいでも変わらない。
- 状態は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report/schedule.json`。
  未初期化・JSON破損・不正なanchor等はエラーで止め、今日を基準に作り直さない。`init` は既存ファイルを上書きしない。
  破損時はtimerを止め、ログと公開issueを照合して状態を明示修復する。
- timerは毎日08:00 JSTに `report --if-due` を起動し、`Persistent=true` で停止中の起動を回収する。
  初回前は何もしない。現在以前の最新slotが未公開なら1回だけ生成し、逃した古い回は範囲と件数で `skipped` に記録する。
  未公開で失敗したslotは同じslotで再試行でき、新しいslotを迎えた場合は古い未公開の失敗をスキップして最新の1回にまとめる。
  公開済み未完了回はスキップせず、上記の再開手順を優先する。完了レポートだけを成功とする。
  成功済みslotはagentを起動しない。開始の厳密な精度は保証せず、timerには最大10分の遅延を許容する。
- `schedule status` はanchor、`next_slot`（次の予定時刻）、`pending_slot`（未完了公開回があればその回、なければ未成功の最新回）、
  `last_attempt`（running / failed / succeededとエラー）、`last_success`、`skipped` をJSONで表示する。
  実行中でも確認できる。中断して `running` が残ったら再実行し、GitHubに完了公開済みなら成功状態を回復する。
- `schedule.lock` でreportの収集から公開後のメモリPR作成までを直列化する。triageのメモリPR確認もこのロックを使う。
  初期化も同じロックを使い、triageのissue処理とissue公開は別の共通ロックで排他する。
  状態は一時ファイルからのatomic replaceで保存する。ログ先・checkoutを変えても予定状態はマシン共通。
- 手動の `report` も同じ予定判定を使う。初回前と成功済みslotはエラー（`--dry-run` は成功済みslotのプレビューが可能）。
  任意日付の追加公開は提供しない。`--now` はtimezone付きISO日時を使うテスト専用の時計指定。

## 手動実行

```bash
cd ~/projects/tennis-lab
# 導入後の予定・前回結果を確認
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
# 本文を標準出力へ（issueは作らない）。agentは実行される。
.venv/bin/python .agents/ops/ops-report/ops_report.py report --dry-run --output /tmp/report.md
# 本番（ops-report ラベルのissueを作成）
.venv/bin/python .agents/ops/ops-report/ops_report.py report --agent codex
# 採用提案の引渡し（--dry-run で確認、子issueは作成しない）
.venv/bin/python .agents/ops/ops-report/ops_report.py triage --dry-run
```

実headlessでの証拠取得（マージ後に親が実機で行う）:

1. serviceの実効 `CODEX_HOME` と同じ環境で、native `scout` の定義・model/effort・read-only設定を確認する。
   `codex features list` でmulti_agentが使えることを確認する。scoutが無い/使えない場合は定義元を整え、別CLIで代用しない。
   env fileに旧 `OPS_REPORT_AGENT_TIMEOUT`、`OPS_REPORT_AGENT=claude|alternate` があれば削除/修正する。
   `systemctl --user daemon-reload` 後、`systemctl --user show tennis-lab-ops-report.service -p TimeoutStartUSec` がinfinityであることを確認する。
2. `schedule status` で予定回を確認し、予定が来たら `report --agent codex --if-due` を実行する。
   予定前に確認したい場合は `report --dry-run --agent codex` を使う（成功済みslotもプレビュー可能。初回slot前は実行しない）。
   serviceでの定期実行結果は `journalctl --user -u tennis-lab-ops-report.service` で確認する。
3. 実レポートURL、run dirの `audit_plan.json`（全対象と担当）、`audit.json`（全担当受領・2担当以上の時刻重複）、
   `collected.json` のbaseと専用memory入力SHA、CLIのJSONLログと実行環境のCodexセッション記録を保存する。
   agent IDをnativeの起動/完了記録と照合する。自己申告のJSONだけで実際の並列起動を証明した扱いにしない。
4. 未完了なら同じコマンドを再実行し、同じissueが完了へ移ることを確認する。
   完了後、次の `report --if-due` または `triage` でメモリ専用PRを確認し、マージ後の
   `origin/main:.agents/memory/ops-report/` の更新版とjournalのmergedを実レポートへ対応付ける。

自動テストは実Codex/Claudeを起動しない。`test_ops_report_audit.py` のfake担当が並列に動く時刻記録、
全返却、欠落/不正/重複/矛盾、無制限のwrapper設定、部分公開→同issue完了を検証する。
実レポートURL・native親子セッション・mainに反映した専用memoryの実機証拠は上の手順で別途取得する。

実行ごとに `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report/<時刻>-<command>[-dry]-<pid>/` へ
`run.log`、`collected.json`、`audit_plan.json`、`audit.json`、`prompt.md`、`agent_output.md`、agent CLIのJSONLログ、`report.md`、`result.json` を残す。
`--log-dir` / `OPS_REPORT_LOG_DIR` の相対パスは、起動時のカレントディレクトリを基準に解決する。
agentのCLIイベント・実効設定・使用量は各回の `agent-runtime/` に保存する（[共通ランナー契約](../lib/README.md)）。
gh・git・agent・掃除モジュールの失敗、agent出力の形式違反、設定値の不正はすべて非ゼロ終了になる（黙ってskipしない）。
agentと掃除モジュールは専用のプロセスグループで起動し、失敗・中断・終了時（掃除はタイムアウトも）にグループ全体を
SIGTERM → SIGKILL で止めて回収する（claude/codex が起動した子プロセスを残さない）。

集計期間は `[前回完了レポートの until, 今回の初回収集時刻)`（秒単位に切り捨て、メタデータと同じ精度にする）。境界は issue 作成時刻ではなく本文メタデータの `until`
（収集の基準時刻）で引き継ぐので、収集〜公開の間のイベントも欠落・重複しない。初回の開始境界はanchorの3日前で、初回が遅延してもその境界を維持する。merge済みPRは
`--search merged:>=<日付>` でmerge期間を検索条件に含めて取得し、件数上限に達したらエラーにする。

## 採用提案の自動Issue化

整理agentは [共通ランナー](../lib/README.md) の `--role implementation --mode read-only` を使う。
モデル値を複製せず役割既定のSol/maxを選び、設定出典・既定値・選択値・観測model/effort・終了理由を
`organization-runtime/` に保存する。正常なターンと実効設定・プロセス回収を照合してから計画を検証する。
`task_success:null` やexit 0だけでIssue化成功とは判定しない。別モデルや新sessionへの自動振替はない。

選定理由: `ops_report` のAstra/maxとnative scoutは全repoの並列調査用であり、提案済み要件を整理する
この限定処理には使わない。Opusは自律着手・開発親の統括を担う。整理は担当・着手の権限を持たず、
既存作業との比較と共通書式への具体化を実装子の既定で行い、機械的検証後にスクリプトだけが反映する。
実agentの意味的な整理品質は実機検証で別途確認する。

固定入力 `organization-input.json` は提案本文・全選択肢・選択値・内容digest・関連親と、全open/closed issue、
コメント、正式親子関係、先行依存、registryの担当/委任予約の投影、全状態PR、cleanなtrackedコードのcommitを含む。
既存 [dispatcherの収集API](../dispatcher/README.md#入力契約とdigest) を読み取り利用し、判定agentは起動しない。
全ページと参照先を取得し、取得失敗・重複・不正型・コード変更を空入力へ置き換えない。
計画の正本は `organize_prompt.md`。全対象・未知キー・出典・循環・共通本文書式・担当競合を全件検証する。

出典の正本は [issuesの出典契約](../issues/README.md#triageによる出典と復旧契約)。各提案は1つの索引anchorを持ち、
分割時の親または実作業子へ対応付ける。複数提案をまとめる親は各提案の別anchorを子に持ち、contextの出典を保持する。
従来の単一出典索引を維持し、dispatcherはanchorとcontextの全bindingから `source_reports` を収集する。
正式親が変わった後もcontextだけを持つ親・分割子の出典を対象閉包に含め、採用変更を検出する。
目的・範囲・受入条件・選択肢・元の提案内容とdigestはスクリプトが出典節へ引き継ぐ。

既存のホスト共通triage lockとparent registry lock内で、計画をstate dirの `triage-<repo digest>.json` へ
0600のatomic replaceで保存してから起票する。これは担当台帳ではなく、Issue反映のjournalである。
各書込前後に採用・選択肢・提案本文・全入力・担当を照合し、自分の正確な書込みだけを差し引く。
親一覧やレポートの `引渡し` / `→ #N` は自分の表示として区別し、人間の本文・通常コメントを無視しない。
作成応答を失っても、本文の計画digestとnode識別子から全open/closed issueを探し、不足する正式関係・親一覧・リンクだけを揃える。
既存issueのstate・担当・親を戻さない。同一ホストの同時実行・手動親CLIとの競合は同じlockで直列化する。
POST試行前もjournalへ記録する。応答が曖昧で全ページにも同じissueが見つからない場合は再POSTせず停止する。
作成されていないことをGitHub・runログで明示確認できた場合だけ、そのnodeのattempted記録を退避付きで修復する。

保留には理由・根拠・再開条件・次の対応を `引渡し` に残す。同じ入力の保留は計画を再利用し、agentもコメントも連投しない。
採用取消・本文/選択変更後は既存作業を保持して停止し、元の採用へ戻すか別提案として整理する。
journal破損・欠落や出典矛盾は自動初期化しない。run dirとGitHubの計画marker・実番号を照合してjournalを復元し、
同じtriageを再実行する。未照合のissueを消したり、別提案ID・別nodeで作り直したりしない。
既存計画の再開・新しい整理agentの起動・書込みの前に、全open/closed issueのtask markerをjournalへ照合する。
索引anchorとなる子が未作成でも、contextだけの共通親から計画・node・bindingsの一致を検査する。dry-runも同じ検査を行う。
取引が見つからなければissue番号・計画と復旧方法を示して停止する。空のtransactionsを作っても解除しない。
保存済みの `organization-input.json`・`organization-plan.json`・公開draft・GitHubの `ops-report-task` markerを
元のrunログと照合し、対応するjournalを復元してから同じtriageを再実行する。

生成した通常の作業issueは、次の08:00・20:00 JSTの予定回にdispatcherが収集する。専用enqueueやreadyラベルは不要。
triageから割当判定agent・開発親・実装子・レビュー・mergeは起動しない。採用・Issue整理・自律着手根拠は別物で、
後の委任では [parentの出典検証](../parent/README.md#子の作成委任再開) と有効な実行根拠を照合する。
開発workflowは、単一anchorのないcontext出典も含め `parent.adopted_tasks.check_task_adoptions(gh, body)` を
親・対象子の両本文へ実行直前とmerge直前に適用する。この返却は出典の現状確認であり、実行許可ではない。
#1335のworkflowがこの呼出しとdispatcherの外部入力鮮度検査を担う。先行parent/reviewの根拠・gateは書き換えない。

検証は `test_adopted_proposal_issues.py` と既存 `test_ops_report_triage.py`。fake GitHub・fake Codex CLI・一時state・
使い捨てgitで生成/再利用/集約、全ページ、保留、応答喪失・中断・再実行、出典/採用変更、既存の受領記録を検証する。
判定側は共通ランナーとfake Claude CLIを通し、分割入力・構造化出力・結果v2の一部延期と独立候補の維持を確認する。
実レポート #1280 の採用4件から予定回の割当までの実CLI/GitHub/systemd証拠は #1336 に引き継ぐ。

## 設定

CLI引数が優先。systemd実行時は `~/.config/tennis-lab-agents/ops-report.env`（任意）に書く。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `OPS_REPORT_AGENT` | `codex` | native scoutを起動する親。現在はcodexだけ（他の値はエラー） |
| `OPS_REPORT_BASE_REF` | `origin/main` | 期間内commit・行数・TODO集計の対象ref。remote追跡refなら実行開始時に `git fetch <remote> <branch>` し（失敗はエラー）、そのcommit SHAを実行中固定して本文メタデータ `base=` に記録する。ローカルref/SHAはfetchせず、本文に「ローカルref、fetchなし」と明記する |
| `OPS_REPORT_STALE_PR_DAYS` | `7` | 何日更新がなければ放置PRとして載せるか |
| `OPS_REPORT_CLEANUP_CMD` | `<python> .agents/ops/cleanup/cleanup.py scan` | 掃除モジュールのコマンド。`--report-json <一時ファイル>` を自動付与し、そのファイルのJSONを読む |
| `OPS_REPORT_LOG_DIR` | 上記state dir | ログの保存先 |
| `OPS_REPORT_AGENT_EXEC` | `../lib/agent_exec.sh` | 全体調査のagentラッパー（テストで差し替える） |
| `OPS_REPORT_ORGANIZER_EXEC` | `../lib/agent_exec.sh` | 提案整理の共通ランナー（テストで差し替える） |
| `OPS_REPORT_LOCK_FILE` | `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report.lock` | issue書き込みの排他ロック（下記） |
| `OPS_REPORT_LOCK_TIMEOUT` | `600` | ロック待ちの上限（秒）。超えたら非ゼロ終了 |

- 放置PRへの推奨: 完了には対象すべての `{"recommendation": "継続|rebaseして仕上げ|close", "reason": …}` が必要。欠け・余分・形式違反は未完了になる。
- 掃除モジュール: `OPS_REPORT_CLEANUP_CMD` が未設定で `.agents/ops/cleanup/` も無ければ、本文に「掃除モジュール未導入」と書く。
  ディレクトリがあれば `cleanup.py scan --report-json <一時ファイル>` を実行してJSONをそのままagentへ渡す（`cleanup.py` が無ければエラー）。
  共有memory（`.agents/memory/`）も同様に、無ければ「未導入」と書く。あれば `SUMMARY.md` と、層の `index.md` を除く各エントリの先頭を渡す。
- 共有memoryの収集対象は固定base版の workflow / decision / environment / gotcha / review の5層だけ。
  専用memoryは以下の独立入力・保存経路を使う。
- `--repo-root` は収集元repo（既定はこのcheckout）。その変更/未追跡パスも収集するが、調査担当は固定baseの専用checkoutを読む。

## 専用memoryの保存

保存先は [`.agents/memory/ops-report/`](../../memory/ops-report/SUMMARY.md)。観察の内容・再検証・圧縮・削除は
[memory skill](../../skills/memory/SKILL.md#運用レポート専用memory) を正本とする。共有5層の保守（#1192）は行わない。

入力 `ops_memory.files` は、新しい予定回でfetchした `origin/main` の固定SHAから読む（再開は固定入力を維持する）。
`ops_memory.base_sha` と前回保存の状態はレポート冒頭にも表示する。未導入ならその旨を表示し、更新案が返ればエラー。
agentはread-onlyのまま、`memory_update`（`summary` / `upsert` / 理由付き`delete`、変更不要は明示的な `null`）を返す。
パスは受け取らずslugから生成し、親スクリプトだけが書く。未指定の観察は維持し、削除と索引の更新は同一commitにする。

全体調査が完了した場合だけ、公開と保存を次の順に進める。未完了レポートを確認しても保存を開始しない。

1. agentの更新JSONを検証し、`${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/ops-report/memory.json` に
   mainの元データと更新案、公開予定のタイトル・本文SHA-256・認証済みGitHub投稿者をatomic replaceで記録する。
   この `prepared` は公開成功でも有効メモリでもない。
2. 完了レポートissueを公開（または未完了から予定タイトル・本文へ更新）する。新規作成では作成応答のURL、
   更新では編集対象issueのURLを編集前に記録する。失敗時はメモリをcommit/pushしない。
   編集失敗で未完了のままなら、同じURL・投稿者のissueに限って保存を保留し、次回再開する。
   通常実行と公開後の中断からの復旧では、同slotの完了履歴のタイトル・本文ハッシュ・投稿者が準備時の値と一致し、
   作成応答を記録済みならURLも一致した場合だけ公開確認とする。不一致なら停止する。
   更新では編集前に記録したURLとの一致も必須とする。
   確認後はURLと投稿者を照合し、タイトル変更や提案チェックの編集を許容する。
3. 公開を確認した更新だけ、元repoの `.claude/worktrees/ops-report-memory-<slot>` に、入力の `origin/main` SHAから専用branch
   `ops-report-memory/<slot>` を作ってcommitし、originへpush、main向けのメモリ専用PRを作る。メインcheckoutは編集しない。
   `null` は `unchanged` と記録する。ログの `memory_result.json` とjournalに段階・PR・エラーを残す。
4. **次回reportまたは日次triage**で保留PRを1回確認する。CI待ちループは持たず、初回のPR作成直後にはマージしない。
   レポートjobを長時間占有せず、既存の日次triageを利用して新しいtimerも不要にするためである。
5. 変更commitが期待したmainを唯一の親とし、追加・削除・rename元を含む全変更が `.agents/memory/ops-report/**` に閉じること、
   PRのbranch・base・同一repo・本文マーカー・HEADが記録と一致することを検証する。
   mainの専用memoryが入力後に進んでいれば古い更新で上書きせず停止する。
6. [`required_checks.txt`](required_checks.txt) の必須CIがすべて存在し成功し、他のcheckも成功、競合なしの場合だけ、
   PRとCIを再取得してHEADを照合し `gh pr merge --merge --match-head-commit <head>`。
   CI状態の正規化は既存の `review_policy.normalize_rollup` を使い、PRレビューgateは呼ばない。
   通常PRのmerge gateがCIを条件にしなくても、この専用経路のCI要件は維持する。
   GitHubのmergeとorigin/mainへの取り込みを確認して初めて `merged` と記録し、cleanな専用worktreeを削除する。

この専用経路は人間と合意済みのメモリ限定の例外で、Codexレビューや `AGENT_AUTO_MERGE` を要件にしない。
branchに `codex/` / `claude/` prefixやagentラベルを付けず、通常PRのレビュー対象と混ぜない。
PR作成・CI未完了をメモリ更新成功とみなさない。CI未登録/未完了は保留、CI失敗・保存失敗・HEAD不一致・競合・PRの未merge closeは
エラーとして非ゼロ終了し、他のレポート生成やtriageは進める。mainの観察が進んだPRと未mergeでcloseされたPRは
`failed` として記録を残し、以後は再試行しない（次の調査から新しい更新を作る）。レポートの予定状態の成功は**完了issueの公開**を意味し、
メモリの成功はjournalの `merged` / 更新不要の `unchanged` で区別する。
次のレポートは保留/失敗/中断を冒頭に「専用メモリ未反映」と示し、mainの最後の有効版を使う。

採否の入口: 全履歴の完了レポートをGitHubから読み、採用チェック・明示された見送り理由・親/子issue参照を
`ops_memory.decisions`（`proposal_id` / `decision: selected|deferred` / `reason` / `source`）へ自動で渡す。
未チェックだけでは見送りにしない。専用memory更新の親はこの入力を根拠に採否・再検証条件を保存し、
同じ見送り案を新しい根拠なしに再提案しない。提案本文や進捗は複製せずissue参照を残す。
日次triageはrun dirの `decisions.json` に同じ契約で出力する。
`report --memory-decisions <JSONファイル>` は任意の照合用snapshotで、現在のGitHubと一致する行だけを受け付ける。
古い採否をローカルファイルで上書きしない。未完了レポートの採否は入力に含めず、再開時の固定入力も差し替えない。

復旧は次のように行う（ホスト操作は親が実施）。

- `run.log` / `memory_result.json` / journalの `stage` と実際のissue・PR・branchを照合する。JSON破損はエラーで止まり、初期化し直さない。
  journalを手動修復する場合は先に退避する。`prepared` は準備時の公開情報と履歴が一致するときだけ再開される。
  公開照合情報を持たない旧journal（version 1）は自動移行せず停止し、公開issueとの明示的な照合・修復を必要とする。
- push/PR作成/APIの一時失敗は原因を直し、`report --if-due` または `triage` を再実行する。
  commit直後・PR作成直後の記録中断は、期待するcommit内容や既存PRを照合して再利用し、重複公開しない。
- 保存途中で専用worktreeがdirtyになった場合は自動resetしない。ログと差分を確認し、対象slotの専用worktreeだけを修復する。
  commit前なら入力mainのclean状態へ戻して再実行する。commit済みbranchを手動削除・付け替えしない。
- CI失敗は原因解消・同じHEADのCI再実行後に確認する。内容修正でHEADを進めると照合は失敗する。
  HEAD/内容不一致、mainの観察の先行更新、未merge closeは人間がPRとjournalを確認する。
  古いPRを勝手にrebase/force-pushせず、次回のmain入力から新しい観察を作る。失敗の記録を成功へ書き換えない。
- `report` とtriageのメモリ処理は同じ `schedule.lock` で排他される。ロックファイルを消して並行実行しない。

テストは `test_ops_report_memory.py`。fake gh / fake agentと実gitで、2回の引き継ぎ・採否、共有5層のハッシュ不変、
公開/push/PR失敗・CI保留/失敗・HEAD更新・同時起動・保存中断・mainの先行更新・削除/圧縮を検証する。

## systemd --user への導入（merge後に人間が行う）

旧unitが動いたまま新unitを導入しない。以下は親が実機で行う。installerは旧unit本体またはtimerのwantsリンクが
残っていれば、リンク作成・daemon-reload・enableより前にエラーにする（壊れたsymlinkや他のwants内も検出）。
旧source unitも再導入しない。旧PR自動レビューunitが残る場合は [reviewの撤去手順](../review/README.md) も先に実施する。

```bash
# 旧reportと、切替中に古いCLIを呼び得るtriageを止める
systemctl --user disable --now tennis-lab-weekly-report.timer tennis-lab-ops-triage.timer
systemctl --user stop tennis-lab-weekly-report.service tennis-lab-ops-triage.service
rm -f ~/.config/systemd/user/tennis-lab-weekly-report.{service,timer}
rm -f ~/.config/systemd/user/timers.target.wants/tennis-lab-weekly-report.timer
# 任意の旧env fileがある場合だけ移す。旧変数はCLIがエラーにする。
if [ -f ~/.config/tennis-lab-agents/weekly-report.env ]; then
  mv ~/.config/tennis-lab-agents/weekly-report.env ~/.config/tennis-lab-agents/ops-report.env
  sed -i 's/^WEEKLY_REPORT_/OPS_REPORT_/' ~/.config/tennis-lab-agents/ops-report.env
fi
cd ~/projects/tennis-lab
.agents/ops/install_units.sh --dry-run
.agents/ops/install_units.sh                    # link + daemon-reload
systemd-analyze --user verify .agents/ops/ops-report/systemd/*.{service,timer}
# 導入完了後に基準を記録してからtimerを有効化（既存stateがあればinitは繰り返さない）
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule init
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
systemctl --user enable --now tennis-lab-ops-report.timer tennis-lab-ops-triage.timer
# unit実効値、日次の次回check、3日周期の次回slotを照合
systemctl --user cat tennis-lab-ops-report.service tennis-lab-ops-report.timer tennis-lab-ops-triage.service
systemctl --user show tennis-lab-ops-report.timer -p NextElapseUSecRealtime -p LastTriggerUSec
systemctl --user list-timers --all 'tennis-lab-*'
.venv/bin/python .agents/ops/ops-report/ops_report.py schedule status
journalctl --user -u tennis-lab-ops-report.service
```

旧timer/serviceがinactive・disabledでunitリンクも無いこと、新timerだけがreportを起動することを確認する。
`list-timers` の次回は日次の予定判定時刻であり、実際にreportを生成する予定回は `schedule status` で確認する。
旧state/logは新stateに読み込まない。必要なログを保管後、旧 `weekly-report/` と `weekly-report.lock` を親が撤去する。
既存 `ops-report` ラベルの説明は `運用レポート（.agents/ops/ops-report）` に親が更新する（既存ラベルはCLIで上書きしない）。

unitは `%h/projects/tennis-lab`（メインcheckout）と `.venv/bin/python` を使う。`claude` / `codex` / `gh` は
`~/.local/bin` か `/usr/bin` にある前提で `PATH` を設定している。

## 改訂の仕方（固めすぎない）

- 分析観点・品質基準・出力形式は `prompt.md` にある。観点の追加や言い回しの変更はここだけ直せばよい。
  レポート自身が「制御ルールの改訂」カテゴリでこのプロンプトや本ツールの改善を提案してよい（プロンプトにそう書いてある）。
- カテゴリを増やす場合は `opsreport/proposals.py` の `CATEGORIES` を変える（プロンプトのカテゴリ一覧は自動で反映される）。
- 収集項目を増やす場合は `opsreport/collect.py` に決定的な関数を足す。観察データ欄に出したいものは `opsreport/report.py`。
- 挙動を変えたら `tests/e2e/agents_ops/test_ops_report*.py`（fake gh / fake agent_exec）を更新する。

## 改善ループ全体の実機E2E（親が実施）

#1218 / #1219のmerge後、親が実GitHub・実ホストで1件を最後まで検証する。fakeテストの通過だけで
この検証を完了扱いにしない。証拠は親issueにリンクをまとめ、子の実装詳細は各子issueへ置く。

1. 完了レポートの提案ID・本文・採用チェックと、未採用時に起票/実装0件である証拠を記録する。
2. 親未指定でtriageを実行し、採用提案だけが作業issueへ生成・再利用されること、出典digestとレポートリンクを確認する。
   同じtriageを再実行し、重複起票・予定外の割当判定agent/開発親起動が0回であることを記録する。
3. 次の08:00・20:00 JST予定回で作業issueが収集され、独立した自律着手根拠と担当が記録されることを確認する。
   人間合意・共有基盤承認が生成されないこと、取消・重要情報不足・既存担当では理由付きで停止することを確認する。
   既存の手動親への受領・人間合意経路も、独立した作業を止めず継続できることを確認する。
4. 親が対象PR群を確定して明示指示し、統合Codexレビューが1回だけ実行される証拠を保存する。
   未指示時は0回、指摘を担当へ戻して修正・CI/差分確認した後の自動再レビューも0回とする。
5. #1219のgateでcoverage・最新HEAD/base/diff・個別CI・共有基盤条件・依存関係を照合する。
   先行依存を先にmergeし、独立PRのCI待ちだけでは合格済みPRを滞留させず順次mergeする。
   各merge後に残りのcoverageと最新対象を再照合する。
6. 次回レポートの採否・issue/PR参照、専用memoryのmain反映を確認する。
   人間が独立に作ったissueでも同じregister/checking/agree入口を確認する。
7. #1218の導入手順に従い、旧10分周期review timer/serviceの実効停止・無効化・リンク撤去、
   installer再実行で復活しないこと、PR open/push/CI完了だけではreview呼出0回を確認する。
   全体レポートtimerは継続する。systemctl実効値・ログ・GitHub URL・review run IDを証拠として残す。
