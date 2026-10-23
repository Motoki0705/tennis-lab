# 親セッションの担当・進捗・レビュー要求

親issueを1セッションで統括するための明示CLI。人間合意と自律着手判断を別の実行根拠として扱う。
会話の解釈・質問・委任・完了確認は [parent-session skill](../../skills/parent-session/SKILL.md)、
親子issueの書式・ラベル・段階移行は [issues](../issues/README.md) が正本。
このCLIは agent・レビュー・CI・merge を実行しない。timer・poll・hookへの登録も行わない。

Herdrでの担当表示は[herdr_labels](../herdr_labels/README.md)へ連携する。
担当開始・再開・引継ぎ・終了後の`display_sync`で反映結果を確認できる。
同じ位置で[board](../board/README.md)の名前も同期し、結果を`board_sync`へ返す。

## 責務と状態の正本

| 対象 | 正本 |
|---|---|
| 親issueと担当セッションの対応、解除・終了・引継ぎ待ち、委任の起動予約、worktree | マシンローカルの `registry.json` |
| 担当開始、要件確認、要件合意と人間の確認の引用 | 親issueコメント |
| 自律着手の根拠・入力・コメント分類・取消 | 親issueコメントの `execution_basis` / `basis_revoked` イベント |
| 正式な親子関係、子のopen/closedと理由 | GitHub sub-issue / issue |
| 子の具体的な停止理由・根拠・再開条件・次の対応、改善候補 | 子issue本文 |
| 子の一覧と全体進捗、停止/再開通知、レビュー対象・指示・状態 | 親issue本文・コメント |
| レビュー対象のバイト列 | ローカル `reviews/<run ID>/pr-<N>.diff` と `request.json` |

stateは `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
同じホストの全worktree・Claude/Codexで共有し、state内のflockで登録・起票・更新を直列化する。
`registry.json` の外側はversion=1を維持し、`parents` のキーは `owner/repo#親番号`。
キーの `owner/repo` は大文字小文字を区別せず、小文字に正規化する。
既存キーも読込時に正規化し、同じ親の別名レコードが衝突した場合は、統合・上書きせず読込を拒否する。
各行に `session`（実行環境の識別子）、`status`、`generation`、`tasks`（子番号→担当/branch/worktree/依存/委任予約）、
`reviews` を持つ。ファイルは0600、ディレクトリは0700で作り、fsyncとrenameで保存する。
破損・未知version・取得失敗を空の状態に置き換えない。

移行はtasks/reviews内へのversion付き根拠・要求の追加だけとし、外側のversionを上げて運用中の旧CLIを読込不能にしない。
旧行へ架空のauthorizationを補わず、新しい操作時だけ新形式を記録する。読み取りだけでは書き換えない。
担当・generation・tasks（委任予約を含む）・reviews・拡張フィールドを保持し、過去の証拠を最新の根拠に付け替えない。
旧taskにauthorizationが無い場合も予約は維持する。自律根拠で既存成果を使うときは、親が差分・要件・検証を確認し
`record-result` で現在の根拠へ明示的に結び付ける。JSONの重複キー・不正型・未知のreview schemaは読込エラー。
旧公開マーカーと保存済みrequest/result/executionは書き換えず、旧request version=1を引き続き照合する。
導入前の全担当停止や実registryの削除・一括再登録は不要。

識別子は実行agentの環境から取得する。Claudeは `CLAUDE_CODE_SESSION_ID`、Codexは `CODEX_SESSION_ID`。
欠落・空値・両方の混在はエラーにする。人間にIDを聞いたり、引数・PID・作業パスで補ったりしない。
子agentから担当親を装って登録せず、担当する親セッション自身が実行する。
Codexの `CODEX_THREAD_ID` は現在のthread、`CODEX_SESSION_ID` は親子で共有するroot session。
両者が設定されていて異なる場合はsubagentからの操作として拒否し、thread IDを代用しない。
根拠はCodex CLI 0.163.0の公式 [exec_env.rs](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/exec_env.rs#L27-L46) と
[process_manager.rs](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/unified_exec/process_manager.rs#L1414-L1428)。
環境を提供しない起動経路では、この入口を使用できない。
セッションIDはissue・コメント・memoryへ出さず、CLIの通常出力にも含めない。
worktreeの絶対パスと再起動証拠はローカル出力にのみ含まれるため、それを公開コメントへ貼らない。

1セッションのactive/引継ぎ待ち親は1件。同じ親の別セッションによる登録も拒否する。
終了をプロセス一覧などから推測せず、親自身の `release --ended` で記録する。
別セッションへの移管は `handoff --accept` による明示操作だけを許可する。
マシンをまたぐ分散排他やセッションの自動選定は範囲外。

## 入口と合意

以下のコマンドは会話を受けたagentが実行する。人間に実行させる手順ではない。
共通引数はサブコマンドの前に置く。`--repo` 省略時は `gh repo view` で特定する。
`--parent` は番号、`#番号`、同じrepoのHTTPS issue URLを受け取る。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 register
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent https://github.com/Motoki0705/tennis-lab/issues/1216 status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 checking
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --spec tmp/agreement.json
```

`agreement.json` は `purpose` / `scope` / `behavior` / `acceptance` / `human_confirmation` の
5キーだけを持つJSON object。値はすべて非空の文章とし、最後の値に人間の確認発言を引用する。
`checking` → 会話で認識合わせ・合意 → `agree` の順。担当指定だけでは「担当指定のみ」、
`checking` で「要件確認中」、有効な合意コメントで「要件合意済み」となる。
再度 `checking` に戻したら、新しい合意の記録まで分割・委任を拒否する。
`checking` / `agreed` のrevisionは両イベントの過去最大値から進め、取消の連番とは分ける。
取消後の `checking` は新しいrevisionを作り、人間が同じ内容へ再合意しても新しい `agreed` を残す。
取消による要件確認中の状態から直接 `agree` で実際の再合意を記録する場合も、新しいrevisionを使う。
同じ確認中の `checking` と、有効な同一合意への `agree` の再送は冪等で、新しい合意に数えない。
会話に実際の合意があるかを判断する責任は親agentにあり、CLIは人間の確認を捏造しない。
既存issueにマーカーなしの人間との合意がある場合は、その発言と合意内容を読み、
`checking` / `agree` で同じ内容を記録する。記録形式を移すためだけの再承認は求めない。

コメントの1行目は `<!-- parent-session:v1 <JSON> -->`。
JSONは `parent`、`event` とイベント固有の値を持ち、`<` / `>` をUnicode escapeする。
現在の `gh api user` と同じloginによるコメントだけを読み、本文中の引用マーカーを無視する。
合意には構造化した `agreement` と、そのcanonical JSONの `agreement_sha256` を含める。
canonical JSONはUTF-8、`ensure_ascii=False, sort_keys=True, separators=(",", ":")`。
再実行時は同一イベントを再投稿しない。別ghアカウントへの移管は自動で信用せず、
新しい担当アカウントで改めて要件確認・合意を記録する。

人間合意でも親の要件本文とコメントの鮮度を分割・委任・レビュー・merge前に検査する。
新しい `agreed` は要件本文の `requirements_sha256` と、通常コメントのID/login/digestを `comment_inputs` に固定する。
本文は自律根拠・reviewと共通の `requirement_body` 射影を使い、進捗の除外範囲も
[自律着手の実行根拠](#自律着手の実行根拠と呼出契約)と同じ。実質的な本文変更は `checking` → 実際の認識合わせ → `agree` を要する。
同じ `agree` の再送やコメントの無関係分類で変更後の本文を取り込まず、review-prepare前にも拒否する。
現在のpublisherによる先頭行の `parent-session:v1` / `parent-review:v1` 記録を除き、
コメントの追加・編集・削除は未分類の変更として操作を拒否する。
`issue-start-judgement:v1` の表示も、[判定コメントの厳密な識別条件](../dispatcher/README.md#判断結果と保留コメント)を
満たすものだけ同じく除外する。合意時の旧snapshotに含まれる表示も、現在の形式を検証してから除外し、元の記録は書き換えない。
エラーは対象のID・URL・変更種別と解除方法を示す。`status` や旧記録の読取は維持し、
snapshotのない旧合意は合意コメントのID・作成時刻と通常コメントのID・編集時刻で照合する。
旧マーカー・registry・保存済みレビューを読取時に書き換えない。

親が対象コメントを読んで要件と無関係と確認した場合だけ、次の明示入口で分類できる。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agreement-ignore-comments --ignored-comments tmp/ignored-comments.json
```

入力は自律根拠の `ignored_comments` と同じ
`[{"id":123,"sha256":"parent.execution.comment_snapshotのcanonical JSON digest","reason":"要件と無関係な理由"}]`。
未分類変更をすべて含め、現存する本文・login・IDのdigestを指定する。重複・古いdigest・欠落は拒否する。
`event=agreement_comments` に元の合意digest/revision・分類連番・現在のsnapshot・分類理由を記録し、
実行根拠・要件・担当generation・既存レビュー・共有基盤許可は変更しない。再送は冪等。
分類済みコメントも編集・削除を再検査する。人間の指示を無関係として除外してはならない。
要件変更・停止指示・削除への対応は `checking` → 実際の認識合わせ → `agree` で再合意する。
同じ `agree` の再送だけでは鮮度不一致を解除せず、再合意後も旧レビューを流用しない。

`create-child` と `delegate` は、ローカル担当がactiveであり、親がopenであり、
親コメントに有効な人間合意または自律着手根拠があることを**書き込み前**に検査する。
合意後の分割結果への追加承認は設けない。起票元が人間・レポートのどちらでも登録入口は同じ。
単独で完結する既存issueは登録・合意してそのまま扱い、不要な親子階層を作らない。
分割するときだけ、issues規約に従ってそのissueを親へ移行する。

## 自由文の進捗・連絡

担当親が親issueへ自由文の進捗・連絡を公開するときは、素のコメントではなく次の入口を使う。
`sync` 等の既存マーカー付き記録は従来どおり使う。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 note --body-file tmp/note.md
```

activeな担当セッション本人だけが実行でき、要件確認中でも使える。本文の前後の空白を除き、
空・60,000文字超・HTMLコメント記号（`<!--` / `-->`）または予約マーカー名
（`parent-session:` / `parent-review:`、大文字小文字を問わない）を含む本文は拒否する。
他の公開記録と同様、ローカルのセッションID・state/worktree情報を含む公開も拒否する。
先頭行に現在のpublisherの `parent-session:v1` マーカーを付け、`event: "note"`、
担当generation、正規化後の本文のcanonical JSON digest `body_sha256` を記録する。セッションIDは含めない。
出力は `{parent, body_sha256}`。同じ担当generation・本文の再送は再投稿しない。

noteは合意後コメント検査、自律根拠の指示コメント射影、review/mergeの要件digestから除外され、
根拠・要件・既存レビュー・共有基盤許可を変更しない。noteに要件を書いても要件変更の合意にはならない。
要件変更は従来どおり `checking` → 人間との認識合わせ → `agree` で扱う。
note後の通常コメントも毎回検査し、同じ本文でもマーカーのないコメントは自動除外しない。

人間とagentが同じGitHub loginを使うため、loginから両者を区別できない。
人間が有効なnoteマーカーを偽装して書いた指示は進捗と区別できず、検出できない。
この形式を人間の指示に使わず、親も人間の指示をnoteへ移して除外してはならない。

### 稼働中の親への移行

`requirements_sha256` がない旧合意（#1204・#1216・#1261・#1318等を含む）は、`status` と履歴の読取を維持する。
ただし合意時点の本文を証明できないため、分割・委任・レビュー準備/指示・merge等の実行根拠検証は
`legacy human agreement lacks requirements_sha256` と復旧手順を示して停止する。従来動作への黙った切替は行わない。
担当親は現在の本文・指示を確認し、`checking` → `agree` で本文digestを記録する。
旧合意の記録後に本文の要件（目的・範囲・受入条件・停止事項）と指示コメントが実質的に変わっていないことを
親がGitHubの編集履歴・コメントで確認できた場合は、上記の「記録形式を移すためだけの再承認は求めない」に従い、
同じ合意内容と元の人間の確認発言の引用で記録してよい。実質的な変更があるか確認できない場合は、人間と再合意する。
現在の本文digestを旧記録へ自動補完せず、registry・既存task・旧reviewを読み取りや移行で書き換えない。
再合意後も旧reviewを新しい根拠へ付け替えず、現在の要件で成果を検証し、新しい理由付きrunでレビューする。

本文digestを持つ親に合意後の素の進捗コメントがある場合、merge後の次の委任・レビュー・mergeで
未分類コメントとして停止する。registryや合意を一括で書き換えず、担当親が次を一度行う。

1. 拒否メッセージの全ID・URL・変更種別から本文を読み、進捗だけか、要件変更・停止指示かを確認する。
2. 要件と無関係な現存コメントだけなら、次の例で現在のsnapshotのdigestと個別理由を作る。
   `reasons` は例のIDを、確認した全未分類コメントのIDと理由に置き換える。#1216では取得先とCLIの親番号も変える。

```bash
gh api --paginate --slurp repos/Motoki0705/tennis-lab/issues/1204/comments > tmp/parent-comments.json
PYTHONPATH=.agents/ops .venv/bin/python - <<'PY'
import json
from pathlib import Path
from parent.execution import comment_snapshot
from parent.policy import digest

pages = json.loads(Path("tmp/parent-comments.json").read_text(encoding="utf-8"))
comments = {c["id"]: c for page in pages for c in page}
reasons = {123: "本文を確認済み。合意済み範囲の進捗だけで、要件や停止の指示を含まない"}
ignored = [
    {"id": n, "sha256": digest(comment_snapshot(comments[n])), "reason": reason}
    for n, reason in sorted(reasons.items())
]
Path("tmp/ignored-comments.json").write_text(
    json.dumps(ignored, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 agreement-ignore-comments --ignored-comments tmp/ignored-comments.json
```

3. 元の操作を再実行する。分類は要件・根拠・担当generation・既存レビューを変更しない。
   分類後の本文編集や新しい通常コメントは再検査する。以降の自由文の進捗には `note` を使う。

要件変更・停止の指示や削除が含まれる場合は分類で解除せず、`checking` → 実際の認識合わせ → `agree` で再合意する。
旧レビューを変更後の要件へ流用しない。人間の指示を無関係と自動分類する移行処理は設けない。

## 自律着手の実行根拠と呼出契約

判定agentの出力を直接登録しない。着手判断側のスクリプトが出典・重要な未決事項・人間の停止/見送り/対象外・
採用状態・正式関係・既存担当/PRを検証し、着手可能な対象だけを次の入口へ渡す。
このCLIは自然言語の判断器ではなく、構造・同一性・鮮度・担当を検査する。

```bash
# 判断材料を固定する。--context は関連コード/依存/採用等のsnapshot参照・digest JSON（省略時{}）
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-input --context tmp/context.json > tmp/inputs.json
# スクリプトが検証済みの判断を構造化してから公開する。担当登録は行わない
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-record --spec tmp/basis-record.json
# 予約直前の読み取り検査。CLIの終了と別プロセスの予約をatomicとは扱わない
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-check --basis-sha256 64桁hex --generation 1
```

記録specは追加キーなしの `{basis, inputs, ignored_comments}`。
`inputs` はbasis-inputの出力そのもの、`ignored_comments` は通常 `[]`。
`basis` は次のversion付きJSON object（追加キーなし）。

```json
{
  "schema_version": 1,
  "kind": "autonomous_decision",
  "purpose": "目的",
  "scope": "対象と対象外",
  "behavior": "期待する動作",
  "acceptance": "完了条件と検証方法",
  "rationale": "この範囲で着手可能と判断した理由",
  "sources": ["https://github.com/owner/repo/issues/1216", "関連コードのcommit/path等の参照"],
  "input_sha256": "inputsのcanonical JSON SHA-256（64桁hex）",
  "policy_version": "autonomous-start/v1"
}
```

文章とsourcesは非空。`human_confirmation` や共有基盤の許可を入れた自律根拠は拒否する。
`basis_sha256` はbasis全体のcanonical JSON digest。人間合意は既存のagreedイベントを
`kind=human_agreement, policy_version=human-agreement/v1` の同形式（実際のhuman_confirmation付き）へ
読み替える。合意原文・agreement digest・公開マーカーは保持し、自律判断からagreedを作らない。
本文digestを持つ人間合意の `input_sha256` は `{agreement_sha256, requirements_sha256}` のcanonical JSON digestとする。
本文digestのない旧合意は読取時だけ従来のagreement digestで表現し、実行操作には使わない。
旧review schema version=1のauthorizationは従来どおりagreement digestと照合する。

`inputs` はrepo・親番号・タイトル・要件本文digest・needs-human・指示コメントのID/login/digest・contextを含む。
長い本文・議論を公開マーカーへ複製せず、元の判断入力は収集側が保存する。contextにも小さな参照/digestを渡す。
子タスクのチェックリスト、進捗概要、成果物・関連PR、次の改善候補の節と、Acceptance checklist / 全体の完了条件の完了チェックを除外する。
停止事項・対象外・未知の節は残す。現在のpublisherのparent-session:v1 / parent-review:v1記録は
進捗として分離する。新しい未分類コメント、本文/指示の編集・削除は鮮度不一致となり、次の予定回または明示再判断へ戻す。
判定コメントも人間合意と同じ厳密な形式検証で分離し、新しい根拠の `ignored_comments` には固定しない。
旧根拠の除外リストに判定表示がある場合は、現在のコメント全体を検証できたものだけ表示のdigest変更を許す。
それ以外の明示除外は従来どおり本文変更・削除を拒否する。

無関係な通常コメントは、内容を確認した呼出元だけが
`--ignored-comments` に `[{"id":123,"sha256":"comment_snapshotのdigest","reason":"要件と無関係な理由"}]`
を指定できる。`parent.execution.comment_snapshot` は `{id,login,body}` を返す。
同じ分類を記録specへ渡す。除外対象の本文変更・削除・重複IDは拒否する。
分類だけを更新してinputsとbasisが同一ならbasis digestは変わらず、既存のレビュー確認・許可を失効させない。
人間の指示を無関係として除外してはならない。新しい自然言語の意味の確認を、digest照合で代用しない。

後続スクリプトのPython API（import rootは `.agents/ops`）:

- 判断側: `parent.execution.execution_input(session, ignored_comments=(), context=None)` と
  `record_autonomous(session, spec)`。戻り値は `{parent,basis_sha256,generation}`。
  未担当なら環境session IDは不要。activeな担当があれば担当本人だけが更新でき、引継ぎ待ちは拒否する。
- dispatcher: **同じ `Store.locked()` 内で** `validate_candidate(session, basis_sha256, generation)` を呼び、
  返るauthorizationを保持して既存registryへ予約する。対象open・最新入力・最新根拠・担当不在・次generationを検査する。
  予約・実起動・runtime identityとの結合と復旧はdispatcherが実装する。別の担当台帳を作らない。
- 親workflow: 担当登録後に `session.agreed()`（既存名。有効な両種の根拠を受理）と
  `parent.execution.authorization(session)` で再検査する。委任・レビュー前・merge時にも再照合する。
  contextの外部資料・依存・採用状態の再取得と意味の検証は呼出元の責任で、contextの自己申告だけを鮮度証拠にしない。

自律根拠は記録時の次/現generationへ結び付く。担当交代・再登録後は明示的な再判断を要し、旧委任を再起動しない。
有効な根拠の変更は旧reviewを失効させる。新しい根拠の記録だけで旧runを再承認できず、新しい理由とrunが必要。
`basis-revoke --reason-file tmp/reason.md` と既存の `checking` は明示的な取消/保留として扱い、
自律根拠の再登録では解除できない。人間との実際の再合意を既存agree経路で記録する。
子の停止もbasis-recordでは解除しない。共有基盤の許可はreview側の別コマンドだけが扱う。
ops-report報告容器への根拠登録を拒否し、提案由来の子には従来の採用/受領/別途合意の検証を維持する。
採用済み提案の新たなIssue化経路はtriage側の後続実装が接続する。

## 採用提案の受領

日次triageは指定親のactiveな登録を確認し、親コメントに `event: proposal_pending` を記録する。
担当親は自分のセッションから次を実行する（triageの再実行を待たず同じ入口から受領することも可能）。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 receive-proposal --report 1300 --proposal R-20261019-01
# 会話でその提案の要件を認識合わせし、人間の合意を得た後
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --report 1300 --proposal R-20261019-01 --spec tmp/proposal-agreement.json
```

両イベントは `source: {proposal_id, report_number, option}`、`adoption: selected` と
`proposal_sha256` を持ち、セッションIDを公開しない。digestは提案の本文・選択肢の定義に結び付け、
変更可能な引渡し表示と関連親の指定を含めない。受領済みの再送は冪等。
受領は実装への同意ではなく、その提案だけを要件確認待ちにする。親全体のphase・合意と、
既存の独立作業・レビュー証拠は維持する。親自体が合意済みの状態で、上記の提案指定付き `agree` に
同じ5キー形式の合意specを渡す。`event: proposal_agreed` に受領時のsource・提案digestと合意内容を記録し、
その提案由来のcreate-child/delegateだけを許可する。他の受領提案や既存の親合意を流用しない。
statusの `proposal_receipts` / `proposal_agreements` で未合意と合意済みを照合でき、受領直後の中断も
GitHubコメントから復元する。明示的な `checking` は親全体と提案別の合意を取り消すため、
親の `agree` と各提案の合意を再記録するまで、それぞれの実装を拒否する。
既存提案の合意内容を変更する場合も `checking` で明示的に取り消してから記録する。
未取消の合意は同じ内容の再送だけを許可し、レビュー後に対象作業の要件だけを黙って差し替えない。
親未指定・未登録・停止中なら受領できない。人間の指定/再開後に親自身がregister/handoffして同じ入口へ戻る。
採用チェック・選択肢・提案本文が後から変われば起票/委任時も拒否し、担当親が採用と要件を再整理する。
受領・起票・委任時は、出典がopenで完了済みの `ops-report` issueであることも検査する。
triageの引渡し・受領・提案別合意は既存のマーカー付きイベントとして記録する。
担当親の自由文の連絡には [note](#自由文の進捗連絡) を使い、既存合意のコメント鮮度を保つ。
人間が独立に作ったissueには受領操作は不要で、同じregister/checking/agreeから開始する。

## 子の作成・委任・再開

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 create-child --spec tmp/child.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 reconcile
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 delegate --child 1217 --agent codex --branch codex/parent-session --worktree /path/to/main/.claude/worktrees/parent-session
```

子manifestは `{"key":"parent-session", "spec":{...}, "dependencies":[1225]}`。
`spec` は [issues renderの子spec](../issues/README.md#cliと整合検査) をそのまま使う。
親番号・識別名は実際の担当親と一致させる。`key` は同一作業で固定する小文字英数・ハイフンの1〜64文字。
`dependencies` 省略時は子本文の依存欄の `#番号` を先行依存として扱う。
明示した場合はその値で依存欄も生成する。依存先は同じ親の既存の直属子に限り、循環は拒否する。
既存の自由文に後続タスク番号が混じる場合は、`delegate --dependencies 番号...` で先行だけを明示する。
進捗判定は常に最新の子本文の依存欄を読む。明示CLI入力は委任時に本文へも反映し、古いローカル記録で上書きしない。

生成本文の末尾に `<!-- parent-task:v1 parent=N key=KEY spec=SHA256 -->` を付ける。
open/closed全issueをページングしてキーを照合し、spec不一致・複数一致を拒否する。
既存があれば再利用し、ラベル・正式なsub-issue・親一覧の不足だけを揃える。
作成応答の喪失や関係登録・親編集の途中失敗も、**同じキーとspec**で再実行する。
別の正式な親へ所属したissueを自動付替えしない。stateの消失だけを理由に新しいキーで作り直さない。
手動のレポート由来specは受領・その提案を含む要件合意・現在の採用を検証し、共通 `index_proposals` でも照合して、既存提案を別キーで二重起票しない。
triageの `ops-report-task:v1` 出典を持つ既存作業は [自動Issue化](../ops-report/README.md#採用提案の自動issue化) を参照する。
その子の委任では、有効な独立の `autonomous_decision`、現在の正式親、open/完了レポートの採用・選択肢・内容digestを
`adopted_tasks.py` で照合する。関連親の手動記入や架空のproposal_received/proposal_agreedは要求・生成しない。
これは出典の検証経路だけを追加し、担当・generation・レビュー・merge・共有基盤承認の条件を変えない。
開発workflowは `adopted_tasks.check_task_adoptions(gh, body)` を親と対象子の両方へ実行・merge直前に呼ぶ。
単一anchorを持たない共通親・分割した子にもcontext出典があるため、`parse_proposal_source()` だけで省略しない。
返却は出典の確認であり実行根拠ではない。既存のauthorizationとdispatcherの鮮度検査も必要。
同じ提案の既存子はキー・タイトル変更・親付替え後も再利用し、現在の正式な親を返す。自動で親を戻さない。
親関係のない既存提案が別キーなら番号を示して拒否し、孤立issueを明示的に移行・照合してから扱う。

`delegate` は実在する専用git worktree・branch・agent prefixを確認し、起動予約を先に保存する。
`action: launch` を受けた親がサブエージェントを起動する。CLI自身は起動しない。
既存PR・委任予約・完了・取消・停止・依存保留・予約上限・利用できる依存PRの不在があれば起動を拒否する。
`reconcile` は正式な子、全状態のPRの `Closes/Fixes/Resolves #子番号`（同repoのURLも可）、
記録済みbranchを照合して、PRと担当・検証記録を返す。
PRは子番号ごとのclosing指示を明記し、複数子を閉じる場合も各番号に指示を付ける。
ブランチ名だけから親子関係を推測しない。
既存PRや予約がある場合の `do_not_launch` は再開の調査対象であり、成功や完了を意味しない。
予約後の起動失敗・中断時は、既存workerが終了した証拠を確認し、必要なら停止・再開を記録して
`delegate --restart-evidence-file tmp/evidence.md` で明示的に再予約する。既存PRがあれば再予約もしない。

`record-result --child N --evidence-file tmp/tests.md` はclosing指示とbranchを照合し、
子の成果物欄へPRと検証結果を追記する。記録だけで子を完了扱いにせず、過去の検証記録も残す。
委任予約にはauthorizationと子の要件本文digest、結果記録にも現在のauthorizationと要件digestを持たせる。
根拠/担当/子の要件が変わった既存成果は、親が現在の要件で確認してrecord-resultを実行するまで新review/mergeへ渡さない。
要件確認中の結果記録は従来どおり可能だが、authorization=nullの事実記録であり実行の許可には使わない。

### 並列起動の選択と予約上限

`status` / `reconcile` / `sync` / 成功した `delegate` は `launch_plan` を返す。
`launch.py` が最新の子本文・正式関係・全状態のPR・既存予約を使い、次のJSONを計算する。

```json
{
  "policy": {"schema_version": 1, "max_parallel_children": null, "reason": "設定の根拠"},
  "active_reservations": [1217],
  "available_slots": null,
  "launchable_children": [1220],
  "tasks": [{
    "child": 1220, "state": "continue", "action": "ready", "dependencies": [1217],
    "stacked_on": [{"child": 1217, "pr": 1250, "branch": "codex/upstream", "head_sha": "40桁SHA"}],
    "reasons": []
  }]
}
```

`tasks` は全直属子を番号順で含み、`reasons` は `{code,message}` の配列。
起動対象は `action=ready`、対象外は `do_not_launch` と具体的理由を持つ。
既存PR・予約・完了/取消・停止・停止先への依存・依存PRの不在/複数/別repo・親のphase/担当・上限待ちを区別する。
未完了の推移的な依存にも同repoのPRが一意に必要で、閉じた未merge PRは利用しない。
停止した依存先のPRがあっても起動しない。完了済み依存はstackの対象から外す。
計画は予約・起動許可そのものではない。実行根拠の鮮度、提案の採用、専用worktree、依存headは
`delegate` が書き込み直前に検査する。`status` の読み取りだけでは予約もagent起動も行わない。
`sync` は親本文の進捗概要に予約一覧・起動可能・待機理由を反映する。

手動親の既定は `max_parallel_children=null`（依存が許す全子）。上限があれば空き数まで番号順に選ぶ。
上限と根拠は親のregistry行の `launch_policy` に保存し、同じ `Store.locked()` で委任予約と直列化する。
設定はworktree・セッションの再開をまたいで維持する。未知schema・不正型は既定値に置換しない。

```bash
# 実測で上限が必要なら、根拠を非空のファイルに記録して設定する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 configure-delegation --max-parallel-children 4 --reason-file tmp/resources.md
# 制限を解除する場合も理由を残す
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 configure-delegation --max-parallel-children all --reason-file tmp/resources.md
# runnerの終了・子孫の不在を確認した親が、delegateで返ったdelegation_idを指定する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 record-worker-ended --child 1217 --delegation-id 32桁hex --evidence-file tmp/worker-ended.md
```

上の4件は設定方法の例で、推奨初期値ではない。固定3件の根拠だった事故は、ホストRAM上限24GB・GPU 16GBの環境で
2026-10-18にpytestとtraining jobが重なってPCが落ちた事例。子の数の実測上限ではないため、
pytestは下記lock、GPU・Blenderはtraining queueの既存 `half` / `all` 枠で調停し、子の上限は追加の実測に基づき設定する。
自律ループの「各親の実装子2件」という初期値と実プロセスの稼働枠・担当排他はdispatcherの責務で、ここでは変更しない。
dispatcher/親workflowはその実行枠内でこの計画を使い、必要なら同じ予約上限を明示設定する。

終了未確認の委任予約は停止・PR作成・子のclose・親の待機だけで解放しない。
`record-worker-ended` は親が確認した終了証拠を現在の `delegation_id` へ結び付けるだけで、プロセスを停止しない。
旧予約でIDがないものには明示値 `legacy` を使う。再委任は新しいIDになり、古い終了通知は拒否する。
予約枠を解放しても `delegated` は保持し、自動再起動はしない。再予約は従来の証拠付き `delegate --restart-evidence-file` を使う。
停止・根拠・PR検査は再予約でも省略しない。稼働中workerの終了を推測する新しい台帳は作らない。

### stacked branchの確認とrebase

`stacked_on` の各PRはclosing指示で依存子に対応付け、branch名だけから推測しない。
親が全headのコード・インターフェースを確認し、同じrepoに取得してから専用worktreeを作る。
以下は依存先が1つの場合。`upstream_head` は計画の完全なhead SHAで固定し、ローカルに保存する。

```bash
git fetch origin codex/upstream
upstream_head="$(git rev-parse origin/codex/upstream)"
# 計画のheadと一致しなければ再照合する
git worktree add -b codex/downstream .claude/worktrees/downstream "$upstream_head"
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 delegate --child 1220 --agent codex --branch codex/downstream --worktree .claude/worktrees/downstream
```

作成先は必ずメインcheckoutの `.claude/worktrees/`。親自身がworktree内なら上の相対パスをその絶対パスへ置き換える。
複数依存なら全headを含むbranchへ積むか、子専用の統合branchへ取り込む。
`delegate` は全未完了依存headが子のHEADの祖先であることと、保存直前にもhead/依存が同じことを確認する。
予約の `stacked_on` が起動時の境界。先行PRの更新を取り込んだ場合は、実際に積み直した境界もローカルへ記録する。
後続PRのbaseは先行branchにして子だけの差分を表示し、merge順序は依存順を維持する。

先行PRのmerge後は、子worktreeがcleanであることを確認し、保存した境界から子自身のcommitだけをmainへ載せ替える。
通常merge・squash mergeのどちらでも、子の差分から先行変更を除く。

```bash
git fetch origin main
git -C .claude/worktrees/downstream rebase --onto origin/main "$upstream_head" codex/downstream
git -C .claude/worktrees/downstream diff origin/main...HEAD
```

全先行PRのmergeを確認してから子のPR baseをmainへ変更し、差分・テスト・依存を再確認する。
独立した複数依存の統合branchでは、全先行headを取り込んだ統合commitを境界に保存して同じ `--onto` を使う。
競合は親が解消・検証し、判断が必要ならその子を停止する。head/base/diff変更は既存review coverageで承認済みにせず、
rebase後にレビュー対象を固定する。既にレビューした対象の変更には理由付きの新しい明示要求が必要で、自動再レビューしない。

### pytest共有lock

```bash
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_launch.py -q
# キャンペーンが専用の共有lockを指定している場合は、全子に同じ絶対パスを渡す
PYTEST_LOCK=/absolute/shared/pytest.lock bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_launch.py -q
```

ラッパーは自身のworktree rootへ移動し、その `.venv/bin/python -m pytest -n 4` を実行する。
lockは `PYTEST_LOCK`、未指定なら `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/pytest.lock`。
全worktreeで同じホストstateを使い、**1つのpytestだけ**をflockで通す。worktreeごとのlockは作らない。
`-n` / `--numprocesses` の上書きと相対lockを拒否し、pytestの終了値をそのまま返す。
`flock --no-fork` でlockを実行プロセスに保持し、正常終了・異常終了後に解放する。
lockを取らない直接実行は排他の対象にならないため、親・全子・通常の手動テストもこの入口を使う。
テストではfake Pythonを使って2つのworktreeから同時要求し、実行区間が重ならないことと失敗後の解放を検査する。

## 停止・再開・改善候補

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 stop --child 1217 --spec tmp/stop.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 resume --child 1217 --spec tmp/resume.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement --child 1217 --spec tmp/improvement.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 sync
```

停止・再開specは `reason` / `facts` / `impact` / `resume_condition` / `next_action` の5つの非空文章。
再開specには条件が満たされた証拠を記載する。詳細を子へ残してから親の進捗・短い通知を更新する。
子の停止節末尾に `<!-- parent-task-status:v1 state=stopped|resumed detail=SHA256 -->` を記録し、
途中失敗後もその公開状態から停止判定を復元する。本文編集中に取得内容が変わった場合は上書きせず拒否する。
通知は子へのリンク、理由の要約、依存保留と継続の一覧、次の対応を持ち、事実の詳細は転記しない。
停止・改善候補・検証結果の記録は担当activeなら要件確認中でも行える。記録から実装は起動しない。
進捗は停止→依存先へ保留を伝播し、独立した子は継続する。再開で保留を再計算する。
`sync` はGitHubの実状態・完了理由から一覧を更新するだけで、agentを起動しない。
Project表示は明示的な `project-setup` / `project-sync` を使う。既存コマンドからは呼び出さない。
項目・ビュー・節目spec・担当・読み取り専用PMとの境界は [Project表示](../project_progress/README.md) を参照する。
`closed/completed` だけを完了チェックにし、取消は未チェックで残す。
親全体の完了はチェック数から推測せず、全体の受入条件を親agentが確認する。

改善specは `purpose` / `evidence` / `scope` の3つの非空文章。
子の改善候補欄だけを更新し、Acceptance・実装範囲・担当状態は変えない。

## 担当の終了・引継ぎ

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --note-file tmp/handoff.md
# 指定された次の親セッション自身が実行する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --accept --note-file tmp/takeover.md
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 release --ended --note-file tmp/completion.md
```

`handoff` で引継ぎ待ち、`release` で解除、`release --ended` で終了済み。
引継ぎ待ちは旧担当も分割・委任できず、新担当のacceptまで代替起動しない。
acceptは通常、解除/終了/引継ぎ待ちだけを受け取り、同時担当数を再検査してissue/PRを照合する。
中断した同一セッションは `reconcile`、別セッションは明示acceptから始める。
旧セッションが未解除のまま失われた場合は、ローカルstateと既存workerを確認し、
人間の担当指定に基づき `handoff --accept --recover --note-file tmp/recovery.md` を明示実行する。
ノートに旧workerの停止・既存worktree/PRを確認した証拠を残す。通常のregisterはactive担当を上書きしない。
recoverも既存の委任予約を保持し、照合結果から自動再起動しない。stateを消して空き担当と扱わない。
引継ぎノートには完了/未完了の子・PR、停止条件、次の作業、レビューrunを記載し、ローカルIDを書かない。

## レビュー要求契約（schema version 2、version 1読取互換）

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-prepare --prs 1250 1251 1252 > tmp/targets.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-request --run-id parent-1216-UUID --instruction-file tmp/instruction.md > tmp/review-request.json
```

`review-prepare` は全直属子にclosing指示がある全open PRを対象集合と照合する。
PR未取得・欠落・重複・closed・draft・fork・親と無関係をエラーにし、部分的な要求を公開しない。
未実装の子にPRがまだ無いことは許容する。段階ごとに、その時点の全open子PRを束ねる。
分割していない単独issueではそのissueを閉じるPRを同じ方法で扱う。
GitHub取得・git差分取得後にPR集合とhead/baseを再取得し、途中更新を拒否する。
不足オブジェクトだけ `origin` からSHAをfetchする。

prepareのJSON（追加キーなし）:

```json
{
  "schema_version": 2,
  "authorization": {
    "generation": 1,
    "execution_basis_kind": "autonomous_decision",
    "execution_basis_sha256": "basis全体の64桁hex SHA-256"
  },
  "requirements": {"1216": "親の要件本文の64桁hex digest", "1217": "対象子の要件本文の64桁hex digest"},
  "review_run_id": "parent-1216-32桁のhex UUID",
  "repo": "Motoki0705/tennis-lab",
  "parent_issue": 1216,
  "reason": "親が対象PR群を確定した理由",
  "prs": [{
    "number": 1250,
    "head_sha": "40桁のhex SHA",
    "base_sha": "GitHubで取得したbase先端の40桁SHA",
    "base_ref": "main",
    "merge_base_sha": "baseとheadのmerge-baseの40桁SHA",
    "diff_sha256": "64桁hex SHA-256",
    "branch": "codex/issue-format"
  }]
}
```

`prs` は番号昇順。diffは以下の**生バイト列**のSHA-256（文字列への変換・切詰めなし）。

```bash
git diff --no-ext-diff --no-textconv --binary --full-index --no-renames --no-color MERGE_BASE HEAD --
```

親コメントの `event:review, state:targets_fixed` にrun ID・requestのcanonical JSON SHA-256を記録し、
本文のJSONコードブロックにmanifest全体を載せる。この時点ではレビュー未指示。
`review-request` は同じ対象が現在も有効なことを再検査して、親の明示指示をコメントへ記録する。
出力は上記manifestに `"review_state":"instructed"` を追加したJSON。
reviewイベントにもmanifestと同じauthorizationを記録する。
requirementsは親と対象PRが閉じる子の要件本文digest。進捗表示を除く同じ射影を使い、
子の要件変更をrecord-resultだけで旧レビューのcoverageへ持ち込まない。
prepare中の再取得・指示・収集時にclosing対象と固定した要件集合を照合し、HEADが同じでも
対象子の追加・差替え・欠落は新しい理由/runで固定し直す。結果の全 `child_issues` にdigestを必須とし、
欠落する旧結果は結果受理・修正確認・merge gateでも拒否する（version=1の読取契約は維持）。
起動/指示時は全対象、修正確認は当該PRの子、mergeはPRごとに照合する。親全体の要件/根拠変更は全件を止める。
結果受領などのrun進捗だけでは子のcoverageを承認せず、子の停止・変更が独立PRを一律に止めないようgateで判定する。
旧version=1は従来のgeneration/agreement_sha256を使い、human_agreement経路だけで照合する。
保存済みmanifestを変換してdigestを書き換えず、旧runへの指示の付け替えも拒否する。
`request_sha256` は `review_state` 追加前のprepare manifestのcanonical JSONに対して計算する。
**#1218はこのJSONだけで起動せず、親コメントの同じrun ID/request digest/指示済み記録と最新対象を照合し、
run IDごとの実行を一度だけ受理する責任を持つ。** prepareだけのJSONは実行指示にしない。
受理・実行・修正確認・結果スキーマは [親レビュー入口](../review/README.md) を参照する。
同じprepareの再実行は同じrun ID、同じrequestの再送は同じ指示を使う。
次のrunや対象変更には `review-prepare --reason-file` で新しい理由を必須とする。
PR open/push/CI完了ではこの入口もレビューも起動しない。

未指示 → `targets_fixed`（対象確定）→ `instructed`（指示済み）→ `result_received`（結果受領）→
`fixes_verified`（修正確認）→ `complete`（完了）の順。
後半3段階は `review-state --run-id ID --state 状態 --evidence-file tmp/evidence.md` で証拠を残す。
結果受領は成功判定ではなく、失敗・阻害指摘も受領して記録する。親が指摘と修正を照合し、
各PRのCIなどを確認した証拠が揃うまで完了へ進めない。CLIは#1219のmerge gateの代用ではない。

## 将来の割当側と実行条件

自動割当側は上記の候補検査と既存registryの排他を使い、対象親の実行根拠とruntime identityを結び付ける。
同じrepo・親番号/URL・環境由来ID・state・競合エラーを使い、子作成・実装・レビューの検査を飛ばさない。
このCLIから判定agent・実装agent・timerは起動しない。

CLIはJSONをstdout、エラーをstderrへ出し、成功0・設定/取得/整合エラー2で終了する。
ghは現在認証中のアカウント、GitHub書込みにはIssuesのwrite権限が必要。
親子ラベルの作成やsystemd操作は行わない。GitHubとローカルstateは単一トランザクションではないため、
非ゼロ終了後は同じ入力で照合・不足更新を行う。未完了を成功や新規起票に置換しない。
atomic write中にプロセスを強制終了して `.tmp` が残った場合は、稼働中プロセスが無いことと
registryの内容を確認して、その一時ファイルだけを人間の管理下で退避してから再試行する。

テスト: `bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_session.py tests/e2e/agents_ops/test_parent_launch.py tests/e2e/agents_ops/test_pytest_lock.py -q`。
fake gh・一時state・使い捨てgitを使い、本物のCodex/Claude/GitHub書込みは起動しない。
