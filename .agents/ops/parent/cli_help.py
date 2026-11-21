"""Task index and on-demand input examples for the parent CLI."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Mapping

# The parser supplies the option syntax. This index describes the task and
# routes the parent to the matching part of the skill when it needs context.
TOPICS = {
    "ownership": (
        "担当と合意",
        "status register checking agree wait-supplement summary handoff receive-proposal basis-input basis-record basis-check basis-revoke agreement-ignore-comments",
    ),
    "workers": (
        "委任と統合",
        "create-child delegate configure-delegation run-worker wait-worker inspect-worker decide-worker-plan record-result stop resume",
    ),
    "dispatch": (
        "自律親",
        "start-worker continue-worker workflow-report record-cancellation",
    ),
    "review": (
        "正式レビューとマージ",
        "review-request review-inspect merge-status wait-merge merge-continue merge-confirm",
    ),
    "finish": (
        "記録と終了",
        "note notify sync decide release improvement improvement-propose improvement-decide improvement-extract improvement-expire project-setup project-sync",
    ),
    "recovery": (
        "照合と復旧",
        "reconcile stop-worker refresh-worker-plan record-worker-run record-worker-ended evidence worker-log-status publish-worker-log decision-log-status retry-decision-capture publish-decision-log state-migrate",
    ),
}

# Example suffixes follow the common prefix printed by the top-level help.
EXAMPLES = {
    "register": "register --summary 推論時間の短縮\n表示要約は全角12字以内。担当登録後は agree で実行根拠を記録する。",
    "handoff": "handoff --accept --summary 推論時間の検証 --note-file handoff.md\n渡す親は --note-file、受ける親が --accept を使う。",
    "agree": 'agree --spec agreement.json\n{"purpose":"モデルの再読込を減らす","scope":"推論入口","behavior":"共有モデルを再利用する","acceptance":"比較入力の出力が一致する","human_confirmation":"確認した人間の元発言と出典"}',
    "create-child": "create-child --spec child.json\nJSONは key、spec、任意の dependencies（先行子番号の配列）。specは .agents/ops/issues/README.md の「CLIと整合検査」にある子Issue入力。再送時は同じkeyとspecを使う。",
    "delegate": "delegate --child 1217 --agent codex --branch codex/impl --worktree .claude/worktrees/impl\n最初が統合担当。補助担当は別のbranch・worktreeと --worker quality を指定。予約後は run-worker。",
    "run-worker": "run-worker --child 1217 --worker quality --from BASE_SHA --prompt-file compare.md\n--fromは専用branch・worktree・共有入力・venvを準備して予約し、そのまま起動する。既存予約は--fromを省く。終了後に同じ担当へ追加検証を渡すには --resume --prompt-file next.md。調査結果だけを受け取る場合は --planning。\nagy手動親ではsetsid/&を使わずrun_commandの非同期タスクで直接起動し、inspect-workerで稼働確認後にwait-workerを接続する。",
    "start-worker": "start-worker --child 1217 --worker quality --from BASE_SHA --prompt-file compare.md\n自律親用。返されたjob_idで予約を識別し、inspect-workerで終了と成果物を照合する。",
    "continue-worker": "continue-worker --child 1217 --worker primary --prompt-file fix.md\n自律親用。同じbranch・sessionの終了を照合してから修正を渡す。",
    "inspect-worker": "inspect-worker --child 1217 --worker quality\n実行状態とfinal_fileを確認する。",
    "wait-worker": "wait-worker --child 1217 --worker quality --timeout 900\n終了値0は終了・回収済み。手動委任は監視・自動再開の終了まで待つ。runner_summary、final_file、followupで結果を確認する。124は時間満了で再待機可。2は理由を読み、inspect-workerで元runの状態を照合する。agy手動親もsetsid/&を付けずrun_commandの非同期タスクで直接実行する。",
    "configure-delegation": "configure-delegation --max-parallel-children 2 --reason-file capacity.md\n枠は担当ごとに数える。親全体の上限を数値またはallで指定し、変更理由を書く。",
    "decide-worker-plan": "decide-worker-plan --child 1217 --worker quality --action investigate --answer-file next.md\n直近の調査報告に対する判断を書く。approve / investigate / reject と具体的な回答を保存し、次の同sessionの継続時に渡す。",
    "record-worker-ended": "record-worker-ended --child 1217 --worker quality --delegation-id DELEGATION_ID --evidence-file ended.md\nstatusで対象IDを取得し、runnerと子孫の終了・回収を照合した根拠を書く。",
    "record-result": "record-result --child 1217 --evidence-file verified.md\n統合HEAD、受入条件、対応するテスト結果を書く。",
    "review-request": "review-request --prs 1250 --instruction 品質比較を確認 --evidence-file - --wait\n検証結果をstdinで渡す。保存済みnoteは --evidence-ref SHA（またはURL）。成果の記録もこの操作が行う。同じPRの再送は既存runを返す。",
    "merge-continue": "merge-continue --run-id RUN --instruction-file fixes.md --wait\nmajorごとの修正commit・検証結果・親の判断を記す。同じrunで現HEADのマージ条件へ進む。",
    "merge-status": "merge-status --run-id RUN\nPRの段階・停止理由・review_usageを確認する。RUN省略時はこの親の最新run。",
    "wait-merge": "wait-merge --run-id RUN --after 1\n最後に受け取った通知ID以降のイベントを待つ。",
    "merge-confirm": "merge-confirm --prs 1250 --wait\n切替前にマージ済みのPRのmerge commit・Issue終了・main CIを後確認する。",
    "notify": 'notify --spec notification.json\n共通項目は schema_version=1, key, kind, summary, evidence（証拠コメントURL配列）, resolves（質問key配列）, details。\nkind別のdetails:\n  question: question, resume_condition\n  agreement: change, impact, answers（source, quoteの配列）\n  impact: result, decision, next_action, affected, achievements, remaining\n  completion: achievements, artifacts（https URL配列）, remaining。\n完了通知の例（証拠は同じ親の機械記録のURL、成果物はその実URLに置換）\n{"schema_version":1,"key":"completion","kind":"completion","summary":"合意範囲を検証した","evidence":["https://github.com/OWNER/REPO/issues/1234#issuecomment-5678"],"resolves":[],"details":{"achievements":["品質比較を完了"],"artifacts":["https://github.com/OWNER/REPO/pull/1250"],"remaining":[]}}',
    "stop": "stop --child 1217 --spec stop.json\nJSONの5項目は reason, facts, impact, resume_condition, next_action（すべて文章）。Issueと依存する作業を保留する。",
    "resume": "resume --child 1217 --spec resume.json\nreason, facts, impact, resume_condition, next_action に再開条件が満たされた根拠と次の操作を記す。",
    "stop-worker": "stop-worker --child 1217 --worker quality --spec stop.json\nspecの5項目は stop --help を参照。複数担当では指定した担当だけを止める。自律親の単独担当ではIssueと依存jobにも停止が及ぶ。",
    "release": 'release --ended --spec finish.json\nversion 2は通知・公開確認・終了まで同じ入力で行う。evidenceはこの親の機械記録のコメントURL。マージ済みPRのURLは自動で含む。追加の成果物はartifacts、解決した質問keyはresolvesで渡す。basis_refと条件IDはstatus.completion.requirementsの値に置換。\n{"schema_version":2,"state":"completed","evidence":["https://github.com/OWNER/REPO/issues/1234#issuecomment-5678"],"basis_ref":"実際の参照","acceptance":[{"id":"agreement","state":"met","evidence":"合意した動作の検証結果"},{"id":"ac-1","state":"met","evidence":"本文の条件1に対応する測定結果"}],"remaining_required":[],"improvements":{"reviewed":true,"candidates":[]},"summary":"受入結果"}',
    "workflow-report": 'workflow-report --spec report.json\n{"schema_version":1,"state":"waiting","reason":"CI待ち","resume_condition":"CI終了","evidence":"対象PRとHEAD"}\n完了時はrelease --helpと同じversion 2の受入JSONをそのまま渡す。',
    "decide": "decide --spec decision.json\ninputsに観測結果ID、evaluationに判断、reasonに根拠。初回判断はbasisに要求・合意の固定参照が必要。後続操作に --decision-id を渡す。計画の書式は .agents/ops/decision_logs/README.ja.md の「親の判断を明示記録する」。",
    "project-sync": 'project-sync --event start --spec project.json --dry-run\n初期登録済みの子の例。プレビューの確認後は同じevent・specで --apply。\n{"schema_version":1,"items":[{"issue":1217,"stage":"作業中","wait_reason":"","next_action":"実装と品質比較を開始する"}]}',
    "record-cancellation": "record-cancellation --child 1217 --spec cancel.json\nJSONは schema_version=1, reason, handoff, evidence。",
    "improvement": "improvement --purpose 結果の発見を早める --evidence 検証結果の探索に毎回3操作 --scope 結果の表示\n返されたkeyで提案時に詳細化する（improvement --key KEY --spec idea.json）。詳細specはschema_version=1。purpose, observation, evidence, judgment, scope, out_of_scope, behavior, stop_conditions, unknowns, expected_effect は文章。acceptanceとunmet_acceptanceは配列。",
    "improvement-decide": 'improvement-decide --candidate I-1216-result-access --action approve --spec decision.json\n{"schema_version":1,"request_key":"decision-1","revision":1,"previous_decision":null,"human_confirmation":{"quote":"元発言","source":"出典"},"reason":"判断の根拠"}\n複数候補ではrevisionとprevious_decisionを候補ID別のcandidatesオブジェクトにまとめる。',
    "improvement-extract": "improvement-extract --candidates I-1216-result-access\n人間が取り出しを指示した承認済み候補を、新しい親Issueにまとめる。",
}


DESCRIPTIONS = {
    "status": "現在の担当・実行根拠・launch_plan・受入条件・未公開ログを読み取る。",
    "checking": "要件を確認中と記録する。既存の合意は維持する。",
    "wait-supplement": "対象Issueの補足workflowを待ち、確定状態を読み取る。",
    "summary": "担当表示の短い要約を更新する。",
    "receive-proposal": "採用レポートの提案を現在の親へ結び付ける。",
    "basis-input": "自律着手の判断に使うIssue、依存、合意の入力を取得する。",
    "basis-record": "dispatcherの自律着手判断と入力を保存する。",
    "basis-check": "保存した根拠と現在の入力を予約直前に照合する。",
    "basis-revoke": "親全体の実行根拠の取消しを記録する。",
    "agreement-ignore-comments": "確認済みのコメント分類を明示記録する。",
    "configure-delegation": "この親の並列予約枠を変更する。補助担当も各1枠を使う。",
    "decide-worker-plan": "調査報告への回答を保存する。approve / investigate / reject を選ぶ。",
    "review-inspect": "対象PRのレビュー時と現在のHEAD、指摘、修正履歴を読み取る。",
    "note": "自由文の判断根拠を証拠用Issueへ保存する。",
    "sync": "正式な子・PR・停止理由を親Issueの進捗概要に反映する。",
    "decide": "観測結果に親の判断と次の操作を結び付け、判断IDを返す。",
    "improvement-propose": "この親の記録済み候補を一つの改善提案Issueに掲載する。",
    "improvement-expire": "期限が過ぎた未取り出し提案を整理する。--dry-run で対象を確認する。",
    "project-setup": "Projectの構成変更をプレビューする。--apply で適用する。",
    "project-sync": "指定した節目のProject変更をプレビューする。--apply で適用する。",
    "reconcile": "既存PR・委任予約・依存を照合し、現在の作業対象を返す。",
    "refresh-worker-plan": "引継ぎ後の担当と保存済みワーカー計画を照合する。",
    "record-worker-run": "既存の共通runner記録を、現在の手動委任へ登録する。",
    "record-worker-ended": "確認済みの終了証拠を委任IDへ結び付けて枠を解放する。",
    "evidence": "この親の機械証拠の保存先を照合する。--create で不足時に作成する。",
    "worker-log-status": "ワーカーのターン単位の公開状況を読み取る。",
    "publish-worker-log": "保存済みの委任・ターンから公開を続ける。--dry-run で変更前に確認する。",
    "decision-log-status": "親の判断ログの捕捉・公開状況を確認する。",
    "retry-decision-capture": "捕捉に失敗した既存記録を再取得する。",
    "publish-decision-log": "判断ログの公開を --dry-run で確認し、同じ対象を公開する。",
    "state-migrate": "旧registryを退避し、担当・委任・レビュー履歴を保持して移行する。",
}


def configure(
    root: argparse.ArgumentParser, commands: Mapping[str, argparse.ArgumentParser]
) -> None:
    lines = [
        "実行例  PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 status",
        "操作の詳細  PYTHONPATH=.agents/ops .venv/bin/python -m parent delegate --help",
        "--spec / --*-file の入力は1つだけ - で標準入力から読める。",
        "日次jobのimprovement-expireはcheckout_job.py経由で実行する（helpを含む）。.agents/ops/parent/README.md「改善提案の期限」参照。",
        "",
    ]
    for reference, (title, names) in TOPICS.items():
        lines.append(
            textwrap.fill(
                names, width=86, initial_indent=f"{title}  ", subsequent_indent="    "
            )
        )
        for name in names.split():
            command = commands[name]
            command.formatter_class = argparse.RawDescriptionHelpFormatter
            command.description = DESCRIPTIONS.get(name, title)
            path = f".agents/skills/parent-session/references/{reference}.md"
            example = EXAMPLES.get(name)
            command.epilog = (
                f"使用例（--parent などの共通引数の後）\n  {example}\n\n"
                if example
                else ""
            ) + f"手順  {path}"
            if any(
                a.dest == "spec" or a.dest.endswith("_file") for a in command._actions
            ):
                command.epilog += "\n--spec / --*-file は1つだけ - で標準入力にできる。"
    root.epilog = "\n".join(lines)
