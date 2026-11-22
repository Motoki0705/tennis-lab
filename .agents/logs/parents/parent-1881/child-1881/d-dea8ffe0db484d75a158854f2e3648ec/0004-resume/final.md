[PR #1927](https://github.com/Motoki0705/tennis-lab/pull/1927) を同じbranchで修正しました。

- branch: `codex/parent-comment-curation`
- HEAD: `0e4cc010a05d9dcf0f831f730795e780c95fa472`
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43274382822): **Python・Lint・CPU 10 shardを含む13/13成功**。Python checkのSHA一致も確認済みです。

各指摘の修正と回帰nodeは以下です。nodeはすべて `tests/e2e/agents_ops/` 配下です。

| 指摘 | 修正 | 回帰node |
|---|---|---|
| pr-1927-1 | 保存snapshotの`history_owner`で証拠URLを所属Issueへ解決。初回・継続予約を検証 | `test_dispatcher_supervisor.py::test_evidence_comment_citation_survives_runtime_reservations`（`initial`、`continuation`、`changed-source`） |
| pr-1927-2 | improvementのreason_refに実保存先URL helperを使用。実在コメントと本文fingerprintまで照合 | `test_parent_improvements.py::test_candidate_reason_and_approval_are_linked_without_writing_ended_source_journal` |
| pr-1927-3 | 共通completion判定で現在の受入を表示。過去の成果・通知上の残件と区別 | `test_improvement_recovery.py::test_reapproved_requirements_do_not_inherit_previous_acceptance` |

同種の修正として、finishのない親の再合意・未合意な条件変更、後続のincomplete記録でも、通知の「残件なし」を現在へ流用しないようにしました。

`rg`で全ops・skill・各言語READMEのURL生成/解析、reader、完了状態の選択処理を検索し、`git grep`でtracked全体も照合しました。`collect/judgement/schema`、`parent_sources`、`history/record_writer/comment_journal`、review reader、通常コメント専用URL、completion/改善の適用判定、worker_status/project_progress、Issue本体URL、既存出典は変更不要と確認。検索方法と理由付き一覧はPR本文に記載しました。

検証結果：

- 関連3,413件：初回 **3,392 passed／21 setup error／1,582.22秒**。
- 21件は今回のsnapshot前提へのfixture追随漏れ。origin/mainでは同じ21件が成功（279.64秒）。保存入力を用意するfixtureへ修正後、**21 passed／178.40秒**。
- 最終表示差分：**182 passed／428.09秒**。
- 増分7ケース：**7 passed／20.19秒**。既存2nodeを強化。setup最大0.08秒、case最大5.025秒。CLI→自動sync→親本文の連結を守る理由をPRに記載。
- pre-commitのruff/mypy、opsのruffは成功。
- push別CI：`37ce4aa2f`は上記fixtureで2ジョブ失敗、`356fb9a30`と最終`0e4cc010a`は各13/13成功。

変更した全13パス：

```text
.agents/ops/decision_logs/README.ja.md
.agents/ops/decision_logs/README.md
.agents/ops/decision_logs/README.zh-CN.md
.agents/ops/decision_logs/parent_commands.py
.agents/ops/dispatcher/README.md
.agents/ops/dispatcher/supervisor.py
.agents/ops/parent/README.md
.agents/ops/parent/notifications.py
tests/e2e/agents_ops/test_dispatcher_supervisor.py
tests/e2e/agents_ops/test_improvement_recovery.py
tests/e2e/agents_ops/test_parent_completion_resume.py
tests/e2e/agents_ops/test_parent_improvements.py
tests/e2e/agents_ops/test_parent_notifications.py
```

PRはopen・未mergeです。設計上の判断待ちはありません。実Issueへの移行・merge後確認は未実施で、親担当のままです。