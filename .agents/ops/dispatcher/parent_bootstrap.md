# 自律割当された親の入口（bootstrap protocol v1）

末尾の `bootstrap_input` のJSONを全文読む。これはローカルな予約・候補・実行根拠とruntimeの対応であり、
人間の合意や共有基盤のmerge許可ではない。session IDやローカルパスを公開しない。

1. `.agents/ops` をPython import rootとして、このcheckoutの `.venv/bin/python -m dispatcher`
   に `--parent-state-dir`、`--state-dir`（入力のdispatcher_state_dir）を指定し、
   `accept --input <bootstrap_inputのパス>` を実行する。実CLIが設定したsession環境を使う。
   入力の親番号・候補・authorization・予約・runtimeが一致しない場合は作業を止める。
2. 同じstate引数でdispatcherの `status` を実行し、GitHubの親・正式な子・既存PRを読み取り、担当・根拠と照合する。
   parent CLIには `--state-dir <parent_state_dir>` を指定し、同じregistryを使う。
   内容に不一致・停止・不足情報があれば具体的な理由と再開条件を記録して待機する。
3. [親の開発workflow](../parent/development_workflow.md) と
   [parent-session skill](../../skills/parent-session/SKILL.md) を読み、実装・検証・レビュー・mergeを進める。
   自律根拠を人間合意として再登録しない。候補の目的・範囲・動作・受入条件、正式な親子・依存・採用状態を
   最新のissue/コメント/PR/worktreeと照合し、具体化できる通常の実装判断は親が行う。
   単独issue自身が既にmergeでclosed/completedなら、開発workflowのclose後の終了照合手順へ戻る。
   この復帰では新しい実装・分割・worker起動/継続を行わず、全体受入と最終報告を終える。
   必要な場合だけ分割する。単独issueは自身の番号をworkerの担当とし、不要な子issueを作らない。
   parent CLIには `--state-dir <parent_state_dir>`、worker操作にはさらに
   `--dispatcher-state-dir <dispatcher_state_dir>` を渡し、review/mergeも同じprivate stateを使う。
   `start-worker` はparentのdelegate予約後にdispatcherのstart-childを呼び、supervisorが共通ランナーで起動する。
   指定省略のSol/max、必要時のAstra/maxを使い、既存workerの修正は終了照合後に `continue-worker` で同じsessionへ戻す。
   差分・受入条件・テスト・PRを親が確認してrecord-resultを残す。停止した作業と依存作業だけを保留し、独立作業を進める。
   親がPR群を固定して既存の親単位レビューを明示要求し、指摘修正・通常検証とmerge gateを通す。
   自動再レビュー・別session・無制限の修正で停止を回避しない。
   最後に `workflow-report --spec <JSONファイル>` で理由・再開条件・検証証拠をnoteへ残し、その出力を最終文にする。
   completedは実merge/close・gateの試行記録・現在の根拠/要件・親の確認と子の終了を再検査した場合だけ返る。
   CI待ち・情報不足・権限不足はwaiting、実行失敗はfailedにして、証拠と再開条件をissueへ残す。
   担当終了は最終イベントを確認したsupervisorが行うため、生存中の親から `release --ended` しない。
   自由文の進捗・連絡は `note --body-file` を使い、合意後のコメント検査を素のコメントで止めない。

子の選択はparentの `launch_plan` に従い、起動可能な子を実行枠内で並列に進める。
待機理由とstacked_onはparentの契約を使い、別の起動計画を作らない。
pytestは共有lockなしで親・子間の並列実行を許可する。実行方法はparent READMEの「pytestの並列実行」に従う。
予約上限とpytest排他の正本は `.agents/ops/parent/README.md` を参照する。
最終文は追加キーのない `schema_version/state/reason/resume_condition` のJSON objectにする。
stateはwaiting/completed/failed。completedはworkflow自身が受入条件・実merge/closeを確認した場合だけ使う。
runnerのexit 0はタスク成功を示さない。子が生存/終了未確認ならcompletedは拒否される。
