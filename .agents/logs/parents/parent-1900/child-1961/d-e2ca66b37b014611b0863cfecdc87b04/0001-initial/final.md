調査のみ完了しました。コード・state・journalは変更せず、commit・PR・公開操作も行っていません。branchは `codex/log-publication-completion`、調査対象HEADは `b734cb96`、worktreeはcleanです。

実装方針は組めます。ただし、このホストでは**openログPR 23本のうち13本の元journalが見つからず**、全件回収には追加の所在確認が必要です。

1. `mergeable=null` の停止は、安全検査を維持したまま除去できます。

   対象は [publication.py:480](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/worker_logs/publication.py#L480) と494行です。両方ともnullを許容して後続へ進め、既存の `GitHub.merge(number, head)` を呼びます。このメソッドはすでに `{"merge_method":"merge","sha":head}` を送っています。`mergeable=false` の明示拒否は維持できます。

   公式仕様では、nullはバックグラウンド計算中を示します。一方、merge APIに `mergeable=true` の事前取得条件はありません。ただし、**nullの間に要求すれば必ず成功するという保証もありません**。APIの結果で判断する設計になります。[GitHub公式仕様](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)

   | 応答 | 記録する扱い |
   |---|---|
   | 200・`merged=true` | 既存のPR再取得・main照合へ進む。照合完了後だけ `merged` |
   | 405 | `blocked / error=merge_conflict` |
   | 409 | SHA不一致として拒否。再取得でもHEAD不一致なら `pr_identity_changed` |
   | 403・404・422 | 拒否をエラーとして保存し、未公開のまま |
   | 接続断・応答不明 | `pending / merge_response_uncertain` として照合待ち |

   405はmerge不能、409は指定SHAとPR HEADの不一致です。現在の `GitHub.request()` はHTTP失敗をすべて `github_failed` にするため、**HTTP statusを保持する変更も必要**です。stderrの文言だけで分類せず、応答statusを取得します。[HTTP応答の公式定義](https://docs.github.com/en/rest/pulls/pulls#merge-a-pull-request)

   また、現在は送信前に `merge_attempted=True` を保存し、以後は再送を拒否します。確定した405拒否まで応答不明にしてしまわないよう、確定拒否と通信結果不明を分ける必要があります。405を次の操作で再試行する場合も、保存したHTTP拒否を識別し、全検査を再実行する限定経路にします。

   1 commit・3ファイル追加・mode/blob/本文一致・base祖先・ディレクトリ未存在・PR identity/変更ファイル一覧・merge後のmain照合は、いずれも `mergeable` に依存していません。

2. 続行処理は、既存CLIの明示的な呼出箇所に集約する案を推奨します。

   同じrepo・親のjournalを走査し、各対象を一度ずつ処理します。timer・poll・追加hookは不要です。

   | 入口 | 続行するタイミング |
   |---|---|
   | `run-worker`（initial/resume）、`start-worker`、`continue-worker` | 所有者確認後、新しいturnの予約・起動前 |
   | `publish-worker-log` | 指定対象と同じ親の残件をまとめて処理。指定対象を二重処理しない |
   | `publish-decision-log` | 判断ログの公開前にworkerログを照合 |
   | `release`、`handoff`（acceptを含む） | 担当変更とその記録を確定した後。詳細は次項 |
   | 下記のその他の書込み操作 | 本来の書込みを確定した後 |

   その他の対象は、`register`、`summary`、`checking`、`agree`、`agreement-ignore-comments`、`basis-record`、`basis-revoke`、`note`、`notify`、`decide`、`create-child`、`receive-proposal`、`delegate`、`configure-delegation`、`record-worker-run`、`record-worker-ended`、`record-result`、`record-cancellation`、`decide-worker-plan`、`refresh-worker-plan`、`workflow-report`、`sync`、`reconcile`、`stop`、`resume`、`stop-worker`、`improvement`、`improvement-publish`、`improvement-decide`、`review-prepare`、`review-request`、`review-state`、`retry-decision-capture`、`evidence --create`、Project操作の `--apply` です。

   入力・所有者確認で拒否された操作では再開しません。書込み確定後に後段が失敗した場合は、確定済みの操作に対する後処理として一度試します。公開失敗は元操作を取り消さず、workerの終了値も変えません。未公開の内訳と再開先を返し、明示公開CLIでは従来どおりエラー終了を返します。

   ロックは次の順序にします。

   - 親operation lock → registry lockで、所有者・generation・対象を確認する。
   - 両ロックを解放して公開する。公開中の排他は既存のturn journal lockを使う。
   - registryへ結果を記録する際は再取得し、generationを照合する。

   `run-worker` は内部でoperation lockを取得するため、その外側を同じロックで包むと再入問題が起きます。自律worker入口にも既存のロック境界があるため、共通処理を既存 `with` の内側へ単純追加しない設計が必要です。

   読取り入口の `status`、`worker-log-status`、`inspect-worker`、`decision-log-status`、`review-inspect`、`basis-input`、`basis-check`、`wait-supplement`、各dry-runは続行対象から除外します。`state-migrate` も親の通常操作とは別扱いにします。

   **現行実装には読取り契約との食い違いがあります。** `worker-log-status` は [service.py:229](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/worker_logs/service.py#L229) から判断ログを書き、CLIのcapture adapterも動きます。`publish-worker-log --dry-run` も同様で、Publisherはdry-runでもjournal lockを作ります。これらを除去し、通常の `status` もlock作成まで含めて書込みをしない経路にする必要があります。

3. 終了・引継ぎは確定させ、その末尾まで捕捉してから公開します。

   推奨順序は、担当変更の保存 → workerログの残件照合・続行 → `decision_capture.complete()` による終了／引継ぎ結果の捕捉 → 判断ログ公開です。現在のcapture完了は [cli.py:835](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/parent/cli.py#L835) にあるため、それより前に判断ログを公開すると今回の末尾が残ります。

   判断ログは `Journal.status()` と `publication.json` を突き合わせます。既存の未完了batchを先に完了し、その後、今回捕捉した終了記録までを次batchへまとめます。対象IDの上限を固定し、途中で増え続ける記録を追うループにはしません。公開処理自身をcaptureへ戻さず、公開の連鎖も避けます。

   残件があれば、出力とprivateな終了・引継ぎ記録へ次を残します。

   - workerの親・担当issue・公開キー・turn・状態・エラー・PR
   - 判断ログの未公開[local path]、未完了batch、今回の捕捉上限
   - 担当状態、引継ぎ先の有無、実行可能な再開コマンド
   - `parent_action_applied=true` と `saved_to_main=false`

   引継ぎ先が未確定なら「公開担当未割当」と明記します。公開失敗を理由に終了・引継ぎを拒否したり、終了済み担当をactiveへ戻したりしません。

   追加で二点あります。

   - [decision_logs/publication.py:583](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/decision_logs/publication.py#L583) と599行にも同じnull停止があります。判断ログもその場で完了させるには、こちらも同じHEAD指定mergeへ進める必要があります。
   - 自律親は `release` を通らず、[supervisor.py:1138](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/dispatcher/supervisor.py#L1138) でendedになります。`workflow-report` 後の公開に加え、この既存終了イベントのロック解放後にも残件照合が必要です。

4. 一時エラーの回復は、限定した通知形式と完了証拠の組合せで判定します。

   現在の [agent_events.py:294](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/lib/agent_events.py#L294) は、`error` と `turn.failed` を両方とも永久に `failed=True` にします。書込みを伴わないイベント再生で、次を確認しました。

   | イベント列 | 現行判定 |
   |---|---|
   | 再接続通知 → 応答 → `turn.completed` | failed |
   | 再接続通知 → 未完了 | failed |
   | 致命的error → 応答 → `turn.completed` | failed |
   | `turn.failed` → 応答 → `turn.completed` | failed |

   CLIの公式資料は `error`、`turn.failed`、`turn.completed` を列挙していますが、再接続通知のmessage文法やretryable属性までは保証していません。そのため、messageだけで広く成功へ変える判断は避けます。[公式JSONLイベント資料](https://learn.chatgpt.com/docs/non-interactive-mode#make-output-machine-readable)

   推奨する回復条件は、Codexに限り、全errorが確認済みの再接続通知形式であり、それらが最終応答・唯一の `turn.completed` より前にあることです。最初は `Reconnecting... n/m (workspace routing discovery timed out)` を限定的に認識し、未知のerrorや `turn.failed` が一件でもあれば回復扱いにしません。そのうえで、CLI終了値0、非空finalとイベント本文の一致、session/turn・実効設定の照合、cleanup完了、残存プロセスなし、子孫の強制回収なしを従来どおり要求します。完了後のerrorも回復対象にしません。

   [sources.py:261](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/worker_logs/sources.py#L261) はrunnerの `completed` を要求し続けてよく、finalの存在だけを公開条件にはしません。

   ただし、**既存の失敗記録はランナー修正だけでは直りません**。#1877 turn 3を回収するなら、元の `result.json` を保持し、保存済みイベント・final・metadataを同じ分類関数で再検証する明示的な履歴照合が必要です。このホストには当該journal／元runが見つからず、実ログの確認は[親issueの報告](https://github.com/Motoki0705/tennis-lab/issues/1900#issuecomment-6817317609)までです。

   また、dispatcherの判定とops-reportには `runtime_errors` 非空だけで拒否する箇所があります。エラー原文を保持するなら、これらも共通分類へ揃える対象になります。

5. このホストの照合結果と、一括続行入口の案です。

   2026-11-13 **15:25 JST**の指定state配下の集計です。調査中に別workerのcaptureが一件増えたため、開始時19件、終盤20件でした。

   | state | stage | 件数 |
   |---|---|---:|
   | pending | mergeability_pending | 10 |
   | merged | merged | 3 |
   | worker_failed | awaiting_final | 3 |
   | captured | awaiting_final | 4 |
   | blocked（absolute_pathを含む） | — | 0 |
   | ready・その他 | — | 0 |
   | **合計** | | **20** |

   指定の `gh pr list` を実行後、headの `worker-log/` prefixで絞ると23本でした。検索結果には対象外branchが5本混ざっていました。以下の「保留」はすべて `pending / mergeability_pending`、「不在」は指定state配下に対応journalがないことを示します。

   | PR | 親 | 担当issue / turn | journal |
   |---|---:|---|---|
   | #1784 | 1723 | 1727 / 2 | 不在 |
   | #1873 | 1797 | 1861 / 3 | 不在 |
   | #1874 | 1797 | 1861 / 4 | 不在 |
   | #1884 | 1797 | 1862 / 2 | 不在 |
   | #1889 | 1797 | 1862 / 3 | 不在 |
   | #1909 | 1797 | 1863 / 3 | 不在 |
   | #1911 | 1797 | 1863 / 4 | 不在 |
   | #1912 | 1797 | 1863 / 6 | 不在 |
   | #1919 | 1797 | 1914 / 1 | 不在 |
   | #1920 | 1893 | 1915 / 2 | 不在 |
   | #1921 | 1881 | 1881 / 1・`d-0fd767fb…` | 保留 |
   | #1922 | 1913 | 1913 / 2 | 不在 |
   | #1926 | 1866 | 1877 / 4 | 不在 |
   | #1928 | 1866 | 1877 / 5 | 不在 |
   | #1929 | 1881 | 1881 / 1・`d-dea8ffe0…` | 保留 |
   | #1935 | 1899 | 1899 / 2 | 保留 |
   | #1936 | 1881 | 1881 / 3 | 保留 |
   | #1939 | 1881 | 1881 / 4 | 保留 |
   | #1951 | 1878 | 1878 / 2 | 保留 |
   | #1957 | 1893 | 1916 / 1 | 保留 |
   | #1958 | 1859 | 1859 / 2 | 保留 |
   | #1959 | 1893 | 1917 / 2 | 保留 |
   | #1963 | 1893 | 1916 / 2 | 保留 |

   対応する10件はPR HEADとjournal HEADが一致し、readonlyでのpayload再生成一致・schema・ローカルGit差分・base祖先・対象ディレクトリ未存在の検査に通りました。実merge可能という認定ではなく、GitHub側の最終検査は未実施です。

   親 #1723・#1866・#1913 はregistryにも不在でした。#1797・#1878・#1881・#1899 はended、#1859・#1893 はactiveです。#1919と同じissue/turnの現存journalは**公開キーが異なる別系列**なので流用できません。#1880は11月12日にmerge済みでした。過去に報告された22本との完全な差分は、その時点の一覧が必要です。

   `captured / awaiting_final` の4件は #1859 turn 3、#1961、#1962、#1964で、いずれもrunner記録はrunning、cleanup未完了、finalファイルなしでした。今回の時点では通常の実行中です。別の `worker_failed` 3件は #1914の `descendants_reaped` と、#1859・#1899の `stopped` であり、再接続誤判定として救済してはいけません。以前の `absolute_path` 12件が解消したとは、このホストの0件からは判断できません。

   一括入口は既存CLIの拡張を推奨します。

   ```text
   parent --repo Motoki0705/tennis-lab --parent 1900 \
     publish-worker-log --all-parents --dry-run
   ```

   `--parent 1900` は回収を実行する親、`--all-parents` は同repoの既存journalを対象にする明示指定とします。実行時は回収親のactiveな所有者を確認し、元親の担当を取得・変更しません。既存journalの公開キー・branch・HEAD・PRを再利用し、workerは起動しません。絞込み用の `--source-parent` も用意すると再開先を具体化できます。

   dry-runはlock作成・capture・journal更新・fetch・GitHub書込みを行わず、対象ごとの所有状態、公開キー、turn、PR、状態、予定動作、拒否理由、再開先と件数を返します。今回の材料なら「続行候補10、実行中4、失敗3、公開済み3、open PRに対応するjournal不在13」です。

   不在13件はPR本文や公開ファイルからjournalを再構成せず、元journalの所在を確認して回収します。指定state配下とホストの `.local/state` 内を探索しましたが、見つかりませんでした。

実装時の検証は既存の `test_worker_log_publication.py`、`test_agent_runtime.py`、`test_parent_lock_scope.py`、親の終了・引継ぎテストを拡張する構成です。null時merge成功、405/409、応答喪失、次操作での続行、終了記録を含む公開、状態表示[local path]。今回の実測はイベント再生4列と既存payloadのreadonly検査10件で、pytest・pre-commitは実行していません。

親に判断してほしい点は次のとおりです。

- 判断ログ側のnull停止2か所と、自律親のsupervisor終了経路も同じ実装範囲へ含めることを推奨します。
- #1877 turn 3などの既存誤判定を、元runner記録を保持した明示的な再検証で回収するかを決めてください。実施には元runの所在確認が必要です。
- 不在13件の元journalがあるホスト・保存先の確認が必要です。このホストだけでは全件回収の受入証拠を揃えられません。
- 「既存テスト変更なし」は、nullで停止することを期待するテストと両立しません。その期待値・停止用fixtureは更新し、安全検査の拒否条件とassertを維持する解釈を推奨します。