調査のみ完了しました。**51ターン・対応4・未対応47を再現済み**です。実装は、READMEの生成を公開物だけで完結させ、private stateで確認した事実は出典付きの公開観測として追加する方針を推奨します。

1. 固定mainの集計漏れは、親番号による除外で再現しました。

   `c807cd2a1a0ff8e46ace1ae740b2c2c0f72672f4` の `.agents/logs/parents` を `git archive` でメモリ上に読み、全workerの3ファイルを `worker_logs.schema.validate()`、全イベントを `navigation.records()` で検証しました。

   | 親 | 公開workerターン | 対応イベントあり | 対応未確認 |
   |---|---:|---:|---:|
   | #1581 | 7 | 0 | 7 |
   | #1694 | 1 | 0 | 1 |
   | #1698 | 10 | 0 | 10 |
   | #1710 | 24 | 0 | 24 |
   | #1796 | 6 | 4 | 2 |
   | #1797 | 3 | 0 | 3 |
   | **合計** | **51** | **4** | **47** |

   指示・最終報告の102ファイルはmetadataのdigestと一致しました。イベントは404件、明示判断は2件です。

   対応キーは次です。

   ```text
   (repo.casefold(), parent, issue, delegation_key, turn.sequence)
   ```

   イベント側では `scope.delegation.key` と `scope.delegation.turn` を使います。対応した4ターンは、#1816の1・4、#1817の1・2です。activity ID、私的session ID、initial/resume、phaseは対応キーにしません。

   45ターンが落ちる箇所は [navigation.py:173](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/decision_logs/navigation.py#L173) の `parent not in parents` です。`parents` はイベントを持つ親だけなので、残る5親をmetadata検証・ターン登録の前に除外します。また、133行の早期returnにより、イベントが一切ない入力ではREADME自体を生成しません。

   修正では全workerを先に登録し、イベント集合と照合します。「導入前」は断定せず、**「この公開snapshotに対応イベントなし。導入時期・未捕捉・捕捉失敗・未公開の区別は未確認」**と表示するのが妥当です。47は失われた親判断の数ではありません。

   **#1816ターン2・3は、この51ターンの分母に含まれません。** 両者はイベントだけが存在するため、既知ターンの和集合は53です。この分母はREADMEで分ける必要があります。

2. #1816ターン2・3の「最終報告なし」は、固定main内のイベントだけでは確定できません。

   両ターンには各6イベントがあり、内容は起動観測と `captured / awaiting_final` です。`ended` もありますが、これは **`worker_logs.status` コマンドの終了**であり、workerの終了ではありません。

   欠落の明示根拠は、[親の完了報告](https://github.com/Motoki0705/tennis-lab/issues/1796#issuecomment-6786435778)の「ターン2・3はメモリ圧迫で中断し最終報告なし」です。これを対象ターンへ結ぶ観測の追記が必要です。

   | 表示 | 必要な根拠 |
   |---|---|
   | 最終報告なし・根拠あり | 同じターンの終了／中断とfinal欠落を示すrunner観測、または親の明示報告 |
   | 稼働中 | 対象runの稼働を確認した観測。観測時点を表示 |
   | 終了未確認 | 起動・`awaiting_final`だけで、終了確認がない |
   | 最終報告あり・公開保留 | 非空finalの観測／固定receiptと、公開のpending・blockedを独立して確認 |

   [observations.py:119](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/decision_logs/observations.py#L119) のrunner捕捉は、失敗したrunでもfinalがあれば観測できます。一方、workerログ捕捉は現在、`saved_to_main=true` の場合だけfinalを設定します。公開成功とfinalの有無を分離する必要があります。

   終了記録だけからfinal欠落を推定せず、`awaiting_final`やreceipt未保存だけでも欠落と認定しません。重複排除は上記ターンキーで行い、起動・終了・状態取得・公開再試行のイベント数を欠落ターン数へ加算しません。

   README生成時のprivate state参照は不要にできます。ただし、その前段の捕捉ではrunner／receipt／親の明示記録を読み、安全な観測を公開する必要があります。#1814の番号の矛盾は、[途中報告](https://github.com/Motoki0705/tennis-lab/issues/1796#issuecomment-6767080512)と完了報告の両方を残して「要照合」とし、自動解決しません。

3. #1862の途中ターンは、PRと固定HEADから所在を確認できました。

   | ターン | 確認した所在 | 確認内容 |
   |---|---|---|
   | 2 | [PR #1884](https://github.com/Motoki0705/tennis-lab/pull/1884) | 現在open。固定HEADの3ファイルを検証 |
   | 3 | [PR #1889](https://github.com/Motoki0705/tennis-lab/pull/1889) | 現在open。固定HEADの3ファイルを検証 |
   | 4 | 固定mainの `0004-resume/` | instruction・final・metaの3ファイルを検証 |

   ターン2・3のmetadataはいずれもruntime completedです。ターン2・3のファイルは固定mainにはありません。PR上のファイルをmain入力へ混ぜると誤表示になるため、外部の公開根拠として扱います。

   捕捉元はworker journalの `source.public_key / sequence`、`receipt`、`files`、`state / stage / error`、`pr_number / head` と、照合したmainのrevision・内容です。既存の公開観測はPR番号をsummaryに残していないため、現在のmainだけからPRを復元することはできません。

   提案は、次の値を型付き観測として公開することです。

   ```text
   execution: 実行状態・終了根拠
   final: 有無・根拠
   publication: main / public_pr / pending / unknown
                stage・error・PR URL・PR HEAD・照合先main revision
   ```

   private journalを生成時に直読すると、同じ公開HEADから異なるREADMEができ、公開検査側でも再現できません。**journalを読む捕捉段階と、公開snapshotだけを読む生成段階を分けます。**

   PR merge後は、次の対象書込みCLIや終了・引継ぎで再照合し、観測を固定して再生成する方式が適しています。既存の `publish-decision-log --retry` には、後着workerファイルによるREADME修復経路があります。どの既存CLIから呼ぶかは#1961との接続事項です。再生成までは「観測時点の公開PR」と表示します。

4. 親の終了を辿るには、受入根拠・担当終了・公開残件を別々に公開する必要があります。

   #1796の固定main末尾は、11月11日12:24 JSTのmerge行動です。[PR #1898](https://github.com/Motoki0705/tennis-lab/pull/1898)の公開後に、[完了報告](https://github.com/Motoki0705/tennis-lab/issues/1796#issuecomment-6786435778)と[担当終了](https://github.com/Motoki0705/tennis-lab/issues/1796#issuecomment-6786439334)があり、その参照がmainにありません。

   #1864も、[完了報告](https://github.com/Motoki0705/tennis-lab/issues/1864#issuecomment-6785704100)と[担当終了](https://github.com/Motoki0705/tennis-lab/issues/1864#issuecomment-6785708290)は確認できましたが、固定main・現在のcheckoutとも親別ディレクトリがありません。treeだけでは、この親の存在も列挙できません。

   #1961から表示側へ渡してほしい記録案は次です。

   | 記録 | 必要な値 |
   |---|---|
   | 対象・範囲 | repo、parent、generation、観測時刻、照合先main revision、照合対象の範囲 |
   | 最後の受入 | acceptance状態、既存判断IDまたは正本コメントのURL・内容fingerprint。不明なら明示 |
   | 終了・引継ぎ | ended／handoff pending／accepted、終了根拠の正本参照 |
   | 残る公開 | workerのターンキー、判断ログの未公開範囲、state・stage・error、既知PR |
   | 担当・再開先 | 公開用actor、担当なしの区別、再開操作と対象キー。私的session ID・絶対パスは含めない |

   正本コメントは現在の共通history readerで取得し、証拠用Issueへの移動にも対応させます。private側は既存のoperation receipt／capture outboxを使い、公開側は既存の `activity-…/event-….json` に追加する案です。新たな並行journalは不要です。

   **公開順序が重要です。** 現行CLIは [cli.py:835](https://github.com/Motoki0705/tennis-lab/blob/b734cb96a6407426b8d58c85156efe596c4e77a0/.agents/ops/parent/cli.py#L835) で最後の結果を捕捉します。`release` の処理途中だけで公開すると、その終了結果が再び未公開になります。終了根拠・CLI結果の捕捉を確定してから公開し、公開処理自身の成功記録が次の公開を要求する循環は避ける必要があります。

   公開が失敗した場合は、終了自体を取り消さず、残件・担当・再開先を出力と永続記録へ残します。mainの表示はsnapshotなので、再照合・公開が成功するまで最新終了を反映できない限界も明記します。

5. 実装は、集計・観測契約・表示と公開接続に分けるのがよいと考えます。

   - 全workerとイベントの和集合からターンを列挙する純粋処理を切り出し、分母・対応率・重複排除を修正する。
   - 実行[local path]。現行v1はstrict schemaのため、summary文字列の解析に依存せず、版を付けた拡張を推奨する。既存イベントは変更しない。
   - READMEへターン状態・公開先・最後の受入／終了根拠を表示し、#1961の照合結果を接続する。仕様READMEは英日中を更新する。

   再利用できる主なテストは次です。

   | 既存node | 追加・拡張する入力 |
   |---|---|
   | `test_decision_acceptance.py::test_published_real_record_copy_can_be_regenerated_under_existing_exception` | 固定mainのmanifest。404イベント＋153 workerファイルの557パスを選び、51／4／47を固定 |
   | `test_decision_navigation.py::test_existing_turn_links_match_worker_metadata_without_copying_its_text` | イベントなしの5親、workerだけの入力、同一ターンの複数観測 |
   | `test_decision_capture_delivery.py::test_runner_start_end_and_missing_final_are_not_parent_confirmation` | 4状態、別操作のended、親の別記録による欠落確認、失敗runにfinalあり |
   | `test_decision_publication.py::test_late_worker_files_refresh_navigation_without_new_capture_records` | #1862型のPR2本＋main1ターン。後からPRがmergeされ、古い公開観測が残る入力 |
   | `test_decision_capture.py::test_note_reference_stop_resume_handoff_and_release_are_captured` | 最終公開後の完了・担当終了、親ディレクトリなし、引継ぎ、公開中断と再照合 |
   | 既存の公開policy・presentation負例 | 原記録不変、生成一致、unsafe値拒否、任意summaryを型付き根拠へ昇格させないこと |

   固定mainの既存公開ファイルをmanifestで選ぶ方式なら、原文の複製やCIでの過去git履歴・ネットワーク取得は不要です。今回の実測では入力約1.81 MB、現行生成は約0.80秒、87 README・最大30,680 bytesでした。

未確認なのは当時のprivate状態です。既定state領域では、#1796・#1864のjournalと#1862の旧ターンjournalを見つけられませんでした。消失したとは判断していません。

branchは `codex/log-turn-traceability`、HEADは `b734cb96a` のまま、worktreeはcleanです。コード・ログ・stateの変更、commit・PR作成はしていません。pytest・pre-commit・CIは実行していません。

親に判断してほしい点は次の3点です。

- **観測契約**：公開用の型付きsnapshotを追加し、READMEは公開物だけから生成する方針で進めるか。schemaの版と#1961から渡す値を先に共有したいです。
- **過去事例の取り込み**：#1816ターン2・3、#1796・#1864について、既存の明示コメントを出典とする観測を限定的に追記するか。#1814は要照合のまま保持する提案です。
- **#1961との処理順序**：終了結果の捕捉後に公開を行う位置、worker PR merge後の再生成契機、公開が残った場合の担当・再開先を誰が記録するかを確定したいです。