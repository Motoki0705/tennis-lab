調査1〜6をまとめました。推奨は、**prepare時に選択をrequestへ固定し、execution・runtime・公開証拠を同じ検証部品で照合する方式**です。既定のAstra固定は維持できます。

今回確認したHEADも`98817eae6`で、branchは`codex/review-model-selection`、worktreeはcleanです。実装・コミット・PR作成・GitHubへの書込みはしていません。今回はpytestも実行していません。

1. **現状の固定経路**

   | 段階 | 確認した実装 |
   |---|---|
   | 設定 | `REVIEW_CODEX_MODEL=gpt-6-astra`、`EFFORT=xhigh`、`SERVICE_TIER=default` |
   | runner入口 | `agent_exec.sh`はPythonへの転送だけ。モデル指定の拒否は[agent_settings.py:79](https://github.com/Motoki0705/tennis-lab/blob/98817eae6cbd78edeb124e962f532414ccf9517d/.agents/ops/lib/agent_settings.py#L79)。全profileについて`--model[local path]`との併用を拒否する |
   | 正式レビュー | `parent_review.py`は`codex / review / read-only`、MCP、出力schemaを指定してrunnerを1回呼ぶ。モデルは渡さず、runnerが設定から解決する |
   | 保存・実効値 | `agent-runtime/result.json`の`contract.settings`へ選択値・設定出典を保存。Codex transcriptの`turn_context`から実効model・effortを読み、selectedとの不一致を拒否する |
   | recover | [review_recovery.py:106](https://github.com/Motoki0705/tennis-lab/blob/98817eae6cbd78edeb124e962f532414ccf9517d/.agents/ops/review/review_recovery.py#L106)で元runtimeのselected/effective、イベントとfinal、schema、終了・cleanup等を再検証する |
   | 公開・gate | `reviewer="codex"`、`profile="review"`を検証。通常resultのgateは公開コピー・digest・工程を検査するが、model・effortを直接再照合しない。adoptionでは元runtimeまで検証する |
   | inspect | 終了状態等は表示するが、選択モデル・理由・実効モデルは現在表示しない |

   **現状のrequestにはモデル選択欄がありません。** 固定値がrequestから下流へ流れるのではなく、runnerで初めてモデルが決まります。recoverはruntime内部の選択と実効値を照合していますが、requestとの照合はできません。

2. **選択の固定場所・fixed/select・旧request**

   `agents.conf`には既存pinを維持し、次を追加する案です。

   ```ini
   REVIEW_SELECTION_MODE="fixed"
   REVIEW_SELECTION_POLICY_VERSION=1
   REVIEW_CODEX_ALLOWED_PAIRS="gpt-6.1-sol:xhigh,gpt-6-astra:xhigh"
   ```

   Sol/xhighは初期候補の提案です。実装workerのmaxを流用せず、レビューの現行effortへ揃える案であり、実モデルでの比較・利用可能性は未検証です。

   | mode | 推奨する扱い |
   |---|---|
   | fixed | 省略時は従来のAstra/xhigh/default。明示指定は固定値と一致する場合だけ許可し、異なる指定は起動前に拒否する |
   | select | モデルと選択理由を必須にする。effortは許可された組から検証し、曖昧なら明示指定を要求する |

   人間の指定は、元指示への参照とモデル拘束を構造化して渡します。親候補が拘束と矛盾すれば拒否します。許可リストやfixedと矛盾する指定も、別モデルへ置き換えずエラーにします。自由文の解釈と人間発言の確認は親が行い、CLIは渡された拘束を検証する責務です。

   新requestは**version 3**として、次をdigestへ含める案です。

   - 正規化したreview方針：版、mode、固定値、許可された組、tier
   - 選択model・effort・tier
   - 選択元：`settings / parent_choice / human_instruction`
   - 選択理由、人間指定の拘束・参照
   - 引継ぎの場合の元runと追加確認範囲

   同じ選択をexecutionにも保存します。既存の`request.reason`は「なぜ追加レビューするか」であり、「なぜそのモデルか」と分けます。

   固定時点には注意が必要です。[snapshot.prepare](https://github.com/Motoki0705/tennis-lab/blob/98817eae6cbd78edeb124e962f532414ccf9517d/.agents/ops/parent/snapshot.py#L115)がrequestを保存・公開し、digestを確定します。そのため、`review-request --prs`だけでなく、分離操作の`review-prepare`も同じ選択引数を受け取る必要があります。`review-request --run-id`は保存値を使用し、異なる指定を拒否します。prepareの同一run再利用条件にも選択一致を加えます。

   旧version 1/2は原文・digestを保持し、選択理由を補作しません。既存の公開結果、recover、公開再開、gateは旧契約として検証します。表示は「要求時の選択未記録」とruntimeに残る実際の選択を区別します。未起動の旧requestは従来fixed構成と一致する場合だけ従来実行を許し、selectでの起動には新版requestを要求する案です。

3. **実効値の照合と、各消費側の変更**

   `lib/review_settings.py`等へ、方針解析・選択検証・記録間の照合をまとめます。runnerにはreview専用の選択JSON入力を追加し、schema/MCPと同様にrunへ固定する案です。他profileの汎用override制約を変更せずに済みます。

   | 境界 | 検査・表示 |
   |---|---|
   | 起動前 | request＝executionの選択、現在のreview方針との整合、許可された組、Codex/read-onlyを検査 |
   | runner終了時 | 要求した選択＝runtime.selected＝実効model・effortを検査 |
   | 公開・公開再開 | 保存選択・実効値・runtime digestを照合。不整合を成功結果として公開しない |
   | recover | 元runのrequest・execution・selected・effectiveを照合。現在の既定モデルへ読み替えない |
   | inspect | 方針版、mode、選択元・理由、選択値、実効値、照合エラー、採用元を表示 |
   | gate | 新版の通常resultもruntime証拠を読み、同じ検証を使用する |

   model・effortの実効値は`observed`必須にします。**service tierは現行runtimeで取得できない場合があるため、理由付き`unavailable`を維持**します。観測できたtierは一致を要求します。未取得値をselectedで埋めません。

   利用不能は既存の`cli_error/startup_failed`、実効値の不一致は`metadata_error`として失敗させます。同runを別モデルで再実行しません。runnerのresumeでは保存選択を復元して照合し、モデル・effort・関連方針の不整合を起動前に拒否します。現在の「前回overridesを汎用引数へ戻す処理」も、新しいreview専用入力に対応させる必要があります。

   公開上の`reviewer="codex"`、`profile="review"`は維持します。選択はrequestへ、実効値はhostが検証して付加する記録へ載せます。モデル自身の自己申告を証拠にしません。歴史的証拠は保存時の方針で検証し、現在のfixed/select変更だけで失効させない設計です。

   version 3導入では、`parent/state.py`の読取り許可と、[execution.py:442](https://github.com/Motoki0705/tennis-lab/blob/98817eae6cbd78edeb124e962f532414ccf9517d/.agents/ops/parent/execution.py#L442)のrequirements照合も変更が必要です。後者は現在version 2限定なので、更新漏れがあると新版で対象照合を飛ばします。

4. **Sol→Astraの別run引継ぎ**

   親が新しいモデル・選択理由・再レビュー理由を明示し、新requestへ以下を固定します。

   - 元run IDと元request/resultのdigest
   - 対象PRとsnapshotの対応
   - 未解決指摘・未確認事項・追加確認範囲

   既存の`exception.previous_runs`は過去の実行一覧なので、具体的な引継ぎ元を示す欄を別に設けます。旧runは変更しません。

   `superseding_reviews`の順序・判定は変更せず、後続の失敗・中断・取消も残します。新runは既存のPR leaseと終了・cleanup確認を通過してから起動します。元runが終了未確認なら、復旧または`close-interrupted`が先です。1要求1起動を維持し、モデルを変える連鎖起動は作りません。

   **recoverは別の扱いが必要です。** Solのfinalを採用するとき、採用先requestの既定Astraを実行モデルとして記録してはいけません。新版の復旧要求には元run参照を固定する入力を設け、実効モデルは元runの検証済み値を表示する案を推奨します。この要求からagentを起動する操作は拒否します。

   現行coverageはPR単位です。Astraが同じPRの一部分だけを確認しても、Solとの合成coverageにはなりません。子3では引継ぎ記録と既存coverage契約を維持し、部分確認しか成立しない場合は`not_reviewed`でgateを止める案です。**PR内の部分coverage合成まで必要なら追加設計になります。**

5. **自律親の経路と最小変更・下流影響**

   自律親の経路は既にあります。

   `supervisor → parent_bootstrap.md → development_workflow.md → parent review-request → parent_review.py run`

   [開発workflowのコマンド](https://github.com/Motoki0705/tennis-lab/blob/98817eae6cbd78edeb124e962f532414ccf9517d/.agents/ops/parent/development_workflow.md#L96)が手動親と同じ入口です。dispatcherのPythonコードはレビューを直接起動しません。

   最小変更は、共通parent CLIへの引数追加と、workflow・bootstrapへの選択手順の追記です。dispatcherの予約・supervisor・timerに新しい選択処理や起動処理は不要です。

   共通libの具体的な影響先は、手動workerのモデル解決・実行契約比較、自律workerとClaude親の起動・継続、`ops_report / ops_scout / issue_judgement`の固定profile、`read_pins`を使うagy・説明HTML、runtimeを読む判断記録です。review専用分岐に限定し、他profileの設定・argv・継続契約を維持する検証が必要です。`agent_exec.sh`本体は転送だけなので変更不要と考えます。

   #1796接続では、現行`parent_commands.py`が`args.model`だけで選択元を判別します。`--review-model`追加だけでは指定元が`settings`のままになるため、#1863へ渡す接続点として明示します。

6. **実装単位・テスト設計・見積もり**

   | 作業単位 | 主な変更ファイル | 見積もり |
   |---|---|---:|
   | 方針とrunner | `agents.conf`、新`review_settings.py`、`agent_settings.py`、`agent_runtime.py` | 1.5〜2時間 |
   | requestへの固定 | parentの`cli.py / review_handoff.py / snapshot.py / state.py / execution.py` | 1.5〜2時間 |
   | 証拠と引継ぎ | reviewの`parent_review.py / parent_review_policy.py / review_recovery.py / review_inspection.py / merge_gate*.py` | 2〜3時間 |
   | 回帰確認・文書 | README、workflow、fake producer、テスト、pre-commit・CI確認 | 1.5〜2時間 |

   合計**6.5〜9時間程度**です。CI待ち・基盤障害と、PR内coverage合成は含めません。

   テストでは、守る契約を「選択の固定、実効値との一致、起動一回性、既存gate条件の維持」とします。異常分類はunitに寄せ、公開入口で全組合せを重ねません。追加は30〜45ケース程度を目安とし、件数・時間は実装時に収集・計測します。

   | 検証対象 | 必要な追加・拡張 |
   |---|---|
   | 選択 | fixed省略・一致・不一致、selectのSol/Astra、許可されない組、理由欠落、人間拘束との矛盾 |
   | fake runtime | 通常確認のSol、高リスク確認のAstra、人間指定、実効model/effort不一致、利用不能、継続時の変更拒否 |
   | 共通契約 | 両モデルのread-only・MCP・schema・追加agent無効化、手動/自律の同一経路 |
   | 証拠消費 | recover・公開再開・inspect・gateが同じ不一致を拒否し、復旧を別モデル実行と偽装しない |
   | 引継ぎ・旧request | 元run対応、PR排他、後続requestによる旧証拠採用拒否、旧digest不変・理由の補作なし |

   既定fixedの退行確認には、既存の`test_profile_uses_its_own_pins`、`test_review_mcp_is_read_only_and_records_frozen_configuration`、`test_failed_turns_preserve_evidence_and_never_claim_task_success`、`test_combined_handoff_registers_results_and_instructs_subset`、`test_later_review_request_supersedes_only_its_pr`、recover・公開再開・gate・排他の既存nodeを再利用します。意味上の期待動作とAstra/xhigh/defaultは維持し、新request版への対応だけを追加します。

   現在の`FAKE_PARENT_REVIEWER`は`review-fixture/max/priority`という値を生成するため、新版の照合を入れる際はfake producerも呼出しの選択へ対応させる必要があります。拒否条件やgate検査を弱める対応はしません。

   前回の実測は、既存9ケースを`-n 0`で実行して**7 passed・1 failed・1 error、649.26秒**でした。固定profile・MCP・実効値不一致・CLI失敗・設定移動後のresume・統合handoffは成功。旧runのrecoverとgate fixtureの`parent register`は90秒timeoutで、コード未変更の基準結果です。review/gate全体の通過は未確認です。今回は再実行していません。

実装前に親へ返す判断は、**Solの初期effortをxhighとする案、fixedで一致指定だけ許す案、復旧要求へ元run参照を固定する案、子3ではPR内coverage合成を追加しない案**です。実モデルの利用可能性・検出力・処理量は未確認で、比較評価は#1863の範囲です。