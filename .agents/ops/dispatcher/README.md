# issueの自律着手判断

`judgement` はopen issueを収集し、固定した入力とコードをread-onlyのClaude親に渡す。
構造化結果を検証したスクリプトだけが判断コメントと自律着手根拠を記録する。
正式な子は既存の最上位親へ戻し、単独issueは自身を候補にする。新しい階層や担当は作らない。
親の起動・予約・実行枠・再開・schedule/systemd、実装子への委任、レビュー・mergeはこの入口の責務に含めない。

| ファイル | 責務 |
|---|---|
| `collect.py` | 全ページ収集、正式関係と依存の閉包、registryの読取、固定コード、入力射影・鮮度照合 |
| `agent_input.py` | 完全なsnapshotを変更せず、索引・issue・関連PRを読み取り単位に分割 |
| `agent.py` / `prompt.md` / `judgement.schema.json` | 共通ランナーでの1回の判定とruntime証拠の照合 / 判断方針 / CLIへ渡す返却形式 |
| `schema.py` | 全件の構造化結果・出典・scope・完了条件・routing・機械的阻害条件の検査 |
| `records.py` | parentと共有する厳格JSON・判断コメント形式の参照 |
| `judgement.py` | 明示実行、記録前の再照合、コメント更新、実行根拠記録、割当候補の返却 |

## 明示実行と保存物

Pythonのimport rootは `.agents/ops`。`--cwd` はcommit済みでtrackedファイルがcleanなgit checkoutを指定する。
`--run-dir` は毎回新規にする。run dirは0700、JSONは0600でatomic writeする。
本文・ローカルパス等が入るため、保存物をそのまま公開issueへ貼らない。
`--parent-state-dir` 省略時は [parent](../parent/README.md#責務と状態の正本) と同じregistryを使う。

```bash
# previewでも判定agentは1回起動する。GitHubへは書かず、割当可能なcandidatesは返さない
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher.judgement \
  --repo Motoki0705/tennis-lab --cwd "$worktree" --run-dir "$run_dir"

# 判断コメントと実行根拠まで記録する。親の予約・起動は行わない
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher.judgement \
  --repo Motoki0705/tennis-lab --cwd "$worktree" --run-dir "$run_dir" --apply
```

`--timeout` は有限正数の秒（既定1800）。`--agent-exec` は通常共通ランナーのまま使い、テストで差し替えられる。
モデル・effort設定の正本は [共通ランナー](../lib/README.md)。必ず `--role parent --mode read-only` で呼ぶ。
`--read-dir <run-dir>/agent-input` でagent入力だけを追加の読み取り対象にし、runtimeの`contract.read_dirs`とも照合する。
Claudeの`--add-dir`への変換とCodexのsandbox契約は共通ランナーが扱う。
`--json-schema judgement.schema.json`で構造化出力を必須にし、runtimeの`contract.json_schema_sha256`を
渡したschemaと照合する。Claudeの自由文からJSONを探さず、ランナーが保存した検証済みの`final.md`を厳格に読む。
返却形式のschema検査に加え、`schema.py`による全subjects・出典・routing・依存等の検証を続ける。
schemaの方言と配列制約は[共通ランナーの契約](../lib/README.md#json-schemaによる構造化出力)に従う。
ランナーの起動前拒否はrunnerのstderr、`cli_error`はprivateなruntimeの`error`（CLI stderr先頭を含む）を
判定結果の`diagnostics`に残す。これらから判断コメントや割当候補を生成しない。
ランナーの正常終了、run ID・実行契約・選択値と観測model・プロセス回収を確認した後に最終JSONを読む。
exit 0や`task_success:null`をタスク成功に読み替えない。失敗時に別モデルや別sessionへ振り替えない。

保存物は `input.json`、`agent-input/`、`prompt.md`、`agent-runtime/`、`runner.stdout` / `runner.stderr`、
`agent-output.json`、`result.json`。stdoutは `schema_version/result/decision_count/deferred_count/candidate_count/diagnostic_count` の短いJSONだけ。
終了値0はpreview・記録完了・入力変更による延期、2は取得・整合・runtime等のエラー。
入力変更と障害の区別、延期の内訳はprivateなresultを読む。部分公開後の障害も `result=failed, candidates=[]`。
途中までのコメントや根拠を消さず、次の予定回または明示実行で照合する。

## 入力契約とdigest

完全な`input.json`はdigest・鮮度照合用に維持し、agentには`agent-input/index.json`から必要な資料を読むよう指示する。
索引は全subjectsと閉包のissue要約・正式関係・依存・関連PR番号・assignees・担当/委任要約・commit・digestを持つ。
各subjectのissue要約に`mechanical_guard`を含める。阻害条件なしはnull、それ以外は`disposition`と根拠となった`facts`のobject。
検証と同じ関数で計算する。これは派生情報であり、完全なsnapshotやdigestの判断材料を変更しない。
要約を1行ずつ配置し、長文を含む詳細は別ファイルにして、Readの行数制限で索引の後半を落とさないようにする。
`issues/<番号>/issue.json`は本文・関係とコメントページへの参照、`pulls/<番号>/pull.json`は関連PRのclosing本文・head・merge状態を持つ。
参照は全て`agent-input/`からの相対パス。本文・コメントは短い`segments`の順序連結、長文は`files`の全パートの順序連結で
原文を復元できる。コメントはページ分割し、内容の要約・切り詰めはしない（`content_complete=true`, `omissions=[]`）。
無関係なPR・code tree・controller内部の`execution_inputs`は渡さず、索引の`omissions`に種類・理由（PRは件数も）を明記する。
コードは固定commitのcwdを直接読む。agent向けの分割・絞り込みは`material()`やdigestを変更しない。

専用memoryの `ops-report-memory-task:v1` 作業issueは公開元レポートも閉包へ含める。
未担当の単独issueと一意なopen PRのslot/HEADが一致する場合、PR存在だけを理由にexisting_ownerとはしない。
これは正式reviewへの自律着手を判断できる入口であり、着手・合意・review・mergeを許可する記録ではない。
担当・assignee・保留・依存の拒否条件は維持する。契約は [memoryの保存](../ops-report/README.md#専用memoryの保存)。

`input.json` schema_version=1:

- `repo`（casefoldしたowner/repo）、`publisher`（現在のgh login）、`subjects`（全open issue番号の昇順配列）。
- `issues`（番号文字列→本文・コメント・state/state_reason・labels・assignees・正式parent/children・dependencies・source_reports）。
  親、子、先行依存、提案出典レポートはclosedも含め再帰的に収集する。別親に属する同repo依存も収集する。
- `pulls`（全状態・全ページのPRとclosing指示・head・merge状態）、`owners`（既存registryの状態・generation・内容digest・委任予約の投影）。
  生session ID・ローカルworktreeは判定入力や公開コメントへ渡さない。別の担当台帳は作らない。
- `code`（commitとgit treeのpath/mode/kind/oid）。agentが必要なtracked blobを調べ、そのcommit/pathを出典にする。
  gitlinkの先にある未固定ファイルやuntrackedファイルを着手根拠にしない。
- `input_sha256`、`execution_inputs`（報告容器以外について `parent.execution.execution_input()` で生成した実行根拠用の入力）。

ページは100件ずつ末尾まで読み、重複・欠落した型・正式関係の不一致・親循環・HTTP失敗を拒否する。
`/parent` の「親なし」以外の404を成功や空配列に変換しない。
正式依存は [GitHub blocked_by API](https://docs.github.com/en/rest/issues/issue-dependencies#list-dependencies-an-issue-is-blocked-by) と
本文の `## 依存関係` の同repo `#N` / issue URLの和集合。別repo依存は明示エラーにする。
欄に後続作業を並べず、先行だけを書く。自然文にしかない依存や未収集の出典が判明した場合は、agentが情報不足として示す。
依存のclosed/completedだけを充足とし、取消・duplicate・理由不明を成功とみなさない。

`input_sha256` は `collect.material(snapshot)` のcanonical JSON（parent.policy.digest）のSHA-256。
全タイトル・要件本文・指示コメント・正式関係・依存の状態・関連PR・担当/委任予約・コードを含む。
本文の表示節とAcceptanceのチェックは [実行根拠の射影](../parent/README.md#自律着手の実行根拠と呼出契約) を使う。
publisherのparent-session/review進捗は除くが、checking/agreed/basis_revokedの制御記録は保持する。
自律根拠の記録・自分の判断表示・GitHubのupdated_atだけではdigestを変えない。
自然言語コメントを機械的に無関係と分類しない。新しい通常コメントは判断材料になる。

`material(snapshot, N, sources=())` は対象の親子・依存・提案出典レポートの閉包と関連PR/担当/コードに絞った同じ射影。
判定の `sources` にあるissue・コメントの所属issueも起点に加え、その正式関係・依存・提案出典を再帰的に含める。
明示引用したPRはclosing先が閉包外でも本文・head・状態を含める。明示引用したコメントはID/login/本文も
`source_comments` に固定する。通常は除外するシステム進捗記録でも、引用した本文の編集・削除は入力変更となる。
自身の判定表示を独立出典として使うことは引き続き拒否する。
閉包外の資料を拒否せず含めるのは、別issueの合意・重複実装・関連する作業を判断材料として保持するためである。
判定前の `input.json` / `input_sha256` / `execution_inputs` は書き換えない。後者は引用を加える前の基本射影であり、
記録時の実行入力は固定snapshotと返されたsourcesから対象contextを導出し、最新の同じ射影と照合する。
判定後と全記録後は固定したsubjectsを再収集し、各subjectの対象閉包を個別に照合する。
全入力digestの一致は書込みの前提にしない。変わっていないsubjectは記録・候補化し、変わったsubjectだけを
`deferred`へ移して`diagnostics`に再判断の案内を残す。previewにも同じ判定後の照合を適用する。
各コメント・自律根拠の書込み直前・直後にも対象閉包を照合する。書込み中・記録後の変更でも対象subjectだけを延期し、
公開済みのコメントや根拠は履歴として保持する。そのsubjectの判断・候補は有効な結果として返さない。

subjectsは最初に全ページ収集したopen issue集合に固定する。agent出力の全件一致検査は維持し、
判定中に新しく開かれたissueを追加判断せず、次の予定回へ回す。既存subjectのcloseは対象閉包の変更として検出する。
新規issueでも正式な子・依存などで既存subjectの閉包へ入った場合は、そのsubjectを延期する。
registryの担当・generation・委任予約は閉包に関係する行だけを照合し、無関係な担当の変更では延期しない。
publisherと固定コードは各対象閉包に含まれるため、その変更は全subjectに影響する。
これらと全件出力検査以外に、全入力一致を要する不変条件は設けない。
API失敗・不正データ・dirtyなコード・registry破損を入力変更として処理せず、run全体を失敗にする。

API呼出しとGitHub書込は単一トランザクションではない。最終照合後にも変更は起こり得るため、予約直前にも対象閉包を再照合する。
本文・コメント・コード・関係・依存・担当の変更からagentを予定外に再起動しない。
この関数に差分による起動skipはなく、空入力や無変更回でも正常な収集後はagentを1回呼ぶ。

## 判断結果と保留コメント

agentの返却schemaと意味の正本は [prompt.md](prompt.md)。`subjects` を過不足なく1件ずつ扱う。
scriptはversion・digest・未知キー・重複・未知ID/出典・routing・dependencies・優先順とstartableの要件4項目を検査する。
全件を検証するまで書かない。不正な1件だけを捨てて残りを候補にする経路は持たない。
ラベルready/ai-proposedや人間合意の有無だけでは着手可否を決めない。

機械的阻害条件は報告容器、正式な既存子、親関係が欠けたissue:child、needs-human/停止/既存のchecking・人間合意、
既存担当・assignee・委任予約・関連PR、未充足依存。同時に複数ある場合は、報告容器→正式な既存子→親関係の欠落→
人間の保留/合意→既存担当等→未充足依存の順で最初の分類を採用する。
agentがstartableを返し、機械的阻害条件がある場合はrun全体を拒否し、書込も候補生成も行わない。
両方が非startableで分類だけが異なる場合は、記録する`disposition`を機械的阻害条件の値にする。
`rationale`に採用した分類とagentの元の分類・理由を併記し、`facts`にはagentの確認事実を保持して機械的根拠を追記する。
これを判断コメントと`result.json`の`decisions`へ保存し、`diagnostics`にもissue・両分類・採用分類を不一致として残す。
不一致だけではrunを失敗にせず、他の着手可能な対象は通常どおり扱う。元のagent出力は`agent-output.json`にそのまま保持する。
未知分類・routing/出典の不正・情報不足の具体性等は、分類を補正する前に全て検査し、不正ならrun全体を拒否する。
既存人間合意を自律根拠で上書きすることはparent APIが禁止しているため、その経路はhuman_holdとして既存担当管理へ返す。
自然言語の明示停止・見送り・対象外、重要な未決事項、scopeと完了条件の妥当性はread-only agentが原文とコードを読んで判断する。
これらの意味的判断をfixtureの正解だけで検証済みとは扱わない。

報告容器を除く判断は当該issueへ、理由・確認事実・不足情報・再開条件・次の対応・出典・管理先を記録する。
同じpublisher・repo・issueに属する厳密な `issue-start-judgement:v1` マーカーと本文を検証し、同内容なら書かず、
変わった場合はそのコメントだけPATCHする。他人の同形マーカーを自分のコメントとして扱わない。
自分のコメントの破損・複数一致は上書きせず診断する。本文や指示に編集があれば対象subjectを延期する。
判断コメントの形式・識別は `parent/judgement_comments.py` を正本とし、人間合意後のコメント検査・
自律根拠の指示射影・判定のdigestから同じ条件で除外する。先頭行のマーカー、publisher、repo・issue、
構造化したdetailとdigest、再構築した本文全体の一致を要求する。マーカーの引用・偽装・追記や他publisherは除外しない。
表示は同じコメントをPATCHし、新しい根拠の `ignored_comments` には固定しない。
表示更新で実行根拠を失効させず、同一内容の再送で投稿を増やさないためである。
旧根拠が表示のID/digestを固定している場合も、現存コメントがこの厳密な形式であると再検証できたものだけを常時除外する。
形式の破損や通常の指示・明示分類した通常コメントの編集/削除は引き続き拒否する。

startableは `execution_input()` の固定値と最新値を照合し、version付きbasisを生成して `record_autonomous()` に渡す。
人間のconfirmationのキーは生成しない。既存の明示取消・checkingを自律判断で解除しない。

## 割当側への候補契約

`result.json` とstdoutのschema_version=2。v1から`deferred`の内訳・件数と`result=deferred`を追加し、
`decisions`は鮮度照合を通ったsubjectだけを保持する契約に変更した。入力・agent出力・候補・実行根拠contextのschemaはv1を維持する。
resultは `result`、`input_sha256`、`decisions`、`deferred`、`publications`、`diagnostics`、`candidates` を持つ。

| result | 意味 | candidates |
|---|---|---|
| `preview` | GitHubへの書込みなし。有効なdecisionsと一部延期の内訳を返す | 常に空 |
| `recorded` | 鮮度照合を通った判断を記録。一部延期を含み得る | 有効なstartableのみ |
| `deferred` | 全subjectが入力変更で延期され、有効なdecisionsが0件（preview/apply共通） | 常に空 |
| `failed` | 取得・schema・runtime・記録等の障害。部分公開後も全候補を無効にする | 常に空 |

`decisions`と`deferred`は、障害のないrunでは最初のsubjectsを過不足なく分け、どちらもissue番号順。
報告容器の判断はローカルのdecisionsに記録し、GitHubへの書込みは行わない。
subjectsが最初から空の回は、延期ではなくpreview/recordedの正常な空結果とする。
`deferred`の1件は `{issue, stage, reason}`。stageは`preflight`（判定後）、`publication`（個別記録中）、
`final_check`（全記録後）のいずれか。reasonとdiagnosticsに「入力変更のため次の予定回または明示実行で再判断」を残す。
元の全判断は`agent-output.json`で確認できる。`publications`は実施済みの判断コメント操作の履歴なので、延期対象も含み得る。
延期されたsubjectはそのrunで再試行せず、変更を契機にagentを追加起動しない。

割当側はv2のresultを検査し、予約に渡せるのは `result=recorded` のcandidatesだけ。
preview/deferred/failedから予約しない。recordedに延期が混在しても、返された候補は通常どおり1件ずつ予約前照合へ進める。
候補のschema_version=1と既存キーは変えず、優先順の値は延期による欠番を詰めない。
`input_sha256`は来歴用の固定全体digestで、最新の全体digestとの一致を要求しない。
鮮度は`target_sha256`に対応する引用資料を含む対象閉包で検査する。候補のsourcesは公開URL形式であり、
`assert_candidate_fresh`はagentの参照形式と同じ資料へ解決して再取得する。引用資料を含まない旧候補のdigestは流用せず再判断する。
候補1件は以下の形（追加する場合は候補のversionを上げる）。

```json
{
  "schema_version": 1,
  "repo": "owner/repo",
  "parent": 123,
  "route": "standalone",
  "priority": 1,
  "basis_sha256": "record_autonomousの戻り値",
  "generation": 1,
  "input_sha256": "全体入力のdigest",
  "target_sha256": "対象閉包のmaterialのdigest",
  "code_commit": "固定したgit commit",
  "requirements": {"purpose": "目的", "scope": "対象と対象外", "behavior": "期待動作", "acceptance": "完了条件と検証方法"},
  "dependencies": [],
  "rationale": "着手可否と優先順位の理由",
  "sources": ["https://github.com/owner/repo/issues/123", "code:COMMIT:PATH"]
}
```

routeはstandaloneまたはparent。parentは正式な最上位親（単独なら自身）で、子を新しい親として返さない。
basis_sha256/generationはmodelの自己申告ではなく `record_autonomous()` の戻り値。
人間合意はこの候補に含まれない。

割当側は保存した `input.json` / `result.json` を読み、**既存Store.locked()の同じ区間内で**順に実行する:

1. `dispatcher.judgement.assert_candidate_fresh(gh, store, root, snapshot, candidate)`。
   外部依存・コード・正式関係・担当を再取得する。入力が変われば候補を破棄し、次の予定回または明示再判断へ戻す。
2. `parent.execution.validate_candidate(session, candidate['basis_sha256'], candidate['generation'])`。
   最新根拠と親本文/指示・担当不在・次generationを検査する。
3. 返るauthorizationを保持し、既存registryへ予約する。ここから先の予約/実起動/runtime結合/復旧は割当側の責務。

実行根拠のcontextは `dispatcher: {schema_version:1, input_sha256, target_sha256, code_commit}`。
contextのdigestだけでは外部資料の鮮度を保証しないため、validate_candidateだけで第1段階を省略しない。
親workflowも依存・採用状態などの再取得を続ける。コードcommitが変わった候補は自動的に付け替えず再判断する。

## 採用レポート提案のIssue化側への入口

triageは生成・再利用した**通常の作業issueをGitHubへ反映するだけ**でよい。専用のenqueue、readyラベル、
手動の親指定や親セッション登録は収集の前提にしない。次の既定予定回で全open issueとして収集する。
共通issues APIの目的・範囲・完了条件・先行依存・正式親子関係・提案出典を保持する。
`ops-report` ラベルおよび本文先頭の `ops-report:v*` は報告容器専用とし、生成する作業issueへコピーしない。
`ops-report-proposal` 出典メタデータを持つ作業issueは報告容器ではなく、通常の着手判定へ進む。
共通 `parse_proposal_source()` のreport番号と `parse_task_metadata()` のcontextにある全bindingのreport番号を、
重複を除いて `source_reports` へ収集する。contextだけを持つ親・分割子でも、索引anchorの正式親が変わった後も、
出典レポートを分割入力とsubjectの対象閉包へ含める。採用状態の変更は関連subjectをv2の`deferred`へ移し、独立候補を維持する。
複数提案の対応・採用取消・復旧とparentの提案受領/合意の旧契約との接続はIssue化側の責務。
不正・未知versionのtaskメタデータは収集エラーにし、空の出典や旧経路へ置き換えない。
本入口から整理agent・割当判定agent・開発親を追加起動する通知経路は作らない。
採用を人間合意に読み替えない。

## 検証と未検証範囲

```bash
.venv/bin/python -m pytest tests/e2e/agents_ops/test_dispatcher_judgement.py -n 4 -q
```

fake GitHub・fake Claude CLI・一時registry・使い捨てgitを使う。既存の統合試験は本番と同じ共通ランナーを通す。
入力分割と引数・契約照合の試験にはfake runnerも使う。統合試験は共通ランナーの`--read-dir`対応が前提であり、
未対応の引数を除去したりskipしたりせず、対応版を取り込んで同じテスト全体を再実行する。
未合意の具体的な作業、情報不足、正式親子、単独、別親依存、取消依存、既存PR/担当、明示停止、報告容器、
100件超のページング、不正出力、API失敗、無変更再実行、自己コメント、記録直前・途中・全記録後の変更を検証する。
無関係なissue/PR/担当の更新、他subjectだけの変更、対象閉包だけの変更、全件延期、新規open issueと正式な子の追加、
担当/委任予約の変更でも、影響のない判断・候補を保持し、previewにも同じ延期の内訳を返すことを確認する。
巨大な本文/コメント・100件超のPRでは、原文の完全な復元・関連PR限定・省略の明示・digest維持を検証する。
本物のagent/GitHub書込/担当予約/systemdはこの試験で起動しない。
実agentによる代表例の意味的評価、実issueへの保留コメント、予定回・予約・起動を含む通し検証は実機検証担当へ引き継ぐ。

## 半日周期とsupervisor

`python -m dispatcher` はschedule・割当・runtime管理の入口。systemd用の同じ入口は `dispatch.py`。
`service.py` は判定agentの予定実行、`schedule.py` はAsia/Tokyoのslotとrun journal、
`supervisor.py` はregistry内の担当予約・イベント・実行枠、`worker.py` / `runtime.py` は共通ランナーへの接続を担当する。
`state.py` は同じparent registryを読み、欠落時の再初期化を防ぐ。担当台帳は増やさない。
`outcomes.py` は割当・controller復旧・表示で共通のv2結果検査を行う。
上の `judgement.py` の明示入口は維持する。日程や実行枠を判定promptへ持ち込まない。

毎日08:00・20:00 JSTに、正常に収集できた入力についてread-onlyの判定agentを1回呼ぶ。
空入力・無変更でも起動する。issue更新・子の終了・supervisorの再起動は判定agentを起動しない。
`run` は最新の未消費slotだけを消費し、PC停止中の複数slotを再生しない。
同じslotの重複通知はnot_due、前のcontrollerがlockを保持する間はbusy。
前のcontrollerを失っても、runtimeの終了・回収が未確認なら次のrunを始めない。
失敗したslotも自動再試行せず、次の予定回または `run --explicit` で再判断する。

schedule/runの正本は `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/dispatcher/schedule.json`。
初期化はdisabledで、初期化時点の直前slotまでを消費済みにする。過去の予定を導入直後に実行しない。
`--state-dir` と `--parent-state-dir` は使い捨て試験用にも指定できる。
scheduleもregistryと同じ `Store.locked()` 内で保存し、長い判定の重複は別の `judgement.lock`、
supervisorの重複は `supervisor.lock` で防ぐ。JSON破損・未知version・不正設定は停止し、初期化し直さない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher init \
  --repo Motoki0705/tennis-lab --cwd "$PWD" --max-parents 1 --max-children 2
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher status
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher enable
# 通常はtimerがrun、常駐supervisorがsuperviseを呼ぶ
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher run
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher supervise --once
# 明示的な再判断。disabled中や終了未確認runがある場合は実行しない
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher run --explicit
# 今後の割当の設定。既存親の子上限・実行設定は黙って変更しない
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher configure --max-parents 2 --max-children 2
```

`status` はenabled、直近/次回/due slot、判定の結果・失敗、候補のstale/既存担当/枠待ち、
親の担当状態と実行枠の使用、子の終了未確認予約、各jobのphaseを返す。
`run` の返却値と `status.judgements[].deferred` は延期subjectのissue/stage/reasonを返す。
内訳の正本は各判定の `result.json` だけで、scheduleへ複製しない。結果未保存時のstatusはdeferred=null。
recorded/deferred/preview/failedはいずれも終了・回収を確認してcontroller中断から復旧できる。
予約へ進むのは前述のv2候補契約を満たすrecordedだけ。部分延期の候補は優先順の欠番を保ち、延期subjectは次回へ回す。
run journalに候補の待機理由を残すが、担当の正本はregistryだけである。

## 担当・実行枠・中断復旧

registry外側のversionは1を維持し、既存の親行にversion付き `dispatch` を追加する。
そこにassignment ID、候補、authorization、phase、child_limit、runtime jobs、順序付きeventsを保存する。
既存の手動担当はreleased/endedを含め自動移管しない。既存行・task・reviewを削除しない。
候補を予約する直前に、前述の `assert_candidate_fresh` → `validate_candidate` → 予約を同じlock内で行う。
候補の変更はstaleとして保留し、agentを予定外に再起動しない。空いた枠へ保存済みの有効候補を優先順に渡す。

初期予約はregistryのactive行に `session=dispatcher-reservation:<random token>` として保存する。
これは起動権の予約であり、runtimeのsession identityではない。
worker自身が同じlockでreserved→startingを取得するため、Popen直後の応答喪失や複数launcherでも実runnerは1つだけ。
取得時も保存したauthorization・担当generation・assignment ID・現在の委任IDと親/子の停止状態を検査し、
parentの実行根拠検査、子の要件digest・正式関係と既存launch_planによる起動条件を再照合する。
各jobのprivateな`start-input.json`とregistry内のdigestを使い、引用資料・依存・PR・コード等の対象閉包も
同じlock内で再取得する。初回親は判定時の入力を使い、再開親・子は予約時点の入力を固定する。
自分のregistry行の変化は上記の担当・委任検査で扱い、他の担当と外部入力の変更は鮮度照合に残す。
不一致は理由と`runtime_cancelled`イベントを残して未起動予約を解放し、予定外に再判断したり別候補を起動したりしない。
初回親の未受理予約だけはreleasedとし、次の予定回または明示判定で得た新しい候補から再予約できる。
その場合も以前のdispatchは`dispatch_history`に保持する。受理済み親・手動担当・明示停止の担当は自動移管しない。
起動証拠がある予約は解放せずrecovery_requiredで枠を保持する。保存入力の欠落・改変を新しい入力で補わない。
workerのlifetime lockはrunnerへFD継承する。workerが失われても起動途中のrunnerを見落として再起動しない。
共通ランナーは親を `--role parent --mode write` で起動し、モデル値の正本は共通設定のまま使う。
担当受理時だけ、観測した生存Claude runtimeとCLI環境のidentityを照合し、同じgenerationで予約に結合する。

初期の開発親上限は1、各親の実装runtime上限は2。これは手動親の `launch_policy` の既定nullとは別である。
自律親だけ予約上限も2に揃え、実runtimeの上限はさらに `start-child` で検査する。
子の選択・依存・予約上限は [parentのlaunch_plan](../parent/README.md#並列起動の選択と予約上限) を使う。
dispatcherは起動選択を再実装せず、既存の委任予約を受け取って実runtimeと終了証拠を照合する。
子の予約/停止/closeや親の待機だけでは枠を解放しない。taskの終了証拠と、過去の全runtime jobsを両方数える。
再委任でdelegation_idやworker_endedを変えても、生存/終了未確認の旧jobを消せない。
終了は共通ランナーのterminal結果・cleanup_complete・runner/CLI/group不在とworker lifetime lockの解放で確認する。
確認後に現在のdelegation_idへだけparentの `worker_ended` 形式の証拠を記録する。
子の結果は親へ終了イベントとして戻す。exit 0や子の最終文を成果物の合格に読み替えない。

親のwaiting/failed/stoppedは担当activeを保持する。親runtimeも子も終了済みなら実行枠は空く。
子終了イベントまたは明示resumeは既存親を再開待ちにし、枠とauthorizationを再確認して、最新runから同じsessionをresumeする。
CI/人間判断待ちの解除はworkflow側の明示通知を使う。時間経過から解除しない。
completedは子の終了確認後にだけ担当endedへ移す。GitHubのmerge/closeと受入条件の検証は親workflowの責務。
単独issueのmerge/close後に中断した親の復帰条件は [開発workflowの終了照合](../parent/development_workflow.md#待機完了担当終了) に従う。

```bash
# 自動の新規判定・割当・起動を無効化（既存runtimeは継続）
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher disable
# 判定と管理対象の全runtimeへ停止要求。担当・履歴は保持
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher shutdown
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher stop --parent 123
# 状態を照合し、同じsessionで戻す。既存子の生存は枠に残る
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher resume --parent 123 --reason-file tmp/resume.md
```

停電・SIGKILL後は `status`、registry内のjob、privateなworker.log/runner.stderr/runtime記録を照合する。
`supervise --once` は既存結果を照合する。失われたrunnerや残存CLIは `stop --parent` が共通ランナーのstopで照合する。
workerが生存中なら終了後にもう一度superviseで確認する。launcherが返した終了値だけで枠を解放しない。
予約保存後/実起動前なら同じ予約を取得するworkerへ戻る。起動後の応答喪失は保存済みrunへ結合する。
実起動前の停止は、worker identity・runner終了値・runtimeディレクトリ・runner出力がないことを確認して
`cancelled_before_start=true`の取消予約として残す。再開予約を取り消した場合、次のresumeは保存した
`resume_from`を辿って実行済みrunの同じsessionへ戻る。取消予約に起動証拠がある場合や、再開元が
一意に照合できない場合は再開を拒否し、実プロセスと記録を照合する。実行済みrunの結果欠落もこの取消扱いにしない。
runtime記録欠落・identity不明・回収未確認はrecovery_requiredで枠を保持する。
sessionが取れなかったrunから別sessionを自動生成しない。このケースは実プロセスと成果物を人間が照合する。
手動handoffで稼働中のregistry identityを書き換えない。旧workerの終了確認を先に行う。
破損stateや起動記録の復元では原本を退避し、実プロセス・runtimeとの照合結果を残す。lockファイルを削除して排他を回避しない。

### registry欠落からの復旧

初回initはscheduleとjudgements/workersの記録がない場合だけ、欠落したregistryを初期状態として扱う。
registryを先に保存してからdisabledのscheduleを作る。初期状態の空registryや、まだ一度も担当を予約していない
判定履歴だけがある空registryは正常である。scheduleまたはrun/worker記録がある状態でregistryが欠落した場合、
あるいはworker/担当予約の履歴があるのにregistryが空の場合は、初回起動とみなさない。
statusを含むdispatcherの全操作とworker取得をエラーで止め、registry・scheduleを保存せず、判定・割当・起動もしない。

このエラー時はtimerとsupervisorへの新規要求を外側から止め、欠落前のregistry・schedule・privateなworkers/judgements記録を
退避する。生存中の親・子・runnerと各runtime identity、担当generation・委任ID・予約を照合し、確認できるバックアップから
registryを復元する。正常に戻るまで実行枠が空いたと解釈しない。復元できない場合は人間が実プロセスと成果物を照合し、
安全な停止と担当の復旧方法を決める。空registryの作成、schedule/run/lockの削除、initのやり直しで回避しない。
復元後はstatusと`supervise --once`で既存runへ結合されることを確認してから、有効化・再開する。

判定controllerの中断は、保存済みresultとruntimeを照合できれば再利用する。
欠落時は実プロセス不在を確認した証拠ファイルで `recover-judgement --evidence-file tmp/recovery.md` を明示実行する。
稼働中controllerや未回収CLIがあれば拒否する。復旧後も勝手に再判断せず、次回または `run --explicit` を使う。
runner喪失時は、先に [共通ランナーの子孫回収確認](../lib/README.md#状態確認停止中断からの復旧) を行う。
`recover-judgement --evidence-file` だけでは、その `cleanup_confirmation` を代替しない。

## 親workflowへのbootstrap契約

`parent_bootstrap.md` は担当受理・既存状態照合から [親の開発workflow](../parent/development_workflow.md) へ接続する。
実装・検証・親による明示レビュー・mergeと実closeを照合し、成功・待機・失敗を最終イベントに分ける。
各jobのprivateな `input.json` は追加キーなしのschema_version=1で、以下を渡す。

- repo、parent、assignment_id、job_id、候補JSON、authorization。
- parent_state_dir、dispatcher_state_dir、runtime_dir、max_implementation_children。

親は `accept --input <path>` を自身のCLI環境から呼ぶ。引数でsession IDを渡さない。
`AGENT_RUNTIME_RUN_DIR/TASK_ID`、観測session、実行契約、候補/予約/generation/根拠が一致する場合だけ受理する。
同じsessionのresumeでも同じ入口を使う。parent CLIには `--state-dir` で入力と同じregistryを指定し、
review/mergeも同じstateを使う。worker操作のPython APIでは同じregistryを指す `DispatchStore` を渡す。
親workflowは外部依存・採用状態・正式関係を再取得し、委任・レビュー・merge前も根拠を再照合する。

子の本文・worktree・モデル選定やdelegateはworkflowが行う。
親workflowはparentの `launch_plan` から起動可能な子を選び、待機理由・stacked_onを引き継ぐ。
delegateで予約した後、受理済みの親環境から次を呼ぶ。起動自体はsupervisorが行い、子を直接spawnしない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher start-child \
  --parent 123 --child 124 --delegation-id <delegateが返すID> --prompt-file tmp/child.md
# Astraを明示選択するときだけ --model gpt-6-astra を加える
```

モデル/effortの既定は共通ランナーを使う。既存子は終了照合後の証拠付き再delegateで新しいIDを得てからstart-childする。
このとき以前の子の最新runから同じsessionをresumeする。モデル・worktree・branch・設定の不一致は共通ランナーが拒否する。
起動前取消の修正予約は、保存した再開元runへ戻す。一度も起動していない取消予約は、
全jobの起動証拠がなく既存PRもない場合だけ、同じ担当で明示的に再予約する。
実行済みrunのsession欠落や取消記録の曖昧さを、別sessionへの振替で回避しない。
pytestは [並列実行の手順](../parent/README.md#pytestの並列実行) に従い、共有lockなしで親・子間の並列実行を許可する。
GPU・Blenderの資源調停はtraining queueの既存枠を使う。

親の最終文は追加キーなしのJSON:

```json
{"schema_version":1,"state":"waiting","reason":"CI待ち","resume_condition":"CIが完了したら明示resume"}
```

stateはwaiting/completed/failed。finalの不正・担当未受理・runner失敗はfailed。
子の生存中はcompletedを拒否する。待機理由と再開条件・イベントはregistryに保持する。
判定・予約・supervisorはレビューを直接起動しない。

## systemdへの導入・有効化・停止

merge後にmain checkoutで人間が実施する。worktreeからinstallerを実行すると、リンク作成前に拒否する。
installerの既定はunitリンクとdaemon-reloadだけで、disabled状態から導入できる。
`--enable` は全timerとdispatch-supervisorを有効化するため、対象だけ有効化する以下の手順を通常使う。

```bash
cd ~/projects/tennis-lab
.agents/ops/install_units.sh --dry-run
.agents/ops/install_units.sh
systemd-analyze calendar '*-*-* 08:00:00 Asia/Tokyo' '*-*-* 20:00:00 Asia/Tokyo'
systemd-analyze --user verify .agents/ops/dispatcher/systemd/*
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher init --repo Motoki0705/tennis-lab --cwd "$PWD"
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher enable
systemctl --user enable --now tennis-lab-dispatch-supervisor.service tennis-lab-dispatch-judgement.timer
systemctl --user cat tennis-lab-dispatch-judgement.timer tennis-lab-dispatch-supervisor.service
systemctl --user list-timers --all 'tennis-lab-dispatch-*'
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher status
journalctl --user -u tennis-lab-dispatch-judgement.service -u tennis-lab-dispatch-supervisor.service

# 停止。timerのdisableだけでは実行中のserviceは停止しない
systemctl --user disable --now tennis-lab-dispatch-judgement.timer
PYTHONPATH=.agents/ops .venv/bin/python -m dispatcher shutdown
systemctl --user stop tennis-lab-dispatch-judgement.service
systemctl --user disable --now tennis-lab-dispatch-supervisor.service
```

再開時はinitを繰り返さず、status/実プロセスを照合してenableし、同じ2unitを有効化する。
timerはPersistent=trueで復帰時を1回へ集約する。ランダム遅延はなく、精度は1秒。
supervisorは1秒ごとにruntime/イベントを照合するが、判定agentを呼ばない。
systemdのKillMode=control-groupは停止時の離脱子孫も対象にする。複数ホストの排他は行わない。

検証は `test_dispatcher_supervisor.py`。fake GitHub/時計/CLI、一時registry、使い捨てgitとfake systemctlで行う。
実エージェント・GitHub書込・ホストunit導入は実機検証へ引き継ぐ。
