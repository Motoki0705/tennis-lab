# issueの自律着手判断

`judgement` はopen issueを収集し、固定した入力とコードをread-onlyのClaude親に渡す。
構造化結果を検証したスクリプトだけが判断コメントと自律着手根拠を記録する。
正式な子は既存の最上位親へ戻し、単独issueは自身を候補にする。新しい階層や担当は作らない。
親の起動・予約・実行枠・再開・schedule/systemd、実装子への委任、レビュー・mergeはこの入口の責務に含めない。

| ファイル | 責務 |
|---|---|
| `collect.py` | 全ページ収集、正式関係と依存の閉包、registryの読取、固定コード、入力射影・鮮度照合 |
| `agent_input.py` | 完全なsnapshotを変更せず、索引・issue・関連PRを読み取り単位に分割 |
| `agent.py` / `prompt.md` | 共通ランナーでの1回の判定、runtime証拠の照合 / 判断方針と返却schema |
| `schema.py` | 全件の構造化結果・出典・scope・完了条件・routing・機械的阻害条件の検査 |
| `records.py` | 厳格JSON、自分の判断コメントの識別、実行根拠への進捗コメントの除外指定 |
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
ランナーの正常終了、run ID・実行契約・選択値と観測model・プロセス回収を確認した後に最終JSONを読む。
exit 0や`task_success:null`をタスク成功に読み替えない。失敗時に別モデルや別sessionへ振り替えない。

保存物は `input.json`、`agent-input/`、`prompt.md`、`agent-runtime/`、`runner.stdout` / `runner.stderr`、
`agent-output.json`、`result.json`。stdoutは `schema_version/result/candidate_count/diagnostic_count` の短いJSONだけ。
終了値0はpreviewまたは記録完了、2は取得・整合・runtime・鮮度エラー。診断の詳細はprivateなresultを読む。
部分公開後の失敗も `result=failed, candidates=[]`。途中までのコメントや根拠を消さず、次の予定回または明示実行で照合する。

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

`material(snapshot, N)` は対象の親子・依存・提案出典レポートの閉包と関連PR/担当/コードに絞った同じ射影。
書込前は全入力、各書込の直前・直後は対象閉包、全記録後は再び全入力を照合する。
API呼出しとGitHub書込は単一トランザクションではない。競合を検知したrunは候補を返さず、予約直前にも再照合する。
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
自分のコメントの破損・複数一致は上書きせず診断する。本文や指示に編集があればfreshnessで停止する。
自分の表示コメントだけを `ignored_comments` へ明示分類し、通常の指示を除外しない。

startableは `execution_input()` の固定値と最新値を照合し、version付きbasisを生成して `record_autonomous()` に渡す。
人間のconfirmationや共有基盤許可のキーは生成しない。既存の明示取消・checkingを自律判断で解除しない。

## 割当側への候補契約

`result.json` schema_version=1は `result`（preview/recorded/failed）、`input_sha256`、`decisions`、
`publications`、`diagnostics`、`candidates` を持つ。予約に渡せるのは `result=recorded` のcandidatesだけ。
preview/failedのcandidatesは常に空。候補は優先順で、1件は以下の形（追加する場合はversionを上げる）。

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
人間合意や共有基盤許可はこの候補に含まれない。

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
共通 `parse_proposal_source()` から得るreport番号をsource_reportsへ収集し、採用状態の変更も候補の鮮度照合へ含める。
複数提案の対応・採用取消・復旧とparentの提案受領/合意の旧契約との接続はIssue化側の責務。
複数出典を持つ新メタデータを導入する場合は、共通issuesの出典契約とこのsource_reportsの抽出を同じ変更で接続する。
本入口から整理agent・割当判定agent・開発親を追加起動する通知経路は作らない。
採用を人間合意や共有基盤許可に読み替えない。

## 検証と未検証範囲

```bash
flock "$PYTEST_LOCK" .venv/bin/python -m pytest tests/e2e/agents_ops/test_dispatcher_judgement.py -n 4 -q
```

fake GitHub・fake Claude CLI・一時registry・使い捨てgitを使う。既存の統合試験は本番と同じ共通ランナーを通す。
入力分割と引数・契約照合の試験にはfake runnerも使う。統合試験は共通ランナーの`--read-dir`対応が前提であり、
未対応の引数を除去したりskipしたりせず、対応版を取り込んで同じテスト全体を再実行する。
未合意の具体的な作業、情報不足、正式親子、単独、別親依存、取消依存、既存PR/担当、明示停止、報告容器、
100件超のページング、不正出力、API失敗、無変更再実行、自己コメント、記録直前・途中の変更を検証する。
巨大な本文/コメント・100件超のPRでは、原文の完全な復元・関連PR限定・省略の明示・digest維持を検証する。
本物のagent/GitHub書込/担当予約/systemdはこの試験で起動しない。
実agentによる代表例の意味的評価、実issueへの保留コメント、予定回・予約・起動を含む通し検証は実機検証担当へ引き継ぐ。
