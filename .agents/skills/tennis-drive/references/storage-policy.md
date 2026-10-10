# Driveの資産・成果物管理方針

この文書を、tennis-labのGoogle Driveにおける配置、版管理、整理、移行、保持の正本とする。
特定の実験やColabだけに閉じない。タスクやデータが増えたときは必要な規約をここへ追加し、
呼び出すスキル・READMEにはリンクを置く。

## 役割と実際の保存場所

ローカルのroot・相対path契約は [タスク出力規約](../../../../src/tasks/OUTPUTS.md) を参照する。
Driveもその役割に対応させる。rootの既定値はCLIが所有し、タスク名・実験名・データ版を
スクリプト内の列挙値にしない。

| 役割 | Drive root相対の出発点 | 内容 |
|---|---|---|
| 入力データ | `data/<data-root相対path>` | 配布データ、生成dataset、付属manifest・split |
| 採用・事前学習重み | `ckpt/<checkpoint-root相対path>` | 学習や推論で明示的に選ぶ入力weight |
| 実験出力 | `outputs/<task>/<purpose>/<experiment>/<run-id>` | 学習・評価・可視化の設定、ログ、checkpoint、結果 |
| 配置・採用前の候補 | `staging/<operation-or-asset-id>` | 検証・移行中のコピーや候補資産 |
| 遠隔実行の運用記録 | `operations/<executor>/<session>/<job-id>` | コマンド、接続・実行状態、stdout、保存receipt。実験成果物は上記outputsへ |

これらは保存先を解決するための規約であり、未知の資産を無理に分類する固定schemaではない。
新しい種類が必要なら、所有するタスク、消費側、寿命、既存の役割との差を示して規約を拡張する。
日付・Issue番号だけを使う今回限りの最上位階層は増やさない。
運用記録のexecutorにはColab等の実行先を使い、taskやexperimentの内容を埋め込まない。
session終了後も障害調査・成果物の由来のため記録を保持し、credentialは含めない。

通常の入力はDriveとローカルでrepository相対pathを対応させる。外部ライブラリが特殊な
pathを要求する場合はstage先を明示する。rootやpathにcredentialを含めない。
Drive上の同名項目は許されるため、pathだけを一意な識別子と考えない。

## 入力資産の版と由来

採用したdataset/checkpointの内容を黙って差し替えない。内容・split・座標系・教師・
生成条件が変わる場合は、新しい版または新しい実験出力として保存する。

既存の `dataset.json`、checkpoint設定、run manifestが持つ情報を正本として使う。
同じ件数・hash・split一覧を別の台帳へ手作業で複製しない。必要なら元manifestへの参照と
digestを記録する。少なくとも次を追跡できるようにする。

- 資産またはrunの識別子、種類、版、実際のDrive IDとroot相対path。
- 出自（配布元、生成run、Gitの版）と、参照するmanifest/config。
- 内容のdigest、ファイル数・サイズ、確認日時。同名や同サイズだけを内容一致としない。
- 読み取り用の確定資産か、書き込み中・候補・中断runか。

既存manifestで足りない情報だけを、その資産に付随する小さなmetadataへ追加する。
独自metadataにはschemaの版を持たせ、未知の版を旧版として解釈しない。
全タスク共通の巨大な台帳やdatabaseを、単発の追加だけを理由に作らない。

`inventory` はある時点の観測であり、dataset宣言やrunの成否を上書きする正本ではない。
必要なsubtreeに範囲を絞り、新しいファイルへ記録する。ID/hashと曖昧なpathも残す。

## 実行中と確定済みの成果物

checkpointは生成後にDriveへ保存し、ログ・設定なども周期的に保存する。
まずDriveへの接続・書き込み先・空き容量を確認する。保存失敗をVM内のみの成功として処理しない。
共通training runner / ArtifactStoreを使い、モデル本体へ保存責務を持ち込まない。

実行ごとにrun IDとGitの版、入力dataset/checkpoint、解決済み設定、環境、状態を記録する。
途中runと完了runを、ディレクトリの存在だけで判定しない。
同じ設定のcheckpoint再開はrun/attemptの関連を記録し、コードや施策の変更は新しいrunとして区別する。

任意のPython/shell処理は、成果物が出るローカルpathとDrive保存先を明示する。
checkpointを生成しないプログラムに、プロセスそのものを復元できると約束しない。
Colab VM消失後の学習再開には、Drive上の完了したcheckpointと対応する設定を使う。

成果物は依頼に応じてローカルへ回収する。採用checkpointを `ckpt/` へ置く場合は
選定理由と由来runを残し、元の学習成果物を移動して消さない。

## 整理と配置変更

入力を消費する設定・実行中の書き込み・再現記録を確認してから整理する。
標準的な移行は次の順とする。

1. 対象subtreeをinventoryし、ID・内容・参照元・不足を確認する。
2. 移行先と命名、影響する参照、必要容量を示す。未知の分類はこの段階で相談する。
3. 新しい場所へcopyし、内容を検証する。既存の移行先へ曖昧に結合しない。
4. 消費側の設定・manifest参照を切り替え、読み取りを確認する。必要な移行記録を残す。
5. 元の資産を削除する場合は、明示された対象だけtrashへ移す。

単純な改名・移動で参照更新が不要と分かっている場合は、検証付き `move` を使える。
途中失敗はsource/destination双方を確認する。同じ操作を盲目的に繰り返さない。
同名folderが複数ある場合はIDと由来を提示して正本を決める。自動dedupeで内容を統合しない。

## 保持と削除

入力資産・確定済み結果・中断runを、容量確保のために自動削除しない。
学習frameworkのcheckpoint間引きがDrive側の削除へ伝播する設定も、明示依頼なしに使わない。
共通training runnerで新しいColab runを作る場合は`training.checkpoint.save_top_k=-1`として
保存済みの世代を保持し、最良checkpointの選択と削除を分離する。`last.ckpt`の更新は
再開用の可変ファイルとして扱い、採用した世代別checkpointを置き換えない。
大きいもの・重複候補・参照されなくなった候補を可視化し、依頼された対象だけtrashへ移す。
ゴミ箱内のファイルも容量を使う。trash操作を容量解放完了と報告しない。
ゴミ箱全体の空化・永久削除・共有設定変更は通常操作に含めない。

将来保持数や保存期間を自動化する場合は、対象role、保護条件、参照検査、候補表示、
復旧方法を明記してこの方針を拡張する。学習frameworkのcheckpoint保持設定も記録する。

## 既存配置の読み取り

`colab-live/`、`colab-runs/`、`colab-interrupted/`、`training_queue/` 等の既存保存物は
作成時のmanifestと読取経路を保持する。新規run開始時に既存treeを自動移動しない。
旧Colab bundleのrequest/status/manifestとarchive digestは当時の証拠として扱う。
過去knowledgeのコマンドやhashを現行CLI名へ機械的に書き換えない。
必要な旧成果物は明示pathから取得・検証し、移行は上の手順で別途行う。
