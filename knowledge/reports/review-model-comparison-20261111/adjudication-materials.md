# 正解論点別の親判定と正解表外の分類

親が全14 run・17指摘を判定した。確定判定は[adjudications.json](adjudications.json)のjudgmentsを正本とし、[scores.json](scores.json)へ集約した。workerの初期対応候補は[worker-notes.json](worker-notes.json)に履歴として保持している。判定内容はworkerが変更していない。

## pr-1699

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1699-astra-20261111/output.json)。

- モデル指摘`pr-1699-1`: **正解一致 pr-1699-1**。親の理由: URLスキームの大小文字でuserinfo検査を回避できるという同じ論点・同じ箇所（safety.py:21）。
  発生条件: 大文字のHTTPSスキームとuserinfoを持つURLが、既知private値に含まれない場合。
  処理の主張: URL抽出はcase-sensitiveだが、絶対パス検査はcase-insensitiveでHTTP(S)を除外する、というモデルの指摘。
  影響の主張: 指示やfinal内の資格情報付きURLが拒否されずGit公開される、という主張。
  修正対象: `.agents/ops/worker_logs/safety.py:21`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1699-2`: **valid_new**。親の理由: dispatcherの自律経路では、作成済みの正当なログPRがhtml_urlの大小文字差でpr_identity_changedになる。静的追跡で成立を確認した（実GitHubでの再現はしていない）。
  発生条件: dispatcherがrepo名を小文字化し、GitHubが正規表記のPR URLを返す場合。
  処理の主張: repo比較はcasefoldだがhtml_urlは入力repoとの完全一致を要求する、というモデルの指摘。
  影響の主張: 正当なログPRがpr_identity_changedで拒否され、自動mergeまで進まない、という主張。
  修正対象: `.agents/ops/worker_logs/schema.py:235`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1699-3`: **false**。親の理由: merge未確認の後に自動で再mergeしないのは、READMEに書かれたfail-closedの契約である。人間がPRを手動でmergeすれば、次のpublishが merged=True の分岐でmainを照合して完了するため、ログが恒久的に保存できなくなるわけではない。
  発生条件: PUT送信前の記録後に通信障害やAPI拒否が起き、PRが未merge・openのままの場合。
  処理の主張: merge_attemptedが永続化され、その後の再送をmerge_response_uncertainで拒否する、というモデルの指摘。
  影響の主張: 障害解消後も同じログPRをmainへ保存できない、という主張。
  修正対象: `.agents/ops/worker_logs/publication.py:435`。検証の根拠は親判定のevidenceを参照。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1699-sol-20261111/output.json)。

- モデル指摘`pr-1699-1`: **正解一致 pr-1699-1**。親の理由: URLスキームの大小文字でuserinfo検査を回避できるという同じ論点・同じ箇所（safety.py:21）。
  発生条件: 大文字HTTPSとuserinfoのURLが既知private値に含まれない場合。
  処理の主張: URL正規表現と絶対パス検査で大小文字の扱いが異なる、というモデルの指摘。
  影響の主張: 認証情報付きURLがinstruction/finalから公開・自動mergeへ通る、という主張。
  修正対象: `.agents/ops/worker_logs/safety.py:21`。検証の根拠は親判定のevidenceを参照。

両モデルの見逃し: `pr-1699-2`。

## pr-1766

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1766-astra-20261111/output.json)。

- モデル指摘`pr-1766-1`: **正解一致 pr-1766-1**。親の理由: 自動継続の許可が取消（cancelled）を上書きしうるという同じ論点・同じ箇所（recovery.py:80）。
  発生条件: 継続判断の保存とauthorize_retryの間に、同じattemptのstopがcancelledを確定する場合。
  処理の主張: work lock解放後のauthorize_retryがcancelledも受理しreadyへ上書きする、というモデルの指摘。
  影響の主張: 取消済みcameraが次tickで自動再起動する、という主張。
  修正対象: `src/tasks/annotation/recovery.py:80`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1766-2`: **valid_new**。親の理由: goldとは別の論点。保存ファイルの破損でholdを一度記録すると、ファイルを修復しても同じattemptのholdが再利用されて完成receiptが作られない。RetryWorkerも完了済みを理由に拒否し、recover APIに完成を確定する操作がない。静的追跡で成立を確認した（再現テストは未実行）。
  発生条件: 全frame保存後のtimeoutで欠損・破損によりholdを保存し、その後ファイルを修復した場合。
  処理の主張: cycleが不変holdを再利用し、RetryWorkerも全frame処理済みを理由に拒否する、というモデルの指摘。
  影響の主張: 保存修復後も正式recover APIで完成receiptを確定できない、という主張。
  修正対象: `src/tasks/annotation/recovery.py:410`。検証の根拠は親判定のevidenceを参照。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1766-sol-20261111/output.json)。

- モデル指摘`pr-1766-1`: **正解一致 pr-1766-1**。親の理由: 自動継続の許可が取消（cancelled）を上書きしうるという同じ論点・同じ箇所（recovery.py:80）。
  発生条件: timeoutと競合するpool.stopが、判断保存後のロック解放中に同じattemptをcancelledへ確定する場合。
  処理の主張: authorize_retryがattempt一致だけでcancelledをreadyへ戻す、というモデルの指摘。
  影響の主張: 明示retry待ちの要件に反し、取消済みcameraが自動起動する、という主張。
  修正対象: `src/tasks/annotation/recovery.py:80`。検証の根拠は親判定のevidenceを参照。

## pr-1867

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1867-astra-20261111/output.json)。

- モデル指摘`pr-1867-1`: **正解一致 pr-1867-1**。親の理由: A承認→B major→CでAを採用し、Bの未解決指摘を隠してgateを通れるという同じシナリオ。
  発生条件: 同一snapshotのAにapprove finalがあり、後続Bが重大指摘を公開した後にCでAを採用する場合。
  処理の主張: recoverは元runとsnapshotだけを確認し、gateは最新Cを選ぶ、というモデルの指摘。
  影響の主張: Bの指摘をresolveせず親確認とCIだけでgateを通せる、という主張。
  修正対象: `.agents/ops/review/parent_review.py:769`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1867-2`: **unconfirmed**。親の理由: gold pr-1867-2（完了イベントとfinalの内容不一致を承認へ変えない）とは別の論点。非ゼロ終了で有効finalを持つrunを、公開中の強制終了後にAI起動なしで採用できないという指摘だが、gold側の修正はrunnerの失敗を証拠として厳格に拒否する方向であり、失敗runのfinalを採用可能にすべきかは契約上決まっていない。影響は追加のAI起動であり、誤ったmergeではない。実行による再現もしていないため、確認不能とする。
  発生条件: 非ゼロ終了runの検証済み候補を保存後、失敗結果の公開中に本体が停止しexecutionがrunningのままの場合。
  処理の主張: recoverはrunningを拒否し、resume-publicationはoutputと失敗resultの一致要求で拒否する、というモデルの指摘。
  影響の主張: 終了とcleanupを確認できてもAI起動0回の復旧経路が無くなる、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:109`。検証の根拠は親判定のevidenceを参照。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1867-sol-20261111/output.json)。

- モデル指摘`pr-1867-1`: **正解一致 pr-1867-1**。親の理由: 指摘箇所はmerge_gate.py:292だが、R1承認→R2 major→R3でR1を採用し、R2の指摘をresolveせずgateを通れるという、gold pr-1867-1と同じシナリオ・同じ修正対象（採用時とgateでの後続run照合）。
  発生条件: 同一snapshotのR1にapprove finalがあり、後続R2がmajorを報告した後にR3がR1をrecoverする場合。
  処理の主張: gateは最新R3を選び、verify_adoptionはR1との整合だけで後続R2を照合しない、というモデルの指摘。
  影響の主張: R2の指摘をresolveせず親確認とCI成功があればgateを通れる、という主張。
  修正対象: `.agents/ops/review/merge_gate.py:292`。検証の根拠は親判定のevidenceを参照。

両モデルの見逃し: `pr-1867-2`。

## pr-1872

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1872-astra-20261111/output.json)。

- モデル指摘`pr-1872-1`: **正解一致 pr-1872-1**。親の理由: 公開状態（published・publication_failed）から終了・cleanupを推定し、同PRの別runを許すという同じ論点・同じ箇所（review_locks.py:78）。
  発生条件: agent-runtimeがSIGKILLされ、別sessionで動くCLIが残っているのに失敗結果の公開まで進んだ場合。
  処理の主張: 排他検査はexecution.statusのrunningだけを拒否し、終了・cleanup証拠を見ない、というモデルの指摘。
  影響の主張: cleanup未確認の旧CLIと同じPRの新runが並行起動できる、という主張。
  修正対象: `.agents/ops/review/review_locks.py:78`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1872-2`: **正解一致 pr-1872-2**。親の理由: runtime記録のない中断runをclose-interruptedで終結できず、残ったrunningが同PRを永久に止めるという同じ論点。gold の契機は起動境界、この指摘の契機はAI未起動のadoptionで経路が異なるため、緩い一致として扱う。
  発生条件: AI未起動のrecover adoptionがrunningを保存して公開前に中断し、後でhead/baseが進んだ場合。
  処理の主張: close-interruptedが未起動runを起動済みと扱い、作成されていないruntime resultを必須にする、というモデルの指摘。
  影響の主張: snapshot更新後はrecover再送と公開再開もできず、同PRの新runを恒久的に止める、という主張。
  修正対象: `.agents/ops/review/review_interruption.py:23`。検証の根拠は親判定のevidenceを参照。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1872-sol-20261111/output.json)。

- モデル指摘`pr-1872-1`: **正解一致 pr-1872-1**。親の理由: 公開状態（published・publication_failed）から終了・cleanupを推定し、同PRの別runを許すという同じ論点・同じ箇所（review_locks.py:78）。
  発生条件: 共通runnerだけが強制終了して旧CLIが残り、agent工程unconfirmedでも公開処理でexecutionがpublishedまたはpublication_failedになった場合。
  処理の主張: running以外を通す排他検査が、起動済みruntimeの終了とcleanupを確認しない、というモデルの指摘。
  影響の主張: lease解放後に同PRの別runが新agentを並行起動できる、という主張。
  修正対象: `.agents/ops/review/review_locks.py:78`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1872-2`: **正解一致 pr-1872-2**。親の理由: runtime記録のない中断runをclose-interruptedで終結できず、残ったrunningが同PRを永久に止めるという同じ論点。gold の契機は起動境界、この指摘の契機はAI未起動のadoptionで経路が異なるため、緩い一致として扱う。
  発生条件: runtimeを作らないrecover adoptionが公開途中で中断し、その間にhead/baseが進む場合。
  処理の主張: not_invokedも起動済み扱いになり、close-interruptedが存在しないruntimeを要求する、というモデルの指摘。
  影響の主張: snapshot更新で再送と公開再開も拒否され、残存runningが同PRの新runを阻止する、という主張。
  修正対象: `.agents/ops/review/review_interruption.py:23`。検証の根拠は親判定のevidenceを参照。

## pr-1888

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1888-astra-20261111/output.json)。

- モデル指摘`pr-1888-1`: **正解一致 pr-1888-1**。親の理由: 一部PRだけreviewedのfailed・incomplete結果でruntime照合が省かれるという同じ論点・同じ箇所（review_recovery.py:347）。
  発生条件: あるPRがreviewed、別PRがnot_reviewedとなり、run全体がfailedまたはincompleteの場合。
  処理の主張: 全体の成功状態だけでruntime検証を省略する一方、coverageとgateはreviewedのPRを独立採用する、というモデルの指摘。
  影響の主張: 保存後の実効設定の不一致や証拠欠損を照合せず、部分結果のPRを承認へ採用できる、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:347`。検証の根拠は親判定のevidenceを参照。

- モデル指摘`pr-1888-2`: **valid_new**。親の理由: goldとは別の論点。--review-model-source human_instruction と --review-model だけを渡し、--human-review-effort を添えると、人間が指定したeffortが通常のeffort引数（省略時None）で上書きされて消える。許可外のmaxでも拒否されず、fixedのxhighが人間指定として記録・実行される。人間の明示指定を黙って失う静かなフォールバックである。
  発生条件: human_instruction指定でreview-modelをAstra、human-review-effortをmaxとし、human-review-modelを省略して理由と参照を渡した場合。
  処理の主張: human modelが無いためhuman effortを通常effort引数で上書きし、明示maxが消える、というモデルの指摘。
  影響の主張: 拒否すべきeffortを既定xhighに変え、人間指定として記録・実行できる、という主張。
  修正対象: `.agents/ops/lib/review_settings.py:210`。検証の根拠は親判定のevidenceを参照。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1888-sol-20261111/output.json)。

- モデル指摘`pr-1888-1`: **正解一致 pr-1888-1**。親の理由: 一部PRだけreviewedのfailed・incomplete結果でruntime照合が省かれるという同じ論点・同じ箇所（review_recovery.py:347）。
  発生条件: PR Aがreviewed、PR Bがnot_reviewedで、全体がfailedまたはincompleteになる場合。
  処理の主張: 早期returnがruntime検査を省略する一方、coverageとgateはAを採用できる、というモデルの指摘。
  影響の主張: runtime記録の欠落やモデル・effortの不一致、digestや公開設定の不整合を検出せず、公開再開とAのmergeを許す、という主張。
  修正対象: `.agents/ops/review/review_recovery.py:347`。検証の根拠は親判定のevidenceを参照。

両モデルの見逃し: `pr-1888-2`。

## pr-1686

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1686-astra-20261111/output.json)。

指摘0件。暫定負例についての欠陥不存在の証明とは扱わない。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1686-sol-20261111/output.json)。

指摘0件。暫定負例についての欠陥不存在の証明とは扱わない。

## pr-1744

### Astra

[モデル出力](../../../knowledge/runs/run-i1863-pr1744-astra-20261111/output.json)。

指摘0件。暫定負例についての欠陥不存在の証明とは扱わない。

### Sol

[モデル出力](../../../knowledge/runs/run-i1863-pr1744-sol-20261111/output.json)。

指摘0件。暫定負例についての欠陥不存在の証明とは扱わない。

## 緩い一致とスコアの扱い

pr-1872-2は、goldの起動境界とモデルの未起動adoptionで契機が異なるが、runtime記録がない中断runを終結できずrunningが残る論点として親が緩い一致を採用した。両モデルで同じ扱いにしており、比較差は変わらない。判定を厳密一致へworkerが書き換えることはしていない。
