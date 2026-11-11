# 固定7ケースの実LLM比較: 実行結果と親の判定資料

2026-11-11、承認済み最小案の14起動を記録し、親が全14 run・17指摘を判定した。追加起動は0回。selectは採用せず、既定のfixed（Astra・xhigh）を維持する。PR #1907の正式レビューで妥当とされたmajor 4件を修正した。再レビューとgateの確認は親の残作業である。

[承認済み規模](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6786941605)と[今回の実行指示](https://github.com/Motoki0705/tennis-lab/issues/1797#issuecomment-6789859748)に従った。7ケース、2モデル各1回、追加枠込み16起動、各1800秒、入力55M・出力0.22Mが承認目安。実際の起動はAstra、Solの順で各ケースを処理し、再試行しなかった。

## Pilotと起動前の再見積もり

| #1699 pilot | 入力 token | うちcache | 出力 token | model秒 |
|---|---:|---:|---:|---:|
| Astra | 1,925,970 | 1,798,528 | 10,465 | 512.448 |
| Sol | 1,586,861 | 1,477,120 | 11,625 | 404.809 |

| 起動許可の見積もり | 入力 token | 出力 token |
|---|---:|---:|
| before_pilot | 49,770,230 | 185,376 |
| after_astra | 49,421,190 | 193,298 |
| after_both_pilots | 34,837,048 | 172,993 |

各見積もりは未使用の追加2枠も含む。pilotのruntime、出力schema、元counter、model/effortの実効証拠が揃い、両pilot後の見積もりが承認目安内だったため残りを起動した。その後も各起動前に再見積もりした。実効service tierはCodex CLIが記録しないため全runで不明のまま保持している。

## 実際に消費した量

物理起動全体: 入力 **18,928,215**、うちcache **17,301,632**、出力 **110,870** token。モデル時間の合計は **4222.006秒**。入力はcacheを含み、出力はreasoningを含む。cacheを入力へ加算していない。14個の元counterを一度ずつ計上し、欠測・非turn counterをゼロ補完していない。

runtime失敗0、形式または出力検証失敗0。code未確認のrunは0/14、evidence waitは0件。モデル出力の指摘は計17件であり、正解の検出数ではない。確認不能率の分母は各モデルが出した全指摘で、Astraは1/11（9.091%）、Solは0/6（0%）。これはモデル出力上のcode未確認やevidence waitとは別の値である。正解blockerは0件であり、blockerの検出力は比較できない。

| case | model | 入力 token | うちcache | 出力 token | model秒 | coverage | 指摘数 |
|---|---|---:|---:|---:|---:|---|---:|
| pr-1699 | Astra | 1,925,970 | 1,798,528 | 10,465 | 512.448 | reviewed | 3 |
| pr-1699 | Sol | 1,586,861 | 1,477,120 | 11,625 | 404.809 | reviewed | 1 |
| pr-1766 | Astra | 1,684,582 | 1,568,000 | 7,546 | 323.246 | reviewed | 2 |
| pr-1766 | Sol | 1,774,189 | 1,657,088 | 10,051 | 389.602 | reviewed | 1 |
| pr-1867 | Astra | 1,358,335 | 1,247,872 | 9,520 | 378.323 | reviewed | 2 |
| pr-1867 | Sol | 1,458,392 | 1,284,864 | 8,936 | 290.402 | reviewed | 1 |
| pr-1872 | Astra | 1,640,423 | 1,507,840 | 8,317 | 313.956 | reviewed | 2 |
| pr-1872 | Sol | 2,042,391 | 1,911,424 | 11,145 | 423.774 | reviewed | 2 |
| pr-1888 | Astra | 1,800,013 | 1,674,368 | 7,221 | 332.067 | reviewed | 2 |
| pr-1888 | Sol | 1,376,872 | 1,244,928 | 7,801 | 260.463 | reviewed | 1 |
| pr-1686 | Astra | 303,633 | 232,704 | 3,344 | 117.894 | reviewed | 0 |
| pr-1686 | Sol | 361,361 | 299,264 | 4,531 | 129.848 | reviewed | 0 |
| pr-1744 | Astra | 635,256 | 541,696 | 4,651 | 161.230 | reviewed | 0 |
| pr-1744 | Sol | 979,937 | 855,936 | 5,717 | 183.944 | reviewed | 0 |

## 親の判定結果

[親の全判定](adjudications.json)をschemaと既存report CLIで検証した。14 runの出力と正解表の照合値は維持され、17指摘の分類漏れ・重複は無い。判定の書式修正も行っていない。[数値集約](scores.json)と[判定検証](adjudication-validation.json)を参照。

| paired比較 | 正解major検出 | 見逃し | 誤指摘 | 確認不能 | 確認不能率 | 正解表外valid_new |
|---|---:|---:|---:|---:|---:|---:|
| Astra | 6/9 | 3 | 1 | 1 | 1/11 = 9.091% | 3 |
| Sol | 6/9 | 3 | 0 | 0 | 0/6 = 0% | 0 |

両モデルが見逃した正解は`pr-1699-2`（.gitignoreのrawログ追跡）、`pr-1867-2`（イベントとfinalの不一致）、`pr-1888-2`（handoff候補の脱落）。`pr-1872-2`は、goldの起動境界とモデルが挙げた未起動adoptionで契機が異なるため、親が理由を明記した緩い一致として扱った。両モデルで同じ扱いであり、この緩い一致を除いた相対比較も変わらない。

暫定負例の#1686・#1744は両モデルとも指摘0件。Astraの正解表外はvalid_new 3件、false 1件、unconfirmed 1件で、Solの正解表外は0件。モデル出力のfinding IDと正解IDは独立であり、同じ文字列でも別論点の場合がある。機械的な`candidate_matches`は候補を示すだけで、確定判定は`judgments`を正本とする。

## 3条件の区別

| 条件 | 実測・投影 | 入力 token | うちcache | 出力 token | model秒 |
|---|---|---:|---:|---:|---:|
| astra_fixed | Astra 7回の実測 | 9,348,212 | 8,571,008 | 51,064 | 2139.164 |
| reuse_select | 事前選択による7回の投影 | 9,750,621 | 8,951,808 | 53,317 | 2173.832 |

条件③の投影は条件①に対し、入力+出力tokenが+4.305%、モデル時間が+1.621%の増減だった（各1回）。承認済みの採否基準との照合と結論は後述する。


条件② reuse_fixedの品質参照は同じAstraの結果で、別のLLM起動・検出結果を作っていない。[#1790の5再起動の既存監査](../../../.agents/ops/review/evaluation/cases/historical_reuse.json)では全件がnot_reviewedであり、現行の復旧で省ける正式レビューは0件。既往の余分な使用量は入力7,019,963・うちcache6,387,456・出力32,106 token、モデル時間1,041.447秒、review全体1,746.034秒。発生済みの量であり、削減実績として扱わない。今回の物理消費へも加算しない。復旧の実測latencyは今回測っていない。

条件③ reuse_selectは[起動前に固定した選択](selection.json)をpaired結果へ当てはめた反実仮想の量であり、実際の物理起動へ足さない。公開・gate・取消などの5ケースはAstra、通常変更の2件はSol。使用量と時間の差は[report.jsonのselection_improvements](report.json)を参照。選択対象の5つの正解ありケースがAstraで、暫定負例2件は両モデルとも指摘0件なので、条件③の品質集計は条件①と同じ6件検出・3件見逃し・誤指摘1件・確認不能1件・valid_new 3件となる。

## 判定の所在と実行証拠

- [正解論点別の親判定と正解表外の分類](adjudication-materials.md)
- [親の判定](adjudications.json): adjudicator=parent。親の記入内容をそのまま保持している。candidate_matchesのpendingは初期の機械的候補であり、judgmentsが確定判定。
- [実行承認要旨](execution-authorization.json)、[全runと成果物の対応](run-index.json)、[詳細集計](report.json)、[実行・公開projectionの指紋](provenance.json)
- [固定snapshotと入力範囲](../../../.agents/ops/review/evaluation/cases/manifest.json)、[正解表と妥当性確認](../../../.agents/ops/review/evaluation/cases/gold.json)

run_idとcounter.keyはharnessが作った比較標本IDであり、ネイティブセッションIDではない。元出力のSHA-256を親の判定との照合用に保持した。今回の再計算も、非公開stateに対する既存report CLIへparent-adjudicationsを渡して行い、AI起動は0回。最初の再計算はeffort修正前の固定版で行った。正式レビュー修正後は、公開provenanceに結び付けた明示的な読み取り専用report経路で、元ledger・非公開入力・保存final・event・runtimeを再検証して再計算した。通常の起動経路の固定コード検査は維持する。公開したoutput.jsonはunsafe URLの例・秘密値の例・ローカルパス・ネイティブ識別子を除いたprojectionであり、その独立した指紋をprovenance.jsonへ記録した。生のstate、runtime、event、CLI transcriptは非公開のまま保持し、Gitへ保存していない。model/effortの観測値と数値は改変せず、実効証拠のsource位置だけを省略した。root付きのrepo参照をrepo-root表記へ置換した場合もprovenanceのsubstitutionsへ記録する。[公開前の照合検査](validation.json)で数値・指紋・識別子除外・逐次実行を確認した。

## 解釈上の制約と残作業

反復は各1回、正解blockerは0件、通常変更2件は独立した静的確認に基づく暫定負例である。shell、テスト実行、Web検索、継承MCPはreviewerから利用できず、通常の正式レビューとtool範囲が異なる。固定入力は当時取得済みの資料だけで、取得されなかったPR議論や生CIログは利用できない。Astra先行の固定順序とcacheの影響を分離しておらず、時間差を一般化しない。

親は正解指摘を事前に知っていた。選択時には正解表を参照していないという親の申告を記録しているが、その既知情報による選択の偏りは残る。特に正解majorの5ケースを全てAstraにした投影から、Solの重大指摘検出力を推定しない。pairedの一致判定は別に行う。使用量から料金・quotaの削減は推定していない。

## 結論

selectは採用しない。条件③は条件①より入力+出力tokenが4.305%、モデル時間が1.621%増え、事前に承認された「時間または使用量で20%以上の改善」を満たさない。比較は各モデル1反復の最小案である。

正解majorの見逃しは両モデルとも3件で同数、Solの誤指摘・確認不能は0件だった。一方、Astraは正解表外の妥当な指摘を3件出した。この差も1反復の結果であり、Solを通常レビューに使う根拠としては不十分である。既定はfixed（Astra・xhigh）のまま維持する。反復を増やす推奨案へ進むかは人間の判断を待ち、追加の実LLM・実agentは起動しない。

## 範囲外の別issue候補とこのPRの修正

親がvalid_newとしたうち、次の2件は現在のmainにも残っていることを静的に確認した。#1797の範囲外なので別issueの候補として残し、起票・修正は行っていない。確認したmainと根拠は[outstanding-candidates.json](outstanding-candidates.json)に保持する。

- `pr-1699`のモデル指摘`pr-1699-2`: `.agents/ops/worker_logs/schema.py`のhtml_url完全一致により、repo名の大小文字差で正当なログPRを拒否する。
- `pr-1766`のモデル指摘`pr-1766-2`: `src/tasks/annotation/recovery.py`の保存済みhold再利用とRetryWorker拒否により、ファイル修復後も完成receiptを確定できない。

もう1件のvalid_new、Astraのモデル指摘`pr-1888-2`（人間指定effortの消失）は#1797のモデル選択に含まれるため、このPRの別commitで修正した。明示effortを保持し、親指定との矛盾と許可外の組を拒否する。修正前に失敗する回帰4 nodeを先に作成し、修正後の成功を確認した。同じIDの正解論点`pr-1888-2`（handoffの候補脱落）とは別の論点であり、過去14 runのスコアは変更しない。

代表的な正式レビューの取得・結果保存・親確認・gateと、実機での経路確認は親が行う。初回の正式レビューはrequest_changesで、下記4件を修正した。再レビューとgate完了は未確認である。


## 正式レビューの4件の修正と再計算

| 指摘 | 修正した契約と検証 |
|---|---|
| pr-1907-1 | 直接serveと取得ごとに正式な判断捕捉の設定を拒否する。有効な捕捉先を持つ試験で、CLIが明示拒否しjournalが不変であることを確認した |
| pr-1907-2 | 全指摘の分類を必須とし、分類漏れ、matchとotherの重複、同一対応の重複、1指摘への複数goldを拒否する。親の未記入と明示unconfirmedを分ける。現行14 runの判定は受理された |
| pr-1907-3 | runtime失敗のfinalはcandidate_onlyとして保存し、品質件数・指摘率の分母から除く。形式失敗も品質から除外し、比較をincompleteとする。失敗起動の使用量・時間は加算する |
| pr-1907-4 | 固定入力#1686の具体的なworktree参照2か所を公開版から除いた。実際に測定した元入力の指紋を変更せず、公開版の指紋・Git取得元・置換規則と件数を別に保存し、verifyで変換を再現する |

[修正後の検証記録](review-fix-validation.json)に、回帰テスト、各指摘の故障注入と復元後の成功、
元の734個の数値またはnull項目の一致を保存した。固定入力だけでなく判定・README・ソースを含む
PRの公開ファイル全体を検査する。[手順と検証設計](checks.md)を参照。

[run別の入力対応](run-index.json)は、実際に測定した元入力と公開入力の指紋を分ける。
[来歴](provenance.json)のinput_publicationsに、元commit・元の指紋・公開指紋・置換規則を残した。
非公開の元bundle、14 runの28個の公開measurement・output、親判定のバイト列は変更していない。
公開projectionからの新規init・prepareは明示拒否し、今回の入力変換を元の測定と同一視しない。

修正後のparent-adjudicationsによる再計算でも、品質対象は14 run、runtime失敗・形式失敗は0件。
両モデルの正解6件検出・3件見逃し、Astraの誤指摘1・確認不能1・valid_new 3、Solの各0件は変わらない。
総入力18,928,215（うちcache17,301,632）・出力110,870 token、モデル時間4222.006秒も同じ。
条件③の使用量4.305%増・時間1.621%増と、selectを採用せずfixedを維持する結論も変更しない。
再計算での実LLM・実agent起動は0回である。
