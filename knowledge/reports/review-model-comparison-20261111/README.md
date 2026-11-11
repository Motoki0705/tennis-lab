# 固定7ケースの実LLM比較: 実行結果と親の判定資料

2026-11-11、承認済み最小案を逐次実行し、14回を記録した。追加起動は0回。指摘の一致・誤指摘・採否は親の判定待ちであり、正式レビューの実機受入は未実施。PR #1907 は `Refs #1863` を維持する。

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

runtime失敗0、形式または出力検証失敗0。code未確認のrunは0/14、evidence waitは0件。モデル出力の指摘は計17件であり、正解の検出数ではない。個々の確認不能率とblocker/majorの検出・見逃し・誤指摘は親の判定まで未確定。親未判定のunconfirmed_finding_rateと、モデルのcode未確認・evidence waitは別の値として扱う。

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

## 3条件の区別

| 条件 | 実測・投影 | 入力 token | うちcache | 出力 token | model秒 |
|---|---|---:|---:|---:|---:|
| astra_fixed | Astra 7回の実測 | 9,348,212 | 8,571,008 | 51,064 | 2139.164 |
| reuse_select | 事前選択による7回の投影 | 9,750,621 | 8,951,808 | 53,317 | 2173.832 |

条件③の投影は条件①に対し、入力+出力tokenが+4.305%、モデル時間が+1.621%の増減だった（各1回）。採否の判断は親に残す。


条件② reuse_fixedの品質参照は同じAstraの結果で、別のLLM起動・検出結果を作っていない。[#1790の5再起動の既存監査](../../../.agents/ops/review/evaluation/cases/historical_reuse.json)では全件がnot_reviewedであり、現行の復旧で省ける正式レビューは0件。既往の余分な使用量は入力7,019,963・うちcache6,387,456・出力32,106 token、モデル時間1,041.447秒、review全体1,746.034秒。発生済みの量であり、削減実績として扱わない。今回の物理消費へも加算しない。復旧の実測latencyは今回測っていない。

条件③ reuse_selectは[起動前に固定した選択](selection.json)をpaired結果へ当てはめた反実仮想の量であり、実際の物理起動へ足さない。公開・gate・取消などの5ケースはAstra、通常変更の2件はSol。使用量と時間の差は[report.jsonのselection_improvements](report.json)を参照し、採否は親が判定する。

## 親が判定する材料

- [正解論点別の対応候補と根拠、未割当指摘](adjudication-materials.md)
- [判定template](adjudications.json): adjudicatorはpendingのまま。親がmatchesとother_findingsを根拠付きで記入する。
- [実行承認要旨](execution-authorization.json)、[全runと成果物の対応](run-index.json)、[詳細集計](report.json)、[実行・公開projectionの指紋](provenance.json)
- [固定snapshotと入力範囲](../../../.agents/ops/review/evaluation/cases/manifest.json)、[正解表と妥当性確認](../../../.agents/ops/review/evaluation/cases/gold.json)

run_idとcounter.keyはharnessが作った比較標本IDであり、ネイティブセッションIDではない。元出力のSHA-256を親の判定との照合用に保持した。公開したoutput.jsonはunsafe URLの例・秘密値の例・ローカルパス・ネイティブ識別子を除いたprojectionであり、その独立した指紋をprovenance.jsonへ記録した。生のstate、runtime、event、CLI transcriptは非公開のまま保持し、Gitへ保存していない。model/effortの観測値と数値は改変せず、実効証拠のsource位置だけを省略した。root付きのrepo参照をrepo-root表記へ置換した場合もprovenanceのsubstitutionsへ記録する。[公開前の照合検査](validation.json)で数値・指紋・識別子除外・逐次実行を確認した。

## 解釈上の制約と残作業

反復は各1回、正解blockerは0件、通常変更2件は独立した静的確認に基づく暫定負例である。shell、テスト実行、Web検索、継承MCPはreviewerから利用できず、通常の正式レビューとtool範囲が異なる。固定入力は当時取得済みの資料だけで、取得されなかったPR議論や生CIログは利用できない。Astra先行の固定順序とcacheの影響を分離しておらず、時間差を一般化しない。

親は正解指摘を事前に知っていた。選択時には正解表を参照していないという親の申告を記録しているが、その既知情報による選択の偏りは残る。特に正解majorの5ケースを全てAstraにした投影から、Solの重大指摘検出力を推定しない。pairedの一致判定は別に行う。使用量から料金・quotaの削減は推定していない。

指摘の一致と正解表外の分類、採否、代表的な正式レビューの取得・結果保存・親確認・gate、段階導入と戻す手順の実機確認は親の残作業。最小案の結果だけで既定をselectへ切り替える承認は無い。
