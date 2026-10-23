---
id: run-court-scale-drive-session-20261023
type: run
task: synthetic_data_generation
sequence: 29
recorded_at: '2026-10-23'
title: Court永続化の接続・親folder再利用と24 cache再開計測
provider: codex
status: done
issue: 1356
date: '2026-10-23'
config:
  scope: stage-1-storage-optimization-only
  execution: local-cpu
  replayed_source: run-court-scale-colab-resume-l4-20261023
  cache_count: 24
  generated_new_images: false
metrics:
  baseline_new_cache_seconds: 329.0448182159962
  new_cache_steady_median_seconds: 13.519391346002521
  new_cache_steady_p95_seconds: 16.329438978005783
  same_host_cache_save_speedup: 24.338730183536683
  baseline_cache_recovery_seconds: 144.99600465899857
  optimized_cache_recovery_seconds: 6.312532910997106
  resumed_storage_replay_seconds: 351.73006164000253
  full_fetch_seconds: 179.38068070499867
  full_fetch_stores: 24
  full_fetch_images: 282
  strict_fetch_matches_source: true
repro:
  commit: 74900149c6d963e32ca85ddea4c77e80743c624f
  command: bash knowledge/runs/run-court-scale-drive-session-20261023/repro.sh tmp/court-drive-session-benchmark data/synthetic_data_generation/court-scale/transport-session-20261023-74900149c data/synthetic_data_generation/court-scale/transport-baseline-20261023-74900149c
artifacts:
  run_dir: knowledge/runs/run-court-scale-drive-session-20261023
  context: knowledge/runs/run-court-scale-drive-session-20261023/run.json
  measurement: knowledge/runs/run-court-scale-drive-session-20261023/measurement.json
  analysis: knowledge/runs/run-court-scale-drive-session-20261023/analysis.json
  drive_locations: knowledge/runs/run-court-scale-drive-session-20261023/drive-locations.json
  validation: knowledge/runs/run-court-scale-drive-session-20261023/validation.json
parents:
- run-court-scale-durability-cost-20261023
- run-court-scale-colab-resume-l4-20261023
relations: []
papers: []
tags:
- court-detection
- drive
- durability
- profiling
---

## 結論

保存・検証の同期境界を維持したまま、同じcacheの永続化を329.0秒から16.74秒へ
短縮した。定常23件の中央値は13.52秒で、目標の「数秒〜十数秒」に到達した。
旧代表cacheと定常中央値の比は約24.3倍で、同じcache同士の比較では約19.7倍だった。
24 cacheの新規保存の合計は334.0秒だった。cache内容・QA・archive・dataset versionの
byte一致と、最後の24 store・282画像の厳格fetchに合格した。
この数値はローカルCPUでの保存replayであり、新しい生成・学習やColab上の速度を示さない。

## 実装と境界

親が承認したA+Bと凍結入力archiveの再取得削減を実装した。
共有skillの変更は、明示済み接続設定を使う`drive_core.RcloneSession`という接続部品に限定した。
既定CLIの挙動は変えず、Court固有のfolder探索と不変ファイル操作は`scale/`に置いた。

- 1つの認証付きloopback `rcd`をcommand中だけ保持し、成功・例外・起動失敗で停止する。
  credentialの取得・更新はrclone自身に任せ、設定ファイルの内容をPythonや記録へ出さない。
- 親folderの正のIDだけを同一attempt内で保持する。新しい一覧を、project rootを含む全祖先と
  対象について操作ごとに取得する。既知のIDで一覧を並行取得できるため、pathを毎回先頭から
  探索しない。file結果・不存在・別attemptの一覧はキャッシュしない。
- 保存前後の同名検査、親IDの不変性、file ID・サイズ・MD5を検査する。
  `copyfile`には`IgnoreExisting`を指定し、事前検査後に現れた既存先を変更しない。
  転送の正常応答だけを成功とせず、新しい一覧の内容hashが一致してからacknowledgeする。
- rclone内部の転送後size/hash検査は、不一致時に宛先を削除する実装のため無効化し、
  同じ検査をadapterで同期実行する。不一致・hash欠落は停止し、未ack objectを残す。
  BundleStoreのSHA-256、正確なinventory、復元時のローカル衝突拒否、教師/JPEG検証は維持した。
- optional receipt取得をまとめ、prepareで取得したreceiptを復元に渡す。
  凍結入力の再公開では、ローカルから再構築したarchiveのSHA-256が既存receiptと一致することと、
  Drive上の新しい内容hashを検査し、同じarchiveの二度目の取得を省く。

RC APIは[rcloneの一次資料](https://rclone.org/rc/)を確認した。
実rcloneのローカル結合テストで、`Immutable`だけではRC `copyfile`の既存先を保護できないことが
分かったため、`IgnoreExisting`と独立した照合を組み合わせた。
RCの[copy処理](https://github.com/rclone/rclone/blob/v1.60.1/fs/operations/operations.go)の
失敗時cleanupも確認した。実行環境の版はrun bundleに保存する。

これは他クライアントとのatomic lockではない。新しい同名や置換をpreflight/readbackで検出して
止めるもので、1 run rootを1 writerで扱う契約は維持する。cacheの64候補・軌道・birth level、
QA・生成hash・dataset versionのschemaと意味、receipt確認後にだけ次cacheへ進む境界は変更しない。

## 計測方法と比較の限界

実装の計測版は`74900149c`、旧adapterは`174ebef9c`から取り出した。
両者は同じskill CLIの従来経路を使えるため、旧adapterでは改修前の呼出経路を再現できる。
内容の指紋と正確なコマンドはrun contextに置く。

元のL4 smokeを読み取り専用でfetchし、同じ24 cache（303候補・282採択）、凍結入力、完了publicationを
新しいDrive pathへコピーした。GPUやrendererは動かしていない。これは永続化の試験であり、
新しい視点の生成・品質確認やColabのthroughput測定ではない。

旧実装は親folderを準備した別の新しいpathで代表cache 1件を保存・復元した。
改善版は凍結入力と1 cacheを保存して接続を終了し、空のローカルroot・新しい接続へ切り替えた。
入力とack済みcacheを検証して復元し、残り23 cacheと完了publicationを順次保存した。
最後にさらに別の空root・新しい接続で標準の`fetch_completed_dataset`を実行した。
描画相当の入力ファイルは元の検証済みcacheからstageしたため、保存前の新規描画時間は含まない。
このreplayに実際の`DurableGeneration`のattempt記録は捏造しておらず、通常の再開・停止契約は別途CPU回帰で確認する。

既存smokeのarchive/receiptや採用済み資産は変更せず、比較用の新しい出力も削除しない。
新旧でarchiveのSHA-256とinventoryが同じであることを照合する。
新しいDrive pathとファイルIDは計測bundleから辿れる。

秒数はmonotonic時計による。rclone起動はargvや設定を記録しないwrapperで数え、
RC呼出は操作名と開始・所要秒数だけを記録した。
並行する一覧の秒数を単純合計せず、区間の和集合を`listing_busy_seconds`として集計する。
copyの時間は内容転送だけでなくmetadata lookup等も含む。
Googleへの正確なHTTP件数は取得していないため、RC件数をDrive API件数と同一視しない。
旧実装はRCを使わないため、JSONのRC busy内訳が0でもI/Oが0だった意味ではない。
旧実装については全体の秒数とrclone起動数を測った。

## 実測結果

| 対象 | 旧実装 | 改善版 | 比較範囲 |
|---|---:|---:|---|
| 新規cache保存 | 329.045秒 | 定常中央値13.519秒 | 同じローカルhost。旧1件、改善版の定常23件 |
| 同じcacheのreceipt取得・復元 | 144.996秒 | 6.313秒 | 約0.89 MB、同じhost。改善版は入力復元後の接続 |
| 最初のcache保存 | 上段の旧値 | 16.735秒 | 改善版はcache用folder新規作成を含む |
| 凍結入力＋最初のcache保存 | 未計測 | 44.239秒 | 新しいpathへの初回接続と終了を含む |
| 1 cache復元＋残り23 cache保存＋完了公開 | 参考: 旧Colab 8,622.014秒 | 351.730秒 | 改善版は描画なし、旧Colabは新規描画124.809秒を含むため直接の速度比にしない |
| 完了版の標準fetch | 参考: 旧Colab 3,663.088秒 | 179.381秒 | 24 store・282画像。hostが異なるため記述的比較のみ |

改善版の定常23件は13.018–17.960秒、p95（nearest rank）は16.329秒、平均13.795秒。
定常の平均内訳は、新しい一覧取得のbusy区間5.895秒、copyのbusy区間7.882秒、
そのほかのローカル処理等0.019秒だった。copyはarchive upload、receipt upload、
receipt readbackの3操作で、rclone内のmetadata取得を含む。不可避なpayload転送時間だけとは解釈しない。
残る主費用は新鮮な検査と3つの転送であり、ローカルhash/tar作成が支配的ではない。

| 起動・呼出 | 旧実装 | 改善版 |
|---|---:|---:|
| 定常新規cacheのrclone追加起動 | 109回 | 0回（attempt開始時だけ1回） |
| cache復元のrclone起動 | 44回 | 0回（同じattemptの接続を使用） |
| 定常新規cacheの操作 | 109 process calls | 62 list＋3 copyの65 RC calls |
| cache復元の操作 | 44 process calls | 30 list＋2 copyの32 RC calls |
| 24 cacheの中断・再開・全量fetch | 旧全体の実起動数は未trace | 3 rclone processes |

起動数はDrive操作を対象とし、環境採取用の`rclone version`は含まない。
比較前のsource fetchは別途1 process、旧path準備は別途1 process。
旧の保存109回と復元44回もtraceで実測し、過去のコードからの算出と一致した。
詳細は[measurement](../../runs/run-court-scale-drive-session-20261023/measurement.json)、
[process trace](../../runs/run-court-scale-drive-session-20261023/process-starts.jsonl)、各phaseのRC traceに残した。
新しい出力のroot ID・代表receiptは[Drive locations](../../runs/run-court-scale-drive-session-20261023/drive-locations.json)から辿れる。

再開時の入力復元は11.243秒、入力再公開の照合は3.351秒だった。
再公開にはreceiptの1回の取得だけがあり、入力archiveの二度目のcopyはない。
入力hash・inventoryを先に検査するので、ローカルrootの存在だけを再利用の根拠にはしない。

## 保存費用モデルと試走への影響

`Knew`を新規cache数として、`T ≈ T入力 + Σ描画秒 + 13.52×Knew + T復元 + T完了公開 + Tattempt + Tローカル検証`。
13.52はこの小規模payloadの定常中央値であり、Colabの保証値や大きなarchiveの帯域モデルではない。
cacheごとの同期保存待ちはL4割当中ならGPU占有wall timeへ加算する。
保守的にはp95の16.33秒でも検討する。HTTP quota、長時間変動、大量file時のpage数は別途測定が必要。

元と同じ充填率なら`K≈ceil(24N/282)`、64候補を詰めた楽観ケースなら
`K≈ceil(N/(64×282/303))`。後者は軌道/birth levelの端数を無視しており、cache境界を変える提案ではない。

| 仮の採択枚数N | 元充填率のK | 保存定常費用 | 全64候補のK | 保存定常費用 | cache archive概算GB |
|---:|---:|---:|---:|---:|---:|
| 6,000 | 511 | 1.92 h | 101 | 0.38 h | 0.416 |
| 12,000 | 1,022 | 3.84 h | 202 | 0.76 h | 0.831 |
| 24,000 | 2,043 | 7.67 h | 403 | 1.51 h | 1.663 |
| 48,000 | 4,086 | 15.34 h | 806 | 3.03 h | 3.326 |

表は`13.519×K/3600`のみ。描画・凍結入力・fetch・学習・checkpoint・Colab setupを含まない。
archive量は元の19,537,920 bytesを採択282枚の比で換算しただけで、品質・採択率一定の保証ではない。
新規26 archiveの実測は30,167,040 bytesで、そこへ26 receiptが加わる。
benchmarkでは実際の生成attemptを作らないため、元smokeの2 attemptファイルはコピーしていない。

段階2の最初の候補は各24軌道・密度0（元の4倍、概算約1,100採択・約96 cache）。
同じ充填なら保存約22分＋描画約9分が目安で、fetch/setup/学習は別。
これは試走の規模案であり、本実験の画像数や更新回数を確定するものではない。
実際のcandidate数・QA採択数と新しいL4の実測で置き換える。

## 検証

不変put、同名file/祖先/rootの競合、事前検査とcopy間の作成、親ID置換・消失、
hash不一致・欠落、転送失敗、保存失敗後の停止、cache破損拒否、VM消失後の再開、
未ack objectの不採用、local衝突拒否をCPUテストで確認した。
呼出予算は定常新規cacheが65 RC calls以下、復元が32以下で固定した。
実rcloneのローカル結合テストは1 processの再利用、既存先非変更、例外時・起動失敗時の停止を検査する。

計測後に、すでに完了したconditionを`--request-sha256`で再開する経路も整理した。
入力検証後のpublication復元を分離し、完了版でも入力archiveを1回だけ取得する。
この追加修正は転送replayの計測対象外で、独立したCPU回帰で取得回数と再描画0を確認する。
通常のfetchと、入力検証後のpublication回収の検査内容は同じである。

実行コマンドと結果はrun bundleのvalidationに記録する。再計測は空の`tmp/`と新しいDrive pathを
3引数で`repro.sh`へ渡す。既存benchmark pathへの上書き再実行は拒否される。
通常のknowledge検証は0 errors、再現path検査は0 missing / 0 unverifiable。
ただし現行`kg_repro_paths`は継続行を行単位でparseするため、このwrapper内のscript参照を
0件と数えた。実行済みの原文は保持し、継続行を結合した同じ検査で1参照のbundle内存在を別途確認した
（[追加検証](../../runs/run-court-scale-drive-session-20261023/repro-reference-check.json)）。
この検査ツールの改善は今回の対象外で、親へ返す。

## 停止範囲と次の手順

#1356の段階2は開始していない。Colab・GPUは起動しておらず、新規生成・学習・写り方別評価は未実施。
TensorBoardは学習をしていないため対象外。
最大画像数・共通更新回数・追加seed・長期学習条件・シーン組合せ・D2は未確定のまま。
この記録の完了は永続化改善の検証を指し、#1356全体の完了を意味しない。

親がbranchをpushした後、push済みの正確なcommitで一意名のColab L4を開始し、
既存README/skillに従ってruntimeと採用済みasset bundleを取得・照合する。
コードfingerprintが変わるため、過去のrequestをrenderに流用せず、新しい出力rootで
同じproduction recipe・seed・4シーンを改めてplanする。
改善後の保存費用から試走の軌道数を選び、`render-colab`による中断・空rootへの再開・`fetch`を行う。
その後は固定ViT-B＋幅768の短時間学習、実画像の動画単位val/test確認と写り方別集計へ進む。
生成完了・renderer解放を確認してから学習し、実行の最後は失敗時もColabを`stop`する。
本実験の数値は実測と推奨案を揃えて親へ返し、人間の判断前には確定しない。
