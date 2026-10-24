---
id: run-court-scale-durability-cost-20261023
type: run
task: synthetic_data_generation
sequence: 28
recorded_at: '2026-10-23'
title: Court規模試走前のDrive永続化コスト分解
issue: 1356
provider: codex
date: '2026-10-23'
status: done
config:
  scope: durability-cost-investigation-only
  source_run: run-court-scale-colab-resume-l4-20261023
  live_probe: one-existing-cache-read-only
  execution: local-cpu
metrics:
  probe_archive_bytes: 890880
  probe_receipt_seconds: 88.18260232399916
  probe_archive_restore_seconds: 59.39676435900037
  probe_rclone_processes: 44
  probe_resolve_listings: 38
  probe_resolve_listing_seconds: 110.77178000998538
  smoke_new_cache_interval_median_seconds: 338.18840150000005
repro:
  commit: 1eedc16ac23418eefd88dfb7dbfc9dfb7bae1211
  command: bash knowledge/runs/run-court-scale-durability-cost-20261023/repro.sh tmp/court-drive-cost-reproduction
artifacts:
  run_dir: knowledge/runs/run-court-scale-durability-cost-20261023
  analysis: knowledge/runs/run-court-scale-durability-cost-20261023/analysis.json
  measurement: knowledge/runs/run-court-scale-durability-cost-20261023/measurement.json
  trace: knowledge/runs/run-court-scale-durability-cost-20261023/rclone-events.jsonl
  validation: knowledge/runs/run-court-scale-durability-cost-20261023/validation.json
parents:
- run-court-scale-colab-resume-l4-20261023
relations: []
papers: []
tags:
- court-detection
- drive
- profiling
---

## 結論と停止範囲

#1356の段階1で、Driveのpath解決と確認を繰り返す非効率を特定した。
約0.89 MBの既存cacheを1件復元するだけで147.579秒、rclone起動44回を要した。
うちpathの同名重複検査に伴う一覧取得38回が110.772秒、計測時間の75.1%を占めた。
不可避なpayload転送だけで過去の長時間を説明することはできない。

依頼の停止条件に従い、修正案と期待効果を親へ返す。productionコードは変更しておらず、
Colabを起動せず、新しい生成・学習・写り方別評価にも進んでいない。
このノードの`done`は費用調査の完了であり、#1356全体の受入条件を満たした意味ではない。
最大画像数・共通更新回数・追加seed・代表条件・シーン組合せ・D2は未確定のまま。
TensorBoardは学習を行っていないため対象外。

## 記録とコードの固定

元データは[4シーンのL4復旧検証](000027-run-court-scale-colab-resume-l4-20261023.md)。
その`colab-operations/{interrupt,resume,fetch}/stdout.log`、cache metrics、Drive receipts、
resumed attempt、dataset versionを読む。元の証拠は編集せず、
[analysis.json](../../runs/run-court-scale-durability-cost-20261023/analysis.json)に入力ファイルの内容指紋と派生集計を残した。

調査checkoutは`1eedc16ac`。下記5ファイルは元のproducer `6ed3f29e8`からbyte変更がないことを
`git diff --stat 6ed3f29e8 HEAD -- <対象5ファイル>`で確認した。

- [durability.py](../../../src/synthetic_data_generation/dataset/court/scale/durability.py)
- [durable_storage.py](../../../src/synthetic_data_generation/dataset/court/scale/durable_storage.py)
- [generate_court_scale.py](../../../src/synthetic_data_generation/scripts/generate_court_scale.py)
- [drive.py](../../../.agents/skills/tennis-drive/scripts/drive.py)
- [drive_core.py](../../../.agents/skills/tennis-drive/scripts/drive_core.py)

ローカル計測は元の`SkillDriveFiles`と`BundleStore`をそのまま呼び、rcloneの前に観測用wrapperを置いた。
既存smokeの最初のcache receiptとarchiveだけを読み、worktreeの新しい`tmp/`へ復元した。
Driveへのupload・上書き・削除は行っていない。元シーンや採用bundleも変更していない。
wrapperは設定内容・HTTPヘッダ・response本文を記録せず、呼出分類、秒数、一覧件数だけを保存する。
archiveのSHA-256・サイズ・正確なファイルinventory検査は既存の`BundleStore.restore`が行う。
取得receiptはGitにある元receiptと完全一致し、6ファイルを復元できた。

## 過去の所要時間と保存量

| 操作 | コマンド全体の秒数 | その操作で新規生成した秒数 | 残りの秒数 |
|---|---:|---:|---:|
| 1 cache後に中断 | 733.712 | 12.252 | 721.460 |
| 空rootへ再開 | 8,622.014 | 124.809 | 8,497.205 |
| Driveから別rootへfetch | 3,663.088 | 0 | 3,663.088 |

元の生成137.061秒は初回12.252秒と再開124.809秒の合計。
再開の98.55%は新規生成以外だが、これをすべてネットワーク待ちと断定しない。
入力復元、tar作成・展開、ローカルhash/教師/JPEG検査、公開処理も含まれる。

新規cacheのarchive保存確認時刻`verified_at`には22個の連続差分が得られ、
中央値338.188秒、最小325.777秒、最大351.393秒だった。
その区間の生成秒数を引いた平均は332.654秒/cache。
時刻はarchiveの`put`終了後・receipt upload前なので、差分は前cacheのreceipt処理と次cacheの処理を含む。
単独uploadの計測値ではない。初回・最終公開の固定費も含めていない。

| Drive上の単位 | 個数 | archive bytes | archive内ファイル数 |
|---|---:|---:|---:|
| 凍結入力 | 1 | 10,567,680 | 33 |
| cache | 24 | 19,537,920 | 144 |
| 完了publication | 1 | 61,440 | 7 |
| 合計 | 26 | 30,167,040 | 184 |

これに26 receiptと2 attemptが加わり、対象runの通常出力は54 remoteファイル。
282 JPEGそれぞれをDriveへ転送しているわけではなく、既にtar単位へまとまっている。
cache archiveにはJPEG/indexだけでなくQA・completion・metricsとtar paddingも含む。
元のJPEG＋indexは15,345,154 bytesなので、保存量の見積りにはその値だけを使えない。
採用3DGS bundle、Colab運用記録、失敗保存物は上表の範囲外。

## 呼出数の分解と再取得

`drive_core.py:127–166`の`resolve`は、`gdrive:`から各path成分を順に`lsjson`し、
毎回新しいrcloneプロセスを起動する。例えばcache receiptでは8成分、objectでは7成分。
各rclone内のremote初期化・親folder探索・ページングは、この明示的な起動数にさらに内包される。

`drive.py:185–208, 104–115`のdownloadは、転送前と転送後の`resolve`、`copyto`、
localとremoteのhash付きstatを行う。深さdの1ファイル取得は`2d+3`回。
`BundleStore.receipt`はさらに`exists`で同じpathを解決してからdownloadするため、
receiptとobjectを合わせたcache復元は`8 + (2×8+3) + (2×7+3) = 44`回になる。

| 成功経路 | rclone起動数（コードから算出） | 内訳・前提 |
|---|---:|---|
| cache 1件fetch | 44 | 一覧38、stat4、copyto2 |
| 凍結入力または完了publication 1件fetch | 41 | receiptが1段浅い |
| 新規cache 1件prepare〜保存確認 | 109 | prepare exists、receipt確認、2 put、receipt readback |
| 既存cache 1件prepare〜復元 | 52 | prepare existsにfetch 44回を追加 |
| 元のfetch全体 | 1,138 | `2×41 + 24×44` |
| 元のresume全体 | 2,772 | `2×41 + 7 + 52 + 23×109 + 93 + 31` |

これは祖先folderが存在し、成功し、retryのない経路の算出値。元のColabでプロセスをtraceした値ではない。
新規cacheでは`SkillDriveFiles.put`が`exists→upload --verify→inspect`を呼び、
upload内にも事前・事後の解決があるため、1 putで4回path全体を辿る。
receipt readbackでも`exists→downloadの事前・事後`と3回辿る。

resume式の最初の`2×41`は凍結入力archiveの二重取得。
CLIが`restore_inputs`で復元後、`DurableGeneration.run → BundleStore.publish`が既存receiptを読み、
同じ入力archiveを再び`restore`する（`generate_court_scale.py:275–278`、
`durability.py:401–409`、`durable_storage.py:195–202`）。
最初のcacheの復元は1回であり、再描画は元記録どおり0回。全cacheを再取得しているわけではない。

fetch全体の1,138回は一覧982回、remote stat52回、local stat52回、copyto52回。
したがって明示的remote操作は少なくとも1,086回相当だが、**正確なDrive HTTP API呼出数は未計測**。
rclone内部の探索・転送session・retry等を含むHTTP数とは一致しない。
`verify --download`も常に全内容を再取得する意味ではなく、共通hashがない場合だけ内容取得する。
今回の新規put/downloadは通常のhash照合経路で、想定HTTP数を実測値として扱わない。

## ローカルの読み取り実測

| 操作・分類 | 回数 | 秒数 |
|---|---:|---:|
| receiptのexists＋取得・照合 | 27 | 88.183 |
| archive取得・hash・展開・衝突検査 | 17 | 59.397 |
| うちpath解決の一覧取得 | 38 | 110.772 |
| うちlocal/remote stat | 4 | 17.163 |
| うちcopyto | 2 | 18.020 |

最初の2行と下3行は同じ処理を別の粒度で集計したもの。下3行を上2行へ加算しない。
総147.579秒とrclone内145.955秒の差約1.624秒はwrapper/Python/展開など。
`copyto`にもremote解決やmetadata操作が含まれるため、18.020秒を純粋なpayload転送時間とはみなさない。
共通祖先`gdrive:`〜対象run rootの6つの親一覧をそれぞれ5回取得していた。
一覧は小さく、親ごとの件数は[trace](../../runs/run-court-scale-durability-cost-20261023/rclone-events.jsonl)で確認できる。

ローカルCPUでの1件だけの計測であり、Colabのネットワーク性能の再計測ではない。
秒数はmonotonic時計による。保存したUTC開始/終了時刻には時計補正の影響があり、差分から秒数を再計算しない。
過去のColab fetchも`3663.088 / 1138 ≈ 3.22秒/起動`、新規cacheの定常区間も
約338秒/109起動であり、今回の結果と同程度だが、HTTP単位の原因別時間を復元できたわけではない。

## 現行方式の時間モデルと規模への影響

`K`をcache数、`g_i`を新規cacheの生成秒数とすると、
`T生成 ≈ T入力 + Σg_i + 332.65×K新規 + T復元 + T完了公開 + Tattempt + Tローカル検証`。
小規模の同程度のcache長では、`Tfetch ≈ 3.22×(82+44K) + 残差`秒が粗い目安になる。
これはこの1 run近傍のモデルで、quota制限や大量folderの遅延を外挿して保証するものではない。

採択数Nに対するcache数は軌道数・密度birth level・64候補chunkの端数に依存する。
元runの充填率を保つ場合は`K≈ceil(24N/282)`。
採択率`p=282/303`を保ち全chunkが64候補なら`K≈ceil(N/(64p))`となる。
後者は端数や軌道を跨げない制約を無視した楽観ケースであり、chunk境界を変更する提案ではない。

| 仮の採択枚数 | 元の充填率のcache数 | 永続化等の定常費用 | 全64候補chunkのcache数 | その定常費用 | cache archive概算GB |
|---:|---:|---:|---:|---:|---:|
| 6,000 | 511 | 47.2 h | 101 | 9.3 h | 0.416 |
| 24,000 | 2,043 | 188.8 h | 403 | 37.2 h | 1.663 |
| 96,000 | 8,171 | 755.0 h | 1,612 | 149.0 h | 6.651 |

表は`332.654×K/3600`だけを示し、描画・凍結入力・fetch・学習・VM setupの費用を含まない。
Driveの現容量や総GPU時間、本実験規模を確定する表ではない。
cache archive量は`19,537,920×N/282` bytesという同じ小規模比率の概算。
凍結入力10.568 MBもA1では軌道数とともに増え、条件別入力・運用記録・checkpointも別に必要。
A1とA2でcache充填率が変わるため、画像数だけで1本の費用比例式にしない。

## 親へ返す修正案と選択肢

**推奨A: verified file操作の中で重複処理をまとめる限定修正を先行する。**

対象案は`durable_storage.py`、必要なら`durability.py`の接続、および共有の
tennis-drive CLI/libraryとそのテスト。shared skill libraryの変更を含むため、親が担当範囲を判断する。

1. 既存Drive libraryに、immutable putを1操作として実行し、照合済みのpath/ID/hashを返す関数を設ける。
   新規putの外側`exists`と独立`inspect`を重複させず、操作内の新鮮な事前・事後path検査と
   hash照合の結果を返す。既存同名objectは内容一致を確認し、不一致・曖昧pathでは停止する。
2. receiptのoptional取得を1操作へまとめる。取得自身の事前検査で明示的に不存在を区別し、
   存在時の事後重複検査・hash照合は残す。通信・認証・重複・hash不一致を不存在へ丸めない。
   `prepare`と`_restore_cache`の二重existsも同じ取得結果で扱う。
3. `SkillDriveFiles`は上記library関数を同じprocessで呼ぶ。rclone自体の呼出しと照合は必要なだけ残し、
   CLIとlibraryで異なる保存・検証規則を二重実装しない。

1 putを`4d+3→2d+3`、receipt取得を`d+(2d+3)→2d+3`とする設計予算なら、
新規cacheは109→71回（34.9%減）、元resumeは2,772→1,819回（34.4%減）、
fetchは1,138→932回（18.1%減）となる。
同程度の呼出時間を仮定すれば、元規模でresumeは約2.4h→1.6h、fetchは約61分→50分が目安。
**未実装・未benchmarkの効果予測**であり、少なくなった呼出しの深さで秒数も変わる。
入力archiveの二重取得削減はこの予測に含めていない。

**選択肢B: Aに加え、各rclone内の親folder探索と起動費用も減らすtransportを設計する。**
毎回の新鮮な同名重複検査を残しつつ、直前の一覧で得たparent IDを使う経路や、
1コマンド内の接続再利用を検討する。ID指定をpathの重複検査の代替にはしない。
効果は未計測で、認証・失敗境界・rclone側の対応確認が必要。Aだけでは本番の短縮が不十分な場合の追加範囲とする。
凍結入力の2回目のrestoreも、同一attemptで検証したbytesと再構成hashを再利用できるか別に検討する。

**選択肢C: 現行実装で段階2を続ける。** コード変更は不要だが、上の費用とL4占有時間を負担する。
描画より確認処理が長いことが実測できており、推奨しない。

どの修正でも、64候補cache/軌道/birth level、camera ID・QA・生成hash・archive hash、
receiptを最後に確認してから次cacheへ進む同期境界、衝突停止、未ack objectの扱い、
fetch後の厳格な教師/JPEG検査を維持する。恒久的なpath cacheやTTLで重複確認を飛ばす案は採らない。

## 修正後の検証と再開条件

親が範囲を決め、別の実装で通常テストと小規模Drive計測を通す。
重複folder/file、既存object不一致、remote変更、hash不一致、欠落receipt、孤立object、
転送失敗・認証失敗、再開時のlocal衝突、保存確認前に次cacheへ進まないことを検証する。
新しいDrive pathで保存・中断・再開・fetchを計測し、同一QA/schema/countとstrict読み取りを確認する。
srcやskillを変えたcommitは親がpushしてからColabへ渡し、生成requestは新しいコードidentityで凍結する。

その後に#1356の段階2へ戻り、実測した保存時間から4シーンの試走規模を決める。
既定recipeとViT-B＋幅768を使う方針は再判断しない。新規生成・短時間学習、実画像val/testの
動画分割確認、全面/ほぼ全面/部分の既存分類に沿った評価、規模・更新回数・補足条件の推奨案を揃え、
人間の判断は親が取る。この調査ではD2見直しや本実験開始を承認したことにしない。
