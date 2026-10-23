# 費用モデルと適用範囲

実測の正本は `evidence/`。`analyze_costs.py` で `cost-scenarios.json` を再計算する。
ここでの提案値は未承認で、精度・収束の実証ではない。時刻はUTC、容量は特記なき限り十進bytes。

## 変数と実測

Sは使用シーン数、G_sは各シーンの軌道数、kは密度level、Cは幾何検査前の候補数、
Rは幾何通過数、NはQA採択画像数、Kはbirth levelごとの最大64候補cache数。
費用はNとSだけでは一意に決まらない。同じNでも短い軌道を増やすA1はKが多い。
CPU planからC/R/Kを先に求め、描画後の実測Nで横軸を確定する。補充・quota選択はしない。

| 計測 | 条件・値 |
|---|---|
| L4 cold setup | 約941.4秒（uv・render-only NHT・DINO submodule）。SSH/Notebook待機は別 |
| 採用bundle取得/幾何照合 | 59.818 / 21.728秒 |
| real / backbone取得 | 52.496 / 36.773秒。1,959,500,719 / 342,860,279 B |
| real split凍結 | 8.948秒 |
| CPU plan | 1scene×24:15.184秒、4×6:29.293秒、4×24:35.716秒、4×24密度1:41.235秒、4×128:72.365秒 |
| generation本体 | 469.574秒。render・QA・JPEG/index公開を含む。Drive/restoreを除く |
| durable command | 初回中断75.916秒＋空root再開1,942.931秒＝2,018.847秒 |
| 定常cache間の非描画時間 | 94区間、平均14.226秒、中央値13.872秒、p95 15.614秒 |
| full fetch | 703.461秒、96cache、688ファイル、strict一致 |
| dataset Drive容量 | 117,691,736 B / 198ファイル。inputs 41,717,760、cache tar 75,632,640、publication tar 174,080 B |
| training32更新 | 803.448秒、初回logまで247.456秒、定常中央値0.813845秒/更新 |
| preflight | 学習64.394秒、評価val64.854/test64.828秒。N=1,102のみの実測 |
| train中のval＋強制sync | 約355.987秒（exact保存完了からbest-val保存開始まで）。純推論時間ではない |
| exact / best checkpoint | 83.218 / 78.176秒、各約1.317GB。2個をDriveでhash verify済み |
| 定期sync_tree | 14.368 / 7.832 / 7.944秒 |
| standalone val/test | process全体370.928 / 363.993秒。eager、bf16、batch8 |
| val/test validation本体 | 283.902 / 277.111秒。このうち追加の写り方集計38.107 / 36.346秒 |
| GPU観測peak | 生成742 MiB、学習10,002 MiB、評価2,890 MiB。0.5秒間隔device-wideの下限、allocator peakではない |

初回247秒はimports・事前検査・model生成・lazy compile・初回更新・初回syncの合計。
compile単独の時間にしない。定常値はnative train/loss eventの差で、最初5更新と最終区間を除外した。
0.814秒はloader/log/metricを含む。compileのTensor.item graph breakや既存の決定性warnも実行条件の一部である。

## 式（秒）

CPU計画の5点から非負最小二乗で
`T_plan ≈ 8.327 + 4.752 S + 0.02721 ΣG_s + 0.004666 C`。
シーンの初期化と候補数の効果を独立に十分同定した式ではなく、少数点の計画用近似。

96cacheのrender/QA/local公開を
`T_render ≈ Σ a_s K_s + 0.13952 R` と近似した。
`a_(B00,B01,B02,B03)=(3.814,3.208,3.023,2.993)` 秒/cache、cache単位MAE 0.521秒。
pilotでは各cache8〜25候補程度、密度追加の64候補cacheは未描画なので外挿の限界がある。

同じbase N0_jに対しbirthの候補数は`b_0=N0_j`、`b_l=N0_j*2^(l-1)`（l≥1）。
`K_A2 = Σ_j Σ_(l=0..k) ceil(b_l/64)`。
A1はbase128軌道の分布を外挿し`K≈ΣG_s`と置いた。将来の長い軌道や不成立軌道はCPU planで置き換える。
base128のCPU実測は6,668候補・6,166幾何通過・512cache。
シーン別のpilot QA通過率を掛けた採択見込みは約6,049枚である。
**24軌道は96 strataの一巡未満**なので、これを将来の厳密な採択率・分布とみなさない。

入力tarの5計画のbytesから
`I ≈ 34,836 S + 291,574 ΣG_s + 11,282 C` Bと近似。
USTARのfile header/paddingと終端を数えて算出し、pilotの実archive 41,717,760 Bに一致することを確認した。
cache archive容量Bはシーン別実測bytes/採択画像（疎教師・QA・tarを含む）から推定する。
これは短いcacheの保守的な1画像単価で、密なcacheの固定分償却を分離していない。

同じL4での大きなcheckpoint `copyto` から有効upload速度`v_up=22.16 MB/s`、
backbone全取得・確認から有効download速度`v_down=9.324 MB/s`を置く。
payload転送だけの独立測定値ではなく、file/host/経路に依存する代理値である。
pilot全体へ校正すると

```
T_generation ≈ T_render + 14.191 K + (I+B)/v_up + 181.681
T_fetch      ≈ 7.196 K + D/v_down
D           ≈ I + B + publication/receipt overhead
```

固定181.7秒には今回の意図的中断・復旧・検査境界が含まれる。新規生成の純粋な固定費として独立同定していない。
14.2秒の区間差はputのHTTPだけの時間ではなく、receipt・local検査・次cache前の処理を含む。
描画中のworker周期保存と同じDriveへの競合も実行条件に含む。

現コードの `DriveFolders.lookup` は毎回、祖先と対象folderのfreshな全一覧を取り、同名・IDを検査する。
Kが大きいfolderでは返却項目数が増え、backendのpage取得も追加され得る。
**大規模時も14.2/7.2秒が一定とは保証しない。HTTP/page件数は今回も未測定**。
表の「control費3倍」は感度分析であり上限保証ではない。

現在のbundleは4sceneを一括取得・検証するため、S≤4のcold setupは約1,121秒で固定とした。
別scene用のsetupを新しく作る推定は含めない。assignment確保・接続待ち・最終worker保存・人の待機は別費用。
同VMで入力・compile cacheを保持できれば下がるが、VM寿命や長時間SSH接続は保証しない。

## 学習・評価・checkpoint容量

U更新、V更新ごとのval/checkpoint、M=ceil(U/V)として

```
T_train ≈ 247.456 + 0.813845*(U-1)
          + ceil(0.813845*U/60)*7.944
          + M*(355.987 + 83.218 + 78.176)
T_eval(val+test, framing付き) ≈ 734.921
D_ckpt ≈ M*(1,317,251,194 + 1,317,251,386) B
```

初回syncの一部を重ねて数える保守性がある。毎回coldなcompile-val費を置き、warm後の高速化は仮定しない。
preflightは固定real部分とN/K依存のsynthetic検査が混じり、データ規模1点では傾きを同定できない。
表は上の固定費シナリオと、全preflightを `max(1,N/1102)` 倍する線形シナリオを併記する。
後者は統計的な上限ではない。Sだけではなくstore数Kも効くため、本番最初の条件で式を更新する必要がある。
通常の学習中valとeagerな最終val/test、追加framing集計を二重に同じ純推論費として扱わない。

checkpoint容量はexactとbest-valを全て保持した場合。現実装はbest-valに`save_top_k=1`を直書きし、
設定の-1を無視する。以前のbestを消す経路があるため、本実験前に親が修正すべきである。
pilotは新run・val1回なので削除は起きず、traceは9 rclone calls全成功・delete無しである。

## キャンペーンの足し方

条件ごとに新しいimmutable Drive rootを使う現在の契約で、logical cacheの保存/再取得をそれぞれ数える。
localの共通cacheで再描画を節約できても、別requestのDrive receipt/保存は省略できない。
globalな共有durable cache bankや、完成datasetの単一配布archiveは実装されていない。

`cost-scenarios.json` はA1の各level、A2の基準を除く各level、B1/B2/B3を独立条件とする。
B4はA1の24k条件を共有。各runでcold setupとfull fetchを1回計上。
既存VMで生成直後に学習すればfetch1回を引ける。追加seed・長時間runで同VMを再利用できればさらに引けるが、
VM喪失・中断では入力/データの再取得とsetupを追加する。生成専用VMを追加する場合もsetupを追加する。
同じNのscene配分はシーン別QA見込みから軌道数を丸めた参考値で、exact quotaを実装していない。

比較案・合計表・採用前の判断事項は [decision-request.md](decision-request.md) を参照する。
