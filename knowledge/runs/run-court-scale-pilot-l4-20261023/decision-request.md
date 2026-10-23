# #1356から親へ返す判断依頼

Refs #1356。Colab L4の試走は完了・全成果保存・session停止済み。本実験条件の人間の合意は未取得。
生成/QA/学習のapplication sourceと採用資産は変更していない。

## 実測と前提

4scene×24軌道・密度0から1,209候補、幾何通過1,123、採択1,102枚。
scene別採択はB00=276、B01=345、B02=243、B03=238。
幾何拒否86、QA拒否21は全てrenderer可視対象KP<4。採択ROI穴率0だが、低い視点のぼけ・B03の暗さは代表図に残る。
全1,102枚のpose/KP往復、96store/688ファイルのstrict fetch、空root復旧（1cache再利用＋95新規）を検証した。

| 項目 | 実測 |
|---|---:|
| 描画・QA・local公開 | 469.6秒 |
| durable command計（中断・再開込み） | 2,018.8秒 |
| full fetch | 703.5秒 |
| cold setup＋入力取得/検証/split | 約1,121秒 |
| 新datasetのDrive容量 | 117.69MB / 198files |
| batch8・32更新の学習process | 803.4秒 |
| 定常更新 | 中央値0.814秒/更新 |
| 学習GPU観測peak | 10,002MiB（L4全体23,034MiB） |
| val/test評価process（framing集計込み） | 370.9 / 364.0秒 |
| exact / best checkpoint | 各約1.317GB、保存83.2 / 78.2秒 |

学習はViT-B凍結＋768、合成のみ、長辺512、poseも学習。step32の全lossは有限。
32更新は時間/容量計測用で、精度・収束・画像数のスケーリングを評価する実験ではない。
実画像は既存real-video-v1と一致し、val4,468画像/249動画、test4,372画像/249動画、動画重複0。

既存の写り方注釈が無いため、教師KPのoriginal画像内数で14=全面、12–13=ほぼ全面、0–11=部分と定義。
合成は正のdepthも要求し、occlusionは分類から除いた（規則・assignment SHAを保存）。

| 集合 | 全面 | ほぼ全面 | 部分 |
|---|---:|---:|---:|
| 合成 | 8 (0.73%) | 10 | 1,084 |
| real val | 4,400 (98.48%) | 55 | 13 |
| real test | 4,210 (96.29%) | 142 | 20 |

参考KP平均誤差pxはval全面250.58/ほぼ全面247.02/部分224.99、test250.67/241.09/235.81。
全dense指標の分類別表はknowledge nodeにあり、全画像集計は通常評価と一致した。
少数partialかつ未収束なので、この差から精度の優劣やD2の効果を結論しない。testで条件を選んでいない。

## 費用の式と選択肢

S=scene数、G=軌道数、C=候補、R=幾何通過、N=採択、K=cache数として、秒単位の計画式は次の通り。
詳しい係数・raw根拠・再計算scriptはcost-model.md / cost-scenarios.json。

```
T_render ≈ Σ a_s K_s + 0.13952 R     (a_s=2.99〜3.81秒/cache)
T_generation ≈ T_render + 14.191 K + (I+B)/22.16MBps + 181.681
T_fetch ≈ 7.196 K + D/9.324MBps
T_setup ≈ 1,121秒/新しいVM（現在の4scene bundleを取得）
T_train(U,V) ≈ 247.456 + 0.813845*(U-1)
              + ceil(0.813845*U/60)*7.944 + ceil(U/V)*517.381
T_final_eval(val+test) ≈ 734.921秒
D_ckpt ≈ ceil(U/V)*2.6345GB
```

I=凍結入力archive、B=cache archive、D=dataset総bytes。
N/SだけでKは決まらず、A2はbirth毎に`Σceil(N0_j*2^(l-1)/64)`（l=0はN0_j）。
G=128/sceneのCPU計画は6,668候補/6,166幾何通過/512cacheで、pilot QA率から約6,049採択を見込む。
これは提案であり未描画。24軌道のpilotは96strataの一巡未満なので将来分布は不確か。

preflightの固定費とN/K依存はデータ規模1点では分離できない。下表の2値は固定費の場合と、
観測preflight全体をN/1,102倍する線形シナリオで、信頼区間・上限保証ではない。
数千cache時のDrive一覧/pagination・障害復旧・D2変更費は別。control費3倍の感度分析ではさらに206/329/527時間増える。

比較のため、U=10k・val/checkpoint間隔2k、主要比較に追加seed43/44、
baseline/A1最大/A2最大の3条件を独立40k更新、各run cold setup＋full fetch1回、全checkpoint保持を仮定した。
追加seedの対象はbaseline/A1最大/A2最大/B1/B4で重複を除く（24kではB4=A1最大）。
生成は条件別rootへ全て保存。共通cacheのlocal再描画節約は見込まず、Driveの条件間共有も仮定しない。

| 最大の目安 | 主比較条件数 | 全学習run数 | 主比較のみL4確保h | 追加seed/40k込みL4確保h | Drive増分 |
|---|---:|---:|---:|---:|---:|
| 24k | 8 | 19 | 120〜127 | 225〜242 | 約385GB |
| 48k | 10 | 23 | 175〜187 | 313〜344 | 約447GB |
| 96k | 12 | 25 | 270〜291 | 440〜494 | 約491GB |

同VMで生成後すぐ学習すればfetchを引ける。一方、新VMや復旧のたびにsetup/fetchを加算する。
小さいdatasetを重ねて保存する条件別archiveと、epochではなく固定更新で回す学習をともに数えた。
固定14.2秒/cacheを数千cacheへ外挿しており、上の大規模予算を確約してはいけない。
Drive空きは観測時4.59TBで上表を収容できるが、checkpoint保持が容量の主成分。
Colab残高344.00CUはアカウント全体の観測値で、他のassignmentも稼働中。
参考1.54CU/L4-hourなら24k全案も約347〜373CUとなり、追加枠または費用削減が必要になる。

## 人間が本実験前に決める項目（全て未確定）

| 論点 | 推奨案 | 代案・判断理由 |
|---|---|---|
| D2 | **大規模生成前に全面を支える視点の比較・再検討** | 現D2を維持するなら、合成全面0.73%対val98.48%を解釈の制約に明記。学習精度による因果判定ではなく支持範囲の警告。方式変更は別の設計合意が必要 |
| 基準/最大規模 | **128軌道/scene・密度0を基準候補、まず最大約24k** | 約6k/12k/24kのA1/A2＋Bを8条件で完結。48k/96kは比較点が増えるが費用が上表まで増大。元の9条件を減らす変更なので親/人間の明示合意が必要。採択実数で横軸を記録し補充しない |
| 共通更新数 | **10k、warmup500、val/checkpoint2k** | 5k/250/1kなら24k全案189〜206h、20k/1k/4kなら297〜314h。10kは80k画像提示＝約6kで13.2周、約24kで3.3周。収束の保証は無く、全条件で同一scheduleを事前固定する |
| 追加seed | **43/44をbaseline・A1最大・A2最大・B1・B4に追加、同一条件は重複排除** | 24kでは4条件×2追加=8run。減らすならA1/A2端点だけ3条件×2にし、Bのseed不確実性を残す。生成seed42は固定 |
| 長時間の代表 | **baseline/A1最大/A2最大の3条件をseed42・40kで独立実行** | 20kに抑える案やA1最大/A2最大の2条件に絞る案は安いが未収束確認が弱まる。max_steps変更resumeは禁止し、別run・独立LR scheduleとして比較する |
| scene組合せ補足 | **24kでB02+B03を1条件追加、seed42・共通更新数** | 主B00+B01との同じ2scene比較。追加約21〜22h/15.5GB。無しならscene数とscene個性の交絡を残す。全2scene組合せ6通りは追加5条件で費用が増える |

推奨案の費用表は現D2を維持する条件の計算で、D2検討とscene補足1条件の費用を含まない。
D2を変えた場合は採択率・N0/cache数・生成単価が変わり得るため、この表を流用せず基準軌道数と費用を再計測する。
選んだ範囲を#1343へ記録してから#1357以降を開始する。
今ある残高で全推奨案を無条件に実行できるとは扱わず、主比較と追加実験の予算配分も合意してほしい。

## 本実験前の実装上の停止事項

`scaling/training.py`のbest-val callbackは`save_top_k=1`を直書きし、設定-1を無視する。
`PublishingCheckpointIO.remove_checkpoint → RcloneArtifactStore.remove_file → deletefile`の経路がある。
pilotは新run・val1回で、9回のrclone callにdeleteが無いことを確認した。
担当境界に従い修正していない。親がretention修正を別途取り込み/pushしてから複数valの本学習を開始する。

成果は `knowledge/nodes/synthetic_data_generation/000030-run-court-scale-pilot-l4-20261023.md` と対応run bundle。
Drive datasetは `data/synthetic_data_generation/court-scale/pilot-l4-20261023-dc5c253`、
学習は `outputs/court_detection/train/court-scale-pilot-l4/s42`、評価は `outputs/court_detection/evaluate/court-scale-pilot-{val,test}/s42`。
