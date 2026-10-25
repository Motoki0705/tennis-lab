# Court scene-diversity preparation

目的は、学習画像数と更新予算を固定し、単一シーンの密な撮影と複数シーンの多様性の差を測ること。既存のDINOv3サイズ比較に使ったB00〜B03のtrain軌道を固定し、追加視点を3DGSから描画する。生成はユーザー指定の GPT-6.1 Sol / xhigh サブエージェントが担当する。

## 確定条件

| 条件 | train枚数（B00/B01/B02/B03） | 合成val枚数（同順） |
|---|---|---|
| 1 scene | 5500 | 500 |
| 2 scenes | 2750 / 2750 | 250 / 250 |
| 3 scenes | 1833 / 1833 / 1834 | 167 / 167 / 166 |
| 4 scenes | 1375 / 1375 / 1375 / 1375 | 125 / 125 / 125 / 125 |

各条件の合計6000枚のうち5500枚をtrain、500枚を合成valに使う。実画像は学習に含めず、前回の元val 2211枚を同じ順序で固定する。前回の評価receiptとのID順序照合をprepareで行う。

元train軌道からsceneごとの固定holdoutを取り、trainとvalの撮影軌道を分離する。全条件を通して同じ軌道がtrainとvalへ移らない。scene内の画像選択はtrain/valを別々にnestedにする。生成poolの元のtrain保存形式は維持し、8個の選択manifestでsplitを定義する。生成条件・QA・画質の証拠は[生成bundle](../run-court-scene-diversity-6000-v1/README.md)が正本。

- DINOv3 ViT-S/16、backbone完全凍結、LoRAなし。
- Transformer dim384 / depth8 / 6 heads / head dim64 / FFN1024、DPT512、head hidden256 / depth2。
- 長辺512、batch8（合成のみ）、seed42、33,160 optimizer updates。
- 元のサイズ比較のwarmup割合3/20を更新数へ移し、warmup4974更新。学習率1e-4、cosine、最小学習率1e-6、weight decay1を維持。
- 評価と定期checkpointは3316更新ごと（全10回）。early stoppingなし。best-val保持は無効で、主比較は最終33,160更新のcheckpoint。train/val両方を含むfit-timeのval/lossをドメイン差として使わない。
- 全条件を同じ事前学習DINO重み・同じ後段seedから独立開始する。5500はbatch8で割り切れないためepoch末尾にsamplerの4枚paddingがあるが、全条件が同じ688 batch/epoch、同じ33,160更新である。

## 実行

このbundleを含む専用worktreeをカレントにし、`.venv` はメイン環境へリンクする。`campaign.py` はgit common rootからメインのdata/ckpt/outputsと共有queueを解決する。

```bash
.venv/bin/python knowledge/runs/run-court-scene-diversity-preparation-s42/campaign.py prepare
.venv/bin/python knowledge/runs/run-court-scene-diversity-preparation-s42/campaign.py preflight
.venv/bin/python knowledge/runs/run-court-scene-diversity-preparation-s42/campaign.py enqueue --session YOUR_SESSION_ID
```

prepareはデータ未生成でも4条件の解決済み設定とplanを作れる。生成先はメインの`data/court_scene_diversity_6000_v1`、選択ファイルは`scene-N.train.subset.json`と`scene-N.val.subset.json`。本番GPU probeは共有queue経由で`gpu_probe.py`を実行し、3更新の有限loss/勾配、batch shape、GPUメモリを確認する。probeの更新は本学習へ引き継がない。enqueueは同じplanとreadyに対するprobe成功も要求する。WSLのホストメモリを抑えるため、コンパイルworker数は`TORCHINDUCTOR_COMPILE_THREADS=1`に固定する。

preflightは全poolの照合、各sceneの枚数、8個の選択、入れ子、条件をまたぐ軌道漏洩、元val、初期重みを検証してready receiptを書く。欠落・不一致なら学習jobは登録しない。固定したplan/readyが変わった場合も停止する。

設定とregistryはメインの`outputs/court_detection/campaign/scene-diversity-6000-v1`にある。enqueueは共有queueへresource allで4つの学習＋評価jobを重複登録防止キー付きで登録する。実行時にもpreflightを繰り返し、33,160更新のcheckpointが一意であることを検査する。既存学習出力の上書きや別checkpointへの代替はしない。

学習後は各jobが合成val500枚と実画像val2211枚を別々に評価する。4つのevaluation.jsonを次に渡す。

```bash
.venv/bin/python knowledge/runs/run-court-scene-diversity-preparation-s42/report.py \
  --evaluations /path/to/scene-1/evaluation.json /path/to/scene-2/evaluation.json \
                /path/to/scene-3/evaluation.json /path/to/scene-4/evaluation.json \
  --output /path/to/new-comparison
```

CSV/JSONとシーン数を横軸にした曲線を作る。KP平均/中央値pixel誤差、SEG/semantic LINE mIoU、LINE Diceを両domainで示し、差分は実画像−合成を符号付きで記録する。誤差とIoU/Diceでは良い方向が逆であり、差分を一つの堅牢性スコアに混ぜない。合成valはscene構成が変わるが、実画像valは全条件で同一。poseは合成だけの補足指標である。

単一seedの比較であり、シーン追加には材質だけでなく背景・照明・視点分布の差も含む。学習済みモデルの精度・GPU所要時間は実測後に各runへ記録し、準備のCPU検証を学習成果と扱わない。
