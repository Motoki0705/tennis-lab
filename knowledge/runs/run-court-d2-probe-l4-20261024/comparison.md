# D2の凍結L4試行：画像未生成で停止

## 結論

4条件×4scene×12枚を変更せず、指定commitでColab L4を開始した。
環境構築と採用資産24ファイルの取得・照合は成功したが、最初のscenario（current/B00）の
一時pathが相対pathだったため、NHT request構築で停止した。

**完成画像0・NHT呼び出し0・QA0。D2候補の採択率・拒否理由・α診断・画質・JPEG容量は未測定である。**
幾何上のfull/near_full/partialから画質の合格を推定しない。原画・overlay・代表動画も生成されていない。
render-probeの終了コード1とDrive保存を確認し、再描画せずsessionをstopした。停止後の割当は0件。

[実行情報](run.json)、[失敗state](evidence/probe-state.json)、[stack trace](evidence/render-stdout.log)、
[Drive status](evidence/remote-render-status.json)、[停止receipt](evidence/stop.json)を保存した。

## 消費枠と再実行前の判断

- 凍結の描画計画：192枚、段階の総上限200枚。
- 永続化されたstate：draw_attempts=12、completed_frames=0、scenarios=[]。
- profiler：NHT clientのrender呼び出し0、evaluate_render呼び出し0。GPU観測peakも0MiB。
- VM消失・成否不明は無く、明示的な契約違反で終了した。台帳の12枚を保守的に消費計上すると残り188枚。
- **同じ192枚を全部再実行すると計204枚となるため、現上限内では全再実行できない。**
  実NHT呼び出し0と台帳12の扱い、再実行の許可・予算を親が判断する。枠を自動で戻さず、新VMも開始しない。

## 原因と修正案

前回runのrender_probe.pyはmainで `TemporaryDirectory(dir="tmp")` を作り、
`Path(temporary)` をそのままrunへ渡す。このPython環境では相対pathが返る。
既存rendererはそのrootから `.render/.../renders` と `request.json` を作り、
NHTRenderCommandRequestが必須の絶対pathを満たさず、subprocess起動前に拒否する。
CPUでの選択と192カメラ復元の検査は、このrequest構築境界を実行していなかった。
cProfileを外しても同じpath契約違反となることをCPU fixtureで再現した。

修正案は、**新しいrunのadapterで共有一時rootを `Path(temporary).resolve()` にしてから既存runへ渡す**こと。
output_directoryだけのresolveではarbitrary_request_pathが相対のままなので不足する。
[CPU検査](evidence/path-contract.json)は2つの拒否とroot全体のresolveによるrequest構築成功を確認した。
renderer起動・L4画質の成功は未検証。今回、src・本番recipe・前回runを変更していない。

## 今回測れた時間と容量

| 計測 | 実測 |
|---|---:|
| uv sync | 27.390秒 |
| render-only NHT導入 | 951.771秒 |
| 採用bundle取得・転送verify | 52.218秒 |
| manifest・幾何照合 | 22.454秒 |
| 上記setup合計 | 1,053.834秒 |
| 失敗process全体（cProfile付き） | 284.179秒 |
| profile内の4scene入力読込 | 24.535秒 |
| profile内の全192カメラ復元・照合 | 240.680秒 |
| profile内のrun準備・契約違反・終了処理 | 5.534秒 |
| renderer実行 / QA / JPEG・NPZ保存 | 未起動 |
| render-probe workerの最終保存 | 14.076秒 |
| Drive実験出力 | 3,374,406B・8ファイル |
| 採用bundle | 958,142,735B・24ファイル（入力、変更なし） |

profileの累積時間は上記の相互に重ならない入口だけを抜き出した。残り約13.43秒にはimport、
profile出力などを含む。worker周期保存はprocessと重なるため、純粋な転送秒数として分離していない。
最終保存はstatusのfinalizing更新からfinished_atまで。画像保存throughputやJPEG単価は得られていない。
profile本体はGitに入れず、Driveと検証付き回収先に保持した。

## 24k/48k/96kの費用感度：setupだけ更新

新しいQA・JPEG実測が無いので、要求された**新D2の採択率・容量を反映した費用確定はできない**。
以下は前回の条件付き費用表からsetup部分だけを今回実測に差し替えた感度表。
今回のsetupに、前回のreal/backbone取得とsplit作成98.217秒を移植すると、
将来のcold training setupは条件付きで1,152.051秒（旧1,121.163秒、差+30.888秒/run）になる。
学習VMごとにこの導入を行う、従来の保守的な会計を維持している。

10k共通更新・warmup500・val2k、追加seed43/44、代表3条件の独立40k、
各学習run cold setup/full fetch、全checkpoint保持の容量仮定を引き継ぐ。
QA率・JPEG容量・render/QA/local公開・Drive control費は全て前回の仮定のまま。
幅はpreflight固定/線形の感度ケースで、信頼区間でも時間上限でもない。
24k等は旧倍率系列の呼称で、採択実数ではない。Gはcurrent/wide125で128、hull25系で96/sceneという未採用案。

| 候補 | 旧最大呼称 | 主比較のみL4時間 | 追加seed/40k込みL4時間 | Drive増分GB |
|---|---:|---:|---:|---:|
| current | 24,000 | 120.2–127.2 | 225.5–242.2 | 384.9 |
| current | 48,000 | 175.4–187.0 | 313.2–343.9 | 446.8 |
| current | 96,000 | 270.4–291.3 | 440.1–494.4 | 491.5 |
| wide125 | 24,000 | 119.0–126.1 | 224.0–241.3 | 385.0 |
| wide125 | 48,000 | 174.4–186.3 | 312.0–343.8 | 447.2 |
| wide125 | 96,000 | 269.7–291.5 | 439.3–495.8 | 492.5 |
| hull25_margin0 | 24,000 | 98.3–105.4 | 195.1–212.2 | 384.3 |
| hull25_margin0 | 48,000 | 143.5–155.4 | 267.3–299.0 | 446.1 |
| hull25_margin0 | 96,000 | 221.0–242.7 | 370.0–426.2 | 490.7 |
| hull25_margin0_wide125 | 24,000 | 98.0–105.2 | 194.7–212.0 | 384.3 |
| hull25_margin0_wide125 | 48,000 | 143.3–155.3 | 267.0–299.1 | 446.3 |
| hull25_margin0_wide125 | 96,000 | 220.9–242.8 | 369.8–426.8 | 491.0 |

機械可読の式・前提は [setup-only-costs.json](evidence/setup-only-costs.json)。
今回の失敗probe自体のVM確保時間、scene補足、assignment確保・人の待機はこの将来費用表の外。
現時点のアカウント残高は停止後337.45CUという観測だけを残し、全条件を実行できる保証にしない。

## 親への推奨と保留

D2の数値変更を採用する根拠は今回追加できなかった。前回のCPU比較と
[前回の提案](../run-court-d2-support-cpu-20261024/evidence/comparison.md)を維持し、L4比較を経るまで採用を保留する。
新規の候補追加・変更や、失敗を理由とするD2/QA緩和は提案しない。

| 未決事項 | 今回の扱い |
|---|---|
| D2 | 採用保留。前回の凍結192枚を検証するためのpath修正・再実行予算を先に判断 |
| 最大画像数 | 前回の「まず約24k」案を条件付き候補として維持。48k/96kへの拡大は追加の実測・予算判断が必要 |
| 共通更新数 | 10k、warmup500、val/checkpoint2kという前回案のまま未採用。32更新の実測は収束の証明ではない |
| 追加seed | 生成seed42を固定し、学習43/44をbaseline・A1最大・A2最大・B1・B4へ。重複条件を除く前回案のまま未採用 |
| 長時間代表 | baseline・A1最大・A2最大をseed42・独立40kで比較する前回案。max_stepsを変えたresumeを行わない |
| scene補足 | 主B00+B01に対してB02+B03を1条件・共通更新数で比較する前回案。上表には含まない |

新しい品質根拠が無い以上、これらを確定して本実験を開始することは推奨しない。
親の合意記録が必要な受入条件は未完了で、issue全体はRefsのままとする。
学習・大規模生成・本実験・採用資産の変更・checkpoint保持の修正は行っていない。

## 検証と未実行の準備

通常テスト40件、CPU path契約3ケース、旧入力による費用300項目の一致、
失敗集計・setup費用のbyte再現、報告scriptのruff/mypy、knowledge検証は [validation.json](validation.json) に記録する。
準備したQA集計・原画/overlay・動画・費用更新のコードは、画像が無いため実データでは実行していない。
実行したcollector jobも存在しない。新しいQA指標や比較図を生成済みと扱わない。
