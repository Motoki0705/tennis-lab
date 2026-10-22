---
id: run-ball-all-frame-protocol-20261022
type: run
task: ball_detection
sequence: 63
recorded_at: '2026-10-22'
title: 全フレーム教師の共通区間・更新予算・不在評価のCPU調査
issue: 1369
provider: codex
date: '2026-10-22'
status: done
config:
  frame_step: 1
  lengths: [32, 64, 128]
  common: all safe clip spans of length at least 128
  supervision: all input frames with existing teacher masks
  training_proposal: shared 128-frame packs, stride16 plus tail, at least one known teacher
  evaluation_proposal: stride T/2 plus tail, nearest input center then earliest start
  frames_per_update_proposal: 128
metrics:
  train_observed_coverage32: 0.9958164455281635
  validation_observed_coverage32: 0.9948496951860416
  test_observed_coverage32: 0.997113365990203
  train_expected_observed_per_128_frame_update: 97.17628964286813
  train_expected_negative_per_128_frame_update: 22.55763019471531
  validation_absence_frames: 31885
  test_absence_frames: 35706
artifacts:
  run_dir: knowledge/runs/run-ball-all-frame-protocol-20261022
parents:
- run-ball-temporal-target-survey-20261022
relations:
- to: run-ball-temporal-target-survey-20261022
  rel: supersedes
papers: []
tags: []
repro:
  commit: 7db9bceb2fc4d17e42c928fec1c4b7d07514196b
  command: SOURCE_REPO_ROOT=<source-repo> bash knowledge/runs/run-ball-all-frame-protocol-20261022/repro.sh
---

## 結論と状態

CPU再調査を完了した。2026-10-22の[親issue #1367](https://github.com/Motoki0705/tennis-lab/issues/1367)の決定
（clip内文脈・全フレーム教師・T上限128・不在評価の追加）に従い、中央16案を置き換える規則を提案する。
**共通区間・窓採用・評価割当・不在集合の重複・128枚/更新はユーザー確認待ち**であり、#1369の完了とはしない。
正本は本runの[改訂protocol](../../runs/run-ball-all-frame-protocol-20261022/protocol.md)。

初回runの事実・スクリプト・protocolは保った。
今回の変更は測定誤りの訂正ではなく、教師領域・集合identity・抽出単位・較正母集団の変更なので、
knowledgeの「1 run = 1ノード」に従い**新run**とした。`supersedes`は規則案の置き換えを示し、初回観測の否定ではない。
画像の精度、GPU速度、採用モデルの判断は更新していない。

## 共通区間・coverage

同じ固定snapshot（全1,800 clips・1,065,949 frames）と旧32窓manifestを読む。
条件(b)の連続区間で128枚を置けるものを**区間全体**として採り、旧play/anchorとの重なりは追加条件にしない案。
全Tがこの範囲だけを使い、ラベルのある全frameを一度ずつ採点する。

| split | 共通区間 / clips | 元動画groups | 全frame | observed | reviewed_empty | out_of_frame | mask | 旧32位置教師coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1,342 / 1,342 | 254 | 664,349 | 491,380 | 109,403 | 14,755 | 48,811 | 99.58% |
| validation | 195 / 195 | 33 | 128,668 | 85,473 | 29,173 | 2,712 | 11,310 | 99.48% |
| test | 225 / 225 | 33 | 143,074 | 91,496 | 33,348 | 2,358 | 15,872 | 99.71% |

coverageは共通集合と旧32窓の**集合積**を分子とする。
分子/分母はtrain 488,440/490,492、validation 85,185/85,626、test 91,192/91,456。
旧窓の外から加わるobservedは別に2,940/288/304枚。新observed総数/旧総数をcoverageとは呼ばない。
初回の中央16・128文脈による約76〜81%と今回の約99.5%は、教師領域とanchor制約を変更した異なる集合の値。
同じ集合の改善として解釈しない。

| split/source | 全frame | observed | empty / out | mask | 旧32位置教師coverage | groups |
|---|---:|---:|---:|---:|---:|---:|
| train / Chat | 624,531 | 456,224 | 108,841 / 14,755 | 44,711 | 100% | 245 |
| train / Meiji | 25,905 | 21,871 | 0 / 0 | 4,034 | 100% | 1 |
| train / TrackNet | 13,913 | 13,285 | 562 / 0 | 66 | 86.61% | 8 |
| validation / Chat | 100,575 | 61,369 | 29,143 / 2,712 | 7,351 | 100% | 31 |
| validation / Meiji | 26,961 | 23,007 | 0 / 0 | 3,954 | 100% | 1 |
| validation / TrackNet | 1,132 | 1,097 | 30 / 0 | 5 | 71.33% | 1 |
| test / Chat | 105,246 | 60,873 | 33,343 / 2,358 | 8,672 | 100% | 31 |
| test / Meiji | 36,006 | 28,806 | 0 / 0 | 7,200 | 100% | 1 |
| test / TrackNet | 1,822 | 1,817 | 5 / 0 | 0 | 87.31% | 1 |

除かれる旧observedは全てTrackNetで、native 30 FPSの保持率はtrain/validation/testで
95.95%/84.75%/95.29%、それ以外のnative FPSは100%。低FPSを一括で落とす規則にはしていないが、
TrackNetの短clipは依然として評価範囲外になる。
Meijiは各splitに元動画1 groupしかなく、validation/testの位置教師の約27%/31%を占める。
source/group/FPSの構成が十分に独立・均等になったわけではない。
正確なnative FPS・source×FPSの全計数は[source-fps.csv](../../runs/run-ball-all-frame-protocol-20261022/source-fps.csv)、
全元動画groupの計数は同runの`groups-common.csv`等にある（0支持のgroupも残す）。

## 学習窓・入力予算

推奨は128枚の組を開始stride16＋末尾backfillで作り、既知教師1枚以上の組を採るもの。
共通128枚で一度だけ採否を決め、同じ組をT=32なら4窓、64なら2窓、128なら1窓へ分割する。
trainは幾何候補32,815組のうち32,509組を採用する。全maskの306組を除いても、共通区間の既知教師は全て残る。
最小位置教師数は0で、位置教師のない不在主体の組2,031件（6.25%）も自然頻度で残す。
新しい不在oversampling比率の決定ではなく、専用不在artifactを学習samplerへ追加しない。

1更新F=128枚のとき、T別に窓を独立・一様に抽出した案と比較した期待値は次のとおり。
「位置25%」は各T窓でobserved率25%以上を要求した案。全て同じ共通区間内での集計。

| 窓の採用・抽出 | T=32 位置 / 不在 | T=64 位置 / 不在 | T=128 位置 / 不在 |
|---|---:|---:|---:|
| 各Tで既知教師1枚以上を独立抽出 | 96.488 / 24.267 | 96.737 / 23.465 | 97.176 / 22.558 |
| 各Tで位置25%以上を独立抽出 | 115.587 / 4.638 | 112.215 / 7.590 | 107.807 / 11.755 |
| **同じ128枚を分割（推奨）** | **97.176 / 22.558** | **97.176 / 22.558** | **97.176 / 22.558** |

同じ教師率の条件でも、窓長により採択と出現重みが変わる。単に4/2/1窓にしても教師量は揃わない。
独立「既知1枚」では32→128で位置は+0.688枚、不在は−1.709枚（約−7.04%）/更新となる。
共有分割なら、期待値だけでなく同じ組列で実際に入力するframe・教師の多重度も一致する。
推奨案の内訳は位置97.176、empty 19.714、out 2.844、mask 8.266枚/更新。
子窓の全maskはT=32で1,019/130,036出現、64で229/65,018出現、128で0。
これらもforwardして更新全体で損失を正規化する必要があり、子窓単位で捨てる実装は不可。
全split・全案の窓数と期待値は[budget.csv](../../runs/run-ball-all-frame-protocol-20261022/budget.csv)。

F=128を最小の提案値とし、Colab試走で必要なら全条件を共通の128の整数倍にする。
Upre/Upost・LR schedule・GPU時間等の値は #1376 で決める。入力枚数の一致は同FLOPsや同実時間を保証しない。

## 評価割当と区間端

各Tの評価窓は開始stride=T/2＋末尾backfillで全共通frameを覆う。
各frameは窓中心に最も近い予測で一度採点し、同距離は早いstartとする。
採点集合はT間で一致し、端frameを落とさない。区間端距離別の副集計を追加する。

| 区間端からの距離 | train observed | validation observed | test observed |
|---|---:|---:|---:|
| 0枚 | 1,656 | 216 | 235 |
| 1–7枚 | 11,858 | 1,528 | 1,670 |
| 8–15枚 | 14,010 | 1,741 | 1,958 |
| 16–31枚 | 28,994 | 3,705 | 4,202 |
| 32–63枚 | 60,427 | 8,093 | 9,097 |
| 64枚以上 | 374,435 | 70,190 | 74,334 |

距離0では全Tで片側の文脈が0枚になる。
[edge-support.csv](../../runs/run-ball-all-frame-protocol-20261022/edge-support.csv)に位置・empty・out・maskの支持数を残した。
owner窓内のoffset/MDD先頭も記録する規則とし、区間端と窓端の影響を分ける。
自然遮蔽付きobservedは全splitで0のため、自然遮蔽別の精度はこのsnapshotでは推定できない。

## 不在評価専用集合と重複

snapshot全体の確認済み不在から、同じ条件(b)の128枚窓に含められるframeを全て列挙した。
窓全体が不在である必要はなく、対象frameが窓内にあればよい。

| split | snapshot不在 | 評価可能なempty / out | 評価可能な不在合計 | 不在ありclips / groups | 共通集合外 |
|---|---:|---:|---:|---:|---:|
| train（支持数監査用） | 124,233 | 109,403 / 14,755 | 124,158 | 991 / 253 | 0 |
| validation | 31,885 | 29,173 / 2,712 | 31,885 | 133 / 32 | 0 |
| test | 35,763 | 33,348 / 2,358 | 35,706 | 134 / 32 | 0 |

Chatの空ラベルは1,120 clipsの原注釈を照合し、notes付き空frame 14,347枚を除外した（初回と一致）。
短区間等で評価に残らない不在はtrain 75枚・test 57枚で、いずれもreviewed_empty。
validationでは全不在を保持するため、初回の中央16案で0だったemptyも29,173枚を評価できる。

共通区間を「全適格区間」とすると、不在評価Aは共通集合内の不在Nと定義上同一になる。
推奨は**専用artifact・専用レポートとして分け、和集合では一度だけ数える**扱い。
二つの独立した評価集合として水増ししない。新しい共通外frameの追加が必須なら、
共通集合へplay等の条件を追加する再判断が必要なので、今回は変更せず確認事項として返す。
専用scopeは追加の学習samplerではない。train一覧はholdoutにしない。

不在のsource構成は上のsource表のempty/out列と同じ。
Chat / TrackNet / Meijiの不在はtrain 123,596 / 562 / 0、validation 31,855 / 30 / 0、
test 35,701 / 5 / 0。Chatが全splitで99.5%以上を占める。
Meijiの不在FPRは支持数0で`null`、少数TrackNet負例からsource全体の安定した不在性能は主張しない。
全groupの支持数は`groups-absence.csv`、重複と非重複は`groups-absence_overlap.csv`・`groups-absence_exclusive.csv`。

## 改訂プロトコル・再現・限界

主指標は共通observedの全frame平均位置誤差（元画像pixel）を維持する。
共通集合と専用集合で不在kind別FPR、共通observedでFNRを出し、専用不在集合単独のFNRは`null`。
validationの共通observedと重複排除した不在和集合で閾値を決め、固定予算checkpointを主評価にする。
seed・不確実性・更新正規化・時間予算の式を改訂protocolに具体化した。

#1372へは共通区間・学習組の開始列・不在frame列・clip対応表・評価規則を小さいファイルで渡す。
形式と再現コマンドは同runの[README](../../runs/run-ball-all-frame-protocol-20261022/README.md)。
入力・採点集合の照合値はJSON/CSVに保ち、本文へ転記しない。

本計測はCPU 96.49秒・最大RSS約0.97 GiB。
元manifest/index/metadata・全1,800 shard・照合に使った1,120 Chat原注釈の計2,923ファイルで
inode/size/mtime/ctimeが不変、画像payload openは0だった。
旧snapshot/旧32窓の全教師計数も初回runと一致した。
記録版と再実行の24証拠ファイルは実行metadataを除いてbyte単位で一致した。
通常検証はREADMEのCPUテスト・ruff・mypyとknowledge構造/再現path検査で行う。
src・既存data/outputs・checkpointを変更せず、GPU・Colab/Drive・外部への書き込みは行っていない。
