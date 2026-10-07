# ボールデータセット統計

CPUで注釈・pose・区間選択を調べる。モデルを実行せず、元データ・学習maskを変更しない。
操作と起動は[レビューUI](../visualization/README.md#データセット統計)。設定値の正本は
[dataset_statistics.yaml](../configs/dataset_statistics.yaml)。公開APIは`__init__.py`。

## 構成と入力の固定

- `inputs.py`: BallFrameStore・原注釈・PlayerPoseStoreの読込。原注釈はproject root内に限定し、snapshotのhashと一致した版だけ使う。poseのartifact/review hash・PTS・frame・配列shape・人物ID・観測maskを検証する。
- `contracts.py`: 検証済みclip入力、単位・有効対象数付きサンプル、分子/分母付き割合。
- `scopes.py`: 全clip、プレイ候補、採用窓被覆、その補集合。採用範囲は`scope_stride`で決め、比較する全strideの窓は別集計する。
- `metrics/`: 以下の指標の計算。Web/ファイルI/Oに依存しない。
- `summaries.py`: finiteな元サンプルから平均・中央値・P5・P95・min/max。分位点はNumPyのlinear方式。有効数0はnull。
- `aggregation.py`: 元サンプルを結合した全体分布と、clipの件数・割合・平均/中央値/分位点/min/maxそれぞれのclip間分布を分ける。全体割合は分子・分母を合算し、clip割合平均で代用しない。
- `pipeline.py`: clip、全体、source、split、source×splitの集約。進捗通知。元動画groupのsplit重複を記録する。

結果は`ball_dataset_statistics.v1`。storeのmetadata/index、pose manifest、対象clip集合のhashと全実行設定を記録する。計算終了時にstore/pose manifestの変更を拒否する。通常storeにはposeを結び付けず、pose承認済みdatasetだけが固定snapshotと対応poseを読む。

原本の`notes`・補間端点が取得できない場合は、source形式に項目なし、pathなし、fileなし、root外、hash不一致を区別する。parse/schema不正やpose破損は計算失敗。missingな原本を別版へ差し替えない。原文`issues`と`notes`を保持し、理由の自動分類は行わない。TrackNetの1/2 visibilityを統一storeのobservedから復元しない。

## 範囲と連続性

`clip`は全表示frame、`play`は候補、`selected`は採用窓の一意frame、`excluded`は非採用frame。参照frameはclipには含まれるが採用には入らない。各範囲の分母は明示する。

区間選択は既存`PlayIntervalConfig`/`infer_play_intervals`をそのまま使用する。統計用の境界は注釈のsegment_break、typed hit/bounce、担当/参照の切替、大きいPTS gap。`segment_break`をシーンカットの確定ラベルと解釈しない。未申告カットは検出できない。

速度は同じtrackの隣接frameが同じ範囲に含まれ、境界を跨がない場合だけ計算する。離れた採用区間や欠損の前後を結ばない。加速度・突出は3frame全てを要求する。

frame時間は次のPTSまでの差。大きいPTS gapと末尾frameだけはnominal FPSの1frame分とし、その仮定を結果に残す。窓の時間幅は末尾PTS−先頭PTSであり、frame時間の和とは別。

## 指標

### 注釈・欠損・補間

- `annotations`: point_kindの球単位・frame単位の割合、reviewed/target/reference、座標あり、教師有効、遮蔽と位置の有無、複数球、原注釈取得率。notes率は原本を読め、frame単位notes欄を提供する形式だけが分母。Meijiのclip単位annotation_notesは別に保持する。
- `gaps`: 座標欠損＝有限座標の球が0個、実測欠損＝observedラベルの球が0個。補間は座標欠損に入らず実測欠損に入る。生の連続列と境界で区切った列を両方持つ。件数・frame長・秒数と全半開区間を保持する。
- `interpolation`: 補間ラベル連続列の長さ・時間。元JSONが得られるChat/Meijiは重複しない端点pairの距離・PTS差も計算する。scopeに一部だけ含まれる補間は連続列をscopeで切り、端点pairは元の支援範囲を保持する。

### 位置・動き

単一球・確認済みのframeを使い、observedだけと推定位置も含むlocatedを分離する。複数球を任意に選ばない。

- `spatial`: sourceのW−1/H−1によるu/v、P95−P5、端までの距離、格子occupancy。clipの幅はclip記述量であり、全体軌道幅と混同しない。端点正規化で0〜1を外れた点は件数を明示する。
- `motion`: source px/frame pair、source px/s、正規化xy/s、水平/垂直速度、clip内累積移動の垂直割合、加速度、方向角変化。正規化xyは`hypot(Δu,Δv)`。元画像の対角長で割った値ではない。
- 格子ごとの移動量、速度和、edge数、方向和と領域間遷移の非ゼロ要素`[出発cell, 到着cell, 件数]`を保存する。平均速度は速度和/edge数。カメラ位置の確定推定はしない。

### 窓・stride

`windows`は32実frame、開始間隔だけを変える。区間先頭を起点とし、末尾の実窓をbackfillする既存選択を全strideで使用する。教師条件は同一。

窓数、一意被覆、延べ入力/教師frame、frameごとの登場回数、教師数、補間数、最大座標欠損、PTS時間幅、実測速度、境界を含む窓、pose欠損/飛び候補、pose速度閾値ごとの影響窓率を計算する。窓内の各位置の教師・欠損数も保持し、MDD先頭位置0を明示する。窓の欠損長は入力そのものの生列で、境界の有無を別に残す。実際のMDD画像信号やencoder特徴は今回の統計対象ではない。

### pose

欠損maskと設定した最小関節スコアを使用する。スコアは確率ではなく1を超えても保持する。人物サイズはclip内の観測bbox高さの中央値で固定し、人物が未観測なら値を作らない。

関節ごとの速度・人物全体の中央値移動を引いた速度・実PTSに基づく前後線形位置からの突出、骨格長の急変、左右交換で対応距離がどれだけ小さくなるか、人物全体の同時ジャンプを測る。これらは誤推定の確定ラベルではない。

候補flagは主速度閾値超過（生/相対）または突出閾値超過。骨格・左右交換は別指標。速度閾値sweepでは関節、手首、frame、人物全関節、全人物を失う割合と窓影響を調べる。実データを除去・補完しない。

欠損復帰の変位は隣接frameの速度と分離し、両端でスコア条件を満たす共通関節だけで計算する。共通の有効関節がなければ復帰変位のサンプルを作らない。player IDはclipローカルで、player_slot別の集約は同一人物の集計ではない。関節の生スコア・有効数・候補frameへの導線を残す。

## Web APIと結果

Ball reviewにtask専用router/assetsを登録する。Court/inferenceには統計入口を追加しない。

- `GET /api/statistics/config`: 明示的な初期設定。
- `POST /api/statistics/jobs`: catalog内dataset IDと全設定。CPU計算はサーバー内で同時1件。
- `GET /api/statistics/jobs/{id}`: 進捗/完了/失敗。失敗時に古い結果を表示しない。
- `GET /api/statistics/jobs/{id}/result`: 全体・群別集計と軽量clip一覧。
- `GET /api/statistics/jobs/{id}/clip?scene=...`: 同じsnapshotのclip明細、原文、確認候補。

最新jobだけをメモリに保持する。新jobで以前の結果は失効し、APIは404を返す。元サンプルを使った正確な分位点計算のため、大きなdatasetほどCPU時間とメモリが増える。WebUIから集計JSONをダウンロードできる。ソース管理に生成レポートや画像を追加しない。
