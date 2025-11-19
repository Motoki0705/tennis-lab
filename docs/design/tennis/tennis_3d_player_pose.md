

# テニスプレーヤー 3D 位置・ポーズ復元パイプライン設計

## 1. 目的と設計方針

### 1.1 最終ゴール

テニスコート座標系において、プレーヤーの

* **3D 位置**（コート内のどこにいるか）
* **3D ポーズ**（人体＋ラケットの姿勢）

を推定できるパイプラインを構築する。

対象とする入力は：

* 実コートで撮影した **多視点映像**
* 将来的には、カメラパラメータが不明な **単眼ネット動画（YouTube 等）**

### 1.2 コアアイデア

1. **2段階構成**

   * **Stage A：画像 → 2D キーポイント**

     * 既存の 2D 姿勢推定（ViTPose）＋コートキーポイント検出器を使用。
   * **Stage B：2D キーポイント群 → コート座標系 3D**

     * 多視点の 2D キーポイント（人物＋ラケット＋コート）から 3D を復元。
     * カメラの intrinsics / extrinsics は、原則としてモデル入力には使わない。

2. **シミュレーションは「画像なし／jsonのみ」**

   * テニスコート上のプレーヤー動作を 3D でシミュレート。
   * 各仮想カメラに投影し、

     * コート座標系 3D キーポイント
     * 各カメラの画像座標系 2D キーポイント
   * を生成し、**構造化データ（json, npz など）** として保存する。
   * 画像レンダリングは行わず、シミュレーションコストを最小化する。

3. **カメラパラメータ非依存の 3D 推定**

   * モデルの入力には **カメラパラメータ（intrinsics/extrinsics）を渡さない**。
   * 代わりに、各カメラごとの **2D コートキーポイント配置**から、

     * 視点の位置・向き・スケールを暗黙的に学習させる。
   * これにより、カメラ情報が分からないネット動画にも適用可能な設計とする。

4. **モーションソース：3DTennisDS**

   * 人体＋ラケットの 3D モーションは、モーションキャプチャデータセット
     **3DTennisDS（3D Tennis Dataset）** を基盤とする。
   * そこから ViTPose 準拠スケルトン＋ラケットへリターゲットしてシミュレーションに用いる。

---

## 2. 座標系・コート・フェンスの定義

### 2.1 コート座標系

* 原点：コート中心・地面
  `origin = (0, 0, 0)`（ネット中央の真下）
* 軸：

  * x 軸：ネットと平行（プレーヤー視点で左がマイナス、右がプラス）
  * y 軸：ネットに垂直（near 側がマイナス、far 側がプラス）
  * z 軸：鉛直上向き
* 単位：メートル [m]

### 2.2 ITF 規格に基づく寸法定数

* コート長さ：`court_length = 23.77`
* コート半長：`half_length = court_length / 2 = 11.885`
* シングルス幅：`singles_width = 8.23` → `half_singles_width = 4.115`
* ダブルス幅：`doubles_width = 10.97` → `half_doubles_width = 5.485`
* サービスライン距離（ネットから）：`service_line_distance = 6.40`
* ネット高さ（中央）：`net_height_center = 0.914`
* ネット高さ（ポスト付近）：`net_height_post = 1.07`
* ネットポスト横オフセット：`net_post_offset_x = 0.914`（ダブルスサイドラインから外側）

補助記号：

* `xs = half_singles_width = 4.115`
* `xd = half_doubles_width = 5.485`
* `yB = half_length = 11.885`
* `yS = service_line_distance = 6.40`

### 2.3 コート 3D キーポイント（0〜19）

#### 2.3.1 0〜14：コート平面上のキー点（z=0）

| idx | name                             | 3D 座標 (x, y, z)                      |
| --- | -------------------------------- | ------------------------------------ |
| 0   | far doubles corner left          | (-xd, +yB, 0) = (-5.485, +11.885, 0) |
| 1   | far doubles corner right         | (+xd, +yB, 0) = (+5.485, +11.885, 0) |
| 2   | near doubles corner left         | (-xd, -yB, 0) = (-5.485, -11.885, 0) |
| 3   | near doubles corner right        | (+xd, -yB, 0) = (+5.485, -11.885, 0) |
| 4   | far singles corner left          | (-xs, +yB, 0) = (-4.115, +11.885, 0) |
| 5   | near singles corner left         | (-xs, -yB, 0) = (-4.115, -11.885, 0) |
| 6   | far singles corner right         | (+xs, +yB, 0) = (+4.115, +11.885, 0) |
| 7   | near singles corner right        | (+xs, -yB, 0) = (+4.115, -11.885, 0) |
| 8   | far service-line endpoint left   | (-xs, +yS, 0) = (-4.115, +6.40, 0)   |
| 9   | far service-line endpoint right  | (+xs, +yS, 0) = (+4.115, +6.40, 0)   |
| 10  | near service-line endpoint left  | (-xs, -yS, 0) = (-4.115, -6.40, 0)   |
| 11  | near service-line endpoint right | (+xs, -yS, 0) = (+4.115, -6.40, 0)   |
| 12  | far service T                    | (0, +yS, 0) = (0, +6.40, 0)          |
| 13  | near service T                   | (0, -yS, 0) = (0, -6.40, 0)          |
| 14  | net center (ground)              | (0, 0, 0)                            |

#### 2.3.2 15〜19：ネットポール＋センターベルト

* ネットポスト x 座標：

  * 左：`x_post_L = -(xd + net_post_offset_x) = -6.399`
  * 右：`x_post_R = +(xd + net_post_offset_x) = +6.399`

| idx | name                | 3D 座標 (x, y, z)   |
| --- | ------------------- | ----------------- |
| 15  | left net post base  | (-6.399, 0, 0)    |
| 16  | left net post top   | (-6.399, 0, 1.07) |
| 17  | right net post base | (+6.399, 0, 0)    |
| 18  | right net post top  | (+6.399, 0, 1.07) |
| 19  | center strap top    | (0, 0, 0.914)     |

### 2.4 フェンス（外周）定義

ランオフ（クリアランス）とフェンス高さを以下とする：

* ベースライン方向ランオフ：`baseline_clear = 6.40`
* サイドライン方向ランオフ：`sideline_clear = 3.66`
* フェンス高さ：`fence_height = 3.0`

#### 2.4.1 フェンス外接矩形

* x 方向：

  * `x_min = -xd - sideline_clear = -5.485 - 3.66 = -9.145`
  * `x_max = +xd + sideline_clear = +5.485 + 3.66 = +9.145`
* y 方向：

  * `y_min = -yB - baseline_clear = -11.885 - 6.40 = -18.285`
  * `y_max = +yB + baseline_clear = +11.885 + 6.40 = +18.285`

#### 2.4.2 フェンス四隅

* 地面上（z=0）：

  * `fence_bl = (x_min, y_min, 0)`  // back-left
  * `fence_br = (x_max, y_min, 0)`  // back-right
  * `fence_fl = (x_min, y_max, 0)`  // front-left
  * `fence_fr = (x_max, y_max, 0)`  // front-right

* 上端（z=fence_height）：

  * 同じ x, y で z=3.0 の 4 点。

カメラは基本的にこのフェンス上に配置する。

---

## 3. スケルトン仕様：ViTPose 準拠 + ラケット拡張

### 3.1 人物スケルトン：ViTPose（COCO 17 joints）

人物スケルトンは ViTPose の COCO 17 キーポイントを採用する。

| idx | joint name     |
| --- | -------------- |
| 0   | nose           |
| 1   | left_eye       |
| 2   | right_eye      |
| 3   | left_ear       |
| 4   | right_ear      |
| 5   | left_shoulder  |
| 6   | right_shoulder |
| 7   | left_elbow     |
| 8   | right_elbow    |
| 9   | left_wrist     |
| 10  | right_wrist    |
| 11  | left_hip       |
| 12  | right_hip      |
| 13  | left_knee      |
| 14  | right_knee     |
| 15  | left_ankle     |
| 16  | right_ankle    |

* 3D 側では、内部的に `hip_center = (left_hip + right_hip)/2` を腰（pelvis）として扱うことができる。
* 将来的に WholeBody（顔・手・足指まで）へ拡張する余地は残すが、現段階では 17 点。

### 3.2 ラケットスケルトン拡張（3 点）

ラケットは 3 つの代表点で表現する：

| idx | joint name      | 説明                   |
| --- | --------------- | -------------------- |
| 17  | racket_handle   | グリップ末端（手に最も近い点）      |
| 18  | racket_throat   | スロート部（シャフトとフレームの接続部） |
| 19  | racket_head_top | フレーム上側の代表点           |

* 合計：**20 キーポイント（人物 17 + ラケット 3）**
* ラケットは右手（`right_wrist`）にリジッド接続していると仮定（左利き時は左手）。

### 3.3 3DTennisDS → ViTPose スケルトンへのリターゲット

3DTennisDS は 39 マーカー（人体）＋ 7 マーカー（ラケット）で構成されるモーキャプデータである。
これを ViTPose + ラケット 3 点へ写像する方針：

1. **人体**

   * 各 ViTPose joint に対応するマーカー群を選び、
     重心または代表マーカー位置を joint 位置として採用。
   * hip_center（腰）は左右ヒップマーカーの中点から算出。

2. **ラケット**

   * 7 マーカーに対して剛体フィットを行い、ラケットの 6DoF 姿勢（R, t）を推定。
   * ラケットローカル座標で定義した 3 点（handle/throat/head_top）を (R, t) でワールド座標に変換。

この処理により、シミュレーション内部では毎フレーム：

* `player_joints_3d[0..16]`
* `racket_points_3d[17..19]`

がコート座標系で得られる。

### 3.4 2D キーポイントフォーマット（Stage A 出力）

各カメラ `cam_i` に対して、2D では以下の形式に統一する：

```jsonc
"player_keypoints_2d": {
  "joints": [
    [u_0, v_0],  // nose
    ...
    [u_16, v_16] // right_ankle
  ],
  "visibility": [1, 1, ..., 0]
},
"racket_keypoints_2d": {
  "points": [
    [u_17, v_17], // racket_handle
    [u_18, v_18], // racket_throat
    [u_19, v_19]  // racket_head_top
  ],
  "visibility": [1, 1, 0]
}
```

シミュレーション出力も同様のスキーマを用い、
実データ側の ViTPose 出力もこの index に揃える。

---

## 4. シミュレーションベースデータセット構築

### 4.1 シーン単位パラメータ

* カメラ台数：`num_cameras ∈ {4, 6, 8}`（ランダム）
* プレーヤー数：`num_players ∈ {1, 2, ..., 20}`（シーンごとにランダム）
* fps：`fps = 60`
* シーン長：`scene_duration_sec ∈ [5, 20]` 秒

### 4.2 シーン生成の流れ

1. **コート・フェンス 3D ジオメトリ構築**

   * 2章の定義に従い、コート・ネット・ポール・フェンスを生成。
   * コート 3D キーポイント（idx 0〜19）は固定テーブルから取得。

2. **カメラ配置（フェンス上）**

   * フェンスの 4 辺（near/far/left/right）を param t∈[0,1] で表現。
   * 各カメラについて：

     * ランダムに辺を選択。
     * `t ~ Uniform(0, 1)` でその辺上の (x,y) をサンプル。
     * 高さ：`z_cam ~ Uniform(2.5, 3.5)` m
     * 位置：`C_i = (x_edge, y_edge, z_cam)`
     * 向き：コート中心 `(0, 0, 0.5)` を LookAt し、パン・チルトに ±5° の揺らぎ。
   * intrinsics/extrinsics はシミュレーション内部で保持するが、
     **学習モデルには渡さない**。

3. **プレーヤー＋ラケットモーション適用（多アセット配置）**

   * 3DTennisDS から人体＋ラケットの 3D モーションを読み込み、
     ViTPose＋ラケット 3 点スケルトンにリターゲット。
   * 各シーンで `num_players ∈ {1..20}` をサンプルし、その数だけ 3DTennisDS クリップを（重複可で）選択。
   * プレーヤー i ごとに：

     * 初期位置をコート上の候補グリッド（ベースライン/サービスライン周辺やサイドライン外）からサンプルし、他プレーヤーとの最小距離を確保して配置。
     * 進行方向（near→far / far→near）やローカル回転をランダムに設定し、球種（フォア/バック/サーブ/ボレーなど）に応じたクリップを割り当て。
     * 必要に応じて時間オフセットを加え、1 シーン内で複数モーションが重ならないよう調整。
   * こうして最大 20 体まで同時配置されたシーンを生成し、対戦・球出し練習・コーチングなど多様な状況を再現する。

4. **各フレームの 3D キーポイント計算**

   * 時刻 t ごとに：

     * 各プレーヤーの `player_joints_3d`（17点）＋ `racket_points_3d`（3点）を算出。
     * コート 3D キーポイント（0〜19）は固定。

5. **各カメラへの射影（理想値）**

   * 各カメラ i について：

     * 3D 点 → カメラ座標 → 画像座標 (u,v) に射影。
     * 視錐台外や背面の点は visibility=0 とする。
   * 得られるもの：

     * `player_keypoints_2d[i]`
     * `racket_keypoints_2d[i]`
     * `court_keypoints_2d[i]`

6. **ノイズレスシーン書き出し**

   * 射影結果はそのまま（理想的な 2D/3D/visibility）で保存し、シミュレーション段階ではノイズや欠損を付与しない。
   * 1 シーンを 1 ファイル（例：`scene_{id}.json`）として保存し、フレーム配列＋カメラ配列＋プレーヤー配列を含む構造にする。
   * ドメインランダム化は DataLoader 側でオンザフライに適用する（詳細は章5）。

---

## 5. ドメインランダム化設定（数値ベース）

> **方針更新**
> P0 ではシミュレーターがノイズ付き 2D を直接書き出していたが、以降は
> **シーンファイルは常にクリーンな GT を保持し、DataLoader 側がオンザフライでノイズ／欠損を注入する**。
> これにより、同じシーンでもエポックごとに異なるサンプルを得られ、
> ノイズ統計のチューニングが容易になる。

### 5.1 背景

* ViTPose は COCO Keypoints 2017 で AP ≈ 80–82 程度の精度を持つ。
* OKS ベースの AP から、キーポイント誤差は「人物スケールの数 %」レベルと考えられる。
* これを踏まえ、**DataLoader（`src/datasets/tennis_pose.py` 内）でキーポイントにガウスノイズ＋欠損**を与え、
  現実の検出結果に近い分布を作る。

### 5.2 人物 2D キーポイントノイズ（ViTPose 模倣）

各カメラ・各フレームで、まず人物の高さを推定：

* `H_person = max_j v_j - min_j v_j` [px]（y 座標の差）

各関節 j について：

[
(u'_j, v'_j) = (u_j, v_j) + \Delta_j,\quad
\Delta_j \sim \mathcal{N}(0, \sigma_j^2 I_2)
]

* 体幹（肩・腰・首など）：
  `σ_torso = 0.02 * H_person`  （人物高さの 2%）
* 四肢末端（手首・足首）：
  `σ_extremity = 0.03 * H_person`  （3%）
* 頭部（目・耳・鼻）：
  `σ_head = 0.015 * H_person`  （1.5%）

極端な外れ値は 3σ でクリップする（|Δu|,|Δv| > 3σ なら再サンプルなど）。
実装は DataLoader が各サンプルを取り出すたびに `H_person` を推定してノイズを付与し、結果だけを Collate に渡す。

### 5.3 ラケット 2D キーポイントノイズ

ラケットは小さく、検出難度が高い前提で、
ラケットの高さ `H_racket`（ラケット 2D キーポイントの y 差）から：

[
\sigma_{\text{racket}} = 0.04 * H_{\text{racket}}
]

を標準偏差とするガウスノイズを全ラケット点に付与。
DataLoader では人物と同じくオンザフライで `H_racket` を測り、ノイズを合成する。

### 5.4 コート 2D キーポイントノイズ

コートキーポイントもモデル検出結果として扱い、ノイズを付与する。

* 画像高さを `H_img` とすると、
  全コート点に対し：

[
\sigma_{\text{court}} = 0.003 * H_{\text{img}}
]

例：1080p 映像なら σ ≈ 3.24 px。
カメラの高さや距離による違いはまず無視し、一律に設定。
シーンファイルは理想コート座標のみを格納し、DataLoader が出力テンソルへ変換する直前にノイズを注入する。

### 5.5 欠損・visibility のランダム化

実世界では、関節・ラケット・カメラが部分的に見えない場合がある。
これを確率的に模倣：

* 人物末端関節（手首・足首）：

  * `p_missing_extremity ≈ 0.10`（10%）
* 体幹関節（肩・腰・膝など）：

  * `p_missing_torso ≈ 0.02`（2%）
* ラケット点：

  * `p_missing_racket ≈ 0.15`（15%）
* コート点：

  * 通常は `p_missing_court ≈ 0.01`（1%）

実装上は DataLoader 内で乱数を生成し、

* 一様乱数 r < p_missing_x なら

  * `(u_j, v_j) = (NaN, NaN)`
  * `visibility_j = 0`

とする。

### 5.6 カメラ関連のランダム化

* **カメラ位置揺らぎ（シミュレーター側）**

  * フェンス上サンプリングに加え：

    * `Δx ~ Uniform(-0.3, +0.3)` m
    * `Δy ~ Uniform(-0.3, +0.3)` m
* **カメラ姿勢揺らぎ（シミュレーター側）**

  * LookAt(コート中心) + パン・チルトそれぞれ ±5°。
* **カメラドロップアウト（DataLoader側）**

  * 各フレーム・各カメラについて、確率 `p_camera_drop ≈ 0.05` で
    そのカメラの全キーポイントを欠損扱いにする。

---

## 6. 学習モデル構想

### 6.1 Stage B-1：多視点 2D → コート座標系 3D（CourtMVPoseNet）

#### 入力

時刻 t において、各カメラ i から：

* `court_keypoints_2d[i][0..19]` + visibility
* `player_keypoints_2d[i][0..16]`（ViTPose 17 joints）+ visibility
* `racket_keypoints_2d[i][17..19]` + visibility
* カメラ ID（one-hot や学習可能埋め込み）

intrinsics/extrinsics は **入力しない**。

#### 出力

* コート座標系での 3D：

  * `player_joints_3d[0..16]`
  * `racket_points_3d[17..19]`

#### モデル構成イメージ

1. **視点ごとの埋め込み**

   * 各カメラ i の

     * 2D コート 20 点
     * 2D 人物 17 点
     * 2D ラケット 3 点
     * visibility
   * を入力とし、小さな MLP / Transformer encoder で視点埋め込み `f_i` を作る。

2. **マルチビュー統合モジュール**

   * 全視点の `{f_i}` を Self-Attention / GNN で統合し、
     視点数変動やカメラドロップアウトにもロバストな特徴 `F` を得る。

3. **3D 回帰ヘッド**

   * 統合特徴 `F` から 3D ジョイント位置（20点 × 3 次元）を直接回帰。

#### 損失

* 3D 位置損失：

  * シミュレーション GT に対する L2 損失：
    [
    \mathcal{L}_{3D} = \sum_j |\hat{\mathbf{X}}_j - \mathbf{X}_j^{GT}|^2
    ]
* 物理正則化：

  * ボーン長一貫性（フレーム間変動を抑制）
  * 足の z < 0 へのペナルティ
  * ラケットハンドルと手首の距離制約
* （オプション）シミュレーションに限り再投影損失も追加できる。

### 6.2 Stage B-2：単眼 + 時系列 → 3D（CourtMonoTemporal）

#### 目的

* CourtMVPoseNet を実多視点データに適用して得た 3D を「擬似ラベル」とし、
* 単眼視点＋時系列コンテキストだけで 3D を復元できるモデルを学習する。

#### 入力

* あるカメラ k の時間窓 `[t-L, ..., t+L]` の各フレームについて：

  * `court_keypoints_2d[k]`
  * `player_keypoints_2d[k]`
  * `racket_keypoints_2d[k]`
  * visibility

#### 出力

* 各フレームの `player_joints_3d` ＋ `racket_points_3d`。

#### モデル構成

* 時系列 Transformer / TCN / Bi-LSTM など。
* 2D キーポイント系列（＋コート2D）から、3D 時系列を回帰。
* コート 2D 配置から、カメラ幾何情報を暗黙に学習する。

---

## 7. 学習フロー

1. **シミュレーションデータで CourtMVPoseNet をプリトレ**

   * 上記の座標系を用いてクリーンなシーンを大量に合成し、DataLoader 側のノイズ設定で実検出に近い分布へ落とし込む。
   * 3D GT に対して学習し、幾何構造・マルチビュー一貫性を獲得させる。

2. **実多視点データで CourtMVPoseNet を微調整**

   * 実コートで撮影した多視点映像から、

     * ViTPose＋コート検出器で 2D キーポイント抽出。
   * 可能なら三角測量や SfM による粗い 3D を併用して fine-tune。

3. **CourtMVPoseNet を実データに適用 → 擬似 3D ラベル生成**

   * 各シーン・各カメラの 2D シーケンスに対して CourtMVPoseNet を適用し、
     単眼視点に対応する 3D タイムシリーズを生成。

4. **CourtMonoTemporal を学習**

   * 単眼 2D シーケンス → 擬似 3D シーケンスのペアで学習。
   * これにより、カメラパラメータ不明なネット動画からも、
     コート座標系の 3D 位置・ポーズを復元可能にする。
