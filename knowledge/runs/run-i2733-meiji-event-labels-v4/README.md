# 明治4カメラ `multiview-v4` ボールイベント詳細再注釈・検証記録（#2733）

明治4カメラ `multiview-v3` 全69クリップ（元注釈 `meiji_event_labels.v1` 全913イベント）を対象に、同期4視点映像を Codex（`gpt-6.2-sol`, `model_reasoning_effort="xhigh"`）で8チャンク並列確認＋境界ケース独立監査（Pass 2）し、詳細ボールイベント注釈 `meiji_event_labels.v2` と `SceneResult v4` ベンチマーク `multiview-v4` を生成・検証した記録。

- 基準入力・出力ハッシュ一覧: [inputs.json](inputs.json)
- 1対1台帳突合・遷移表・境界変更明細: [ledger-reconciliation.json](ledger-reconciliation.json)
- 全69クリップ新旧差分表: [clip-diffs.csv](clip-diffs.csv)
- 採用イベント全620件の元フレーム・根拠台帳: [adopted-events.csv](adopted-events.csv)
- 除外イベント全293件の元フレーム・除外理由台帳: [excluded-events.csv](excluded-events.csv)
- 全69クリップ実データ検証結果: [verification.json](verification.json)
- 再現・検証スクリプト: [repro.sh](repro.sh) / [verify_multiview_v4.py](verify_multiview_v4.py)

---

## 1. 新しいラベル定義とポイント境界規則（`SceneResult v4` / `meiji_event_labels.v2`）

`SceneResult v4` では**検出対象として採用したイベントのみ**をスパース配列へ格納し、`ball_event_in_point` を廃止した。ポイント外および除外イベントは `meiji_event_labels.v2` の `excluded_events` と `event_detection` コンポーネント成果物に理由付きで保持する。旧 `other` と未設定（`None`）コードは v4 で完全に解消した。

### 大分類（`ball_event_kind`, `EVENT_CLASSES_V4`）

| コード | 名称 | 定義 |
| ---: | --- | --- |
| 0 | `shot` | 検出対象となるラケット打球・ポイント終了原因の接触 |
| 1 | `bounce` | 検出対象となるコート面への着地（サービスレット直後の初回着地を含む） |
| 2 | `net` | 検出対象となるネット（白帯・網）接触 |
| 3 | `unknown` | ポイント内の採用イベントのうち**大分類自体を判定できない場合のみ**に使用する専用クラス。細分類（フォア／バック／正面など）の迷いには使用しない |

### 詳細ラベル（`ball_event_subtype`, `EVENT_SUBTYPES_V4`）と判定基準

| コード | 大分類 | 詳細ラベル | 判定基準 |
| ---: | --- | --- | --- |
| 0 | `shot` | `serve_overhead` | トスを上げて頭上・肩上から打つサーブ |
| 1 | `shot` | `serve_underhand` | 腰〜膝付近の低い打点から下から振り上げて打つアンダーハンドサーブ |
| 2 | `shot` | `groundstroke_forehand` | ワンバウンド後（グラウンドストローク）のフォアハンド（利き手側） |
| 3 | `shot` | `groundstroke_backhand` | ワンバウンド後（グラウンドストローク）のバックハンド（非利き手側） |
| 4 | `shot` | `volley_forehand` | ノーバウンド打球（ボレー）でフォアハンドと判断できるもの（フォア／バック優先） |
| 5 | `shot` | `volley_backhand` | ノーバウンド打球（ボレー）でバックハンドと判断できるもの（フォア／バック優先） |
| 6 | `shot` | `volley_center` | フォアまたはバックのどちらとも言えない正面ブロック／頭前上方の短接触ボレー |
| 7 | `shot` | `smash` | ロブ・高弾道球に対するオーバーヘッドスマッシュ（1分類を引き継ぎ） |
| 8 | `net` | `service_let` | サーブがネットに接触して相手コートへ越え、直後にサーブを打ち直すサービスレット |
| 9 | `net` | `net_in` | ラリー中の打球がネットに接触して相手コートへ越えた有効球（返球できず得点となった場合を含む） |
| 10 | `net` | `net_fault` | ネット接触自体がプレー終了の原因になった接触（サーブのネットフォルト、ラリー中のネット失点） |
| 11 | `bounce` | `bounce` | ポイント内のバウンド、および確定した `service_let` 直後の相手側初回バウンド |
| 12 | `unknown` | `unknown` | `ball_event_kind == unknown` に対応する詳細コード（今回69クリップでの該当は0件） |

### ポイント境界・除外規則

1. **ポイント外の球扱い（除外）**: サーブ前の球出し、フォルト確定後の返送・止球・球回収、ポイント終了後の返球やその返球がネットに当たった接触はすべて除外する。
2. **終了原因になった接触（採用）**: プレー終了の原因となった `net_fault` や、同側選手の連続接触などの反則接触は検出対象に含める。一方、`net_fault` の後に打者側コートへ落ちたバウンドは既にポイント終了後のため統一して除外する。
3. **サービスレット直後の初回バウンド（採用）**: 確定した `service_let` の直後に相手コートへ着地する最初の1バウンドは、旧 `in_point` の値にかかわらず必ず `bounce` として採用する。
4. **確度と細分類の分離**: 接触の有無やフレーム位置の曖昧さは既存の `status`（`certain` / `probable` / `uncertain`）と `frame_uncertainty` で表現し、`unknown` に逃がさない。

---

## 2. 実クリップを使った判定例と定義の照合

全12実在クラスおよびポイント境界・旧 `other`・サービスレットの代表例を、4視点映像から確認した元フレーム・根拠と照合した（全913件の明細は [adopted-events.csv](adopted-events.csv)・[excluded-events.csv](excluded-events.csv)）。

| 区分 | クリップ / ID / 元フレーム | 旧ラベル (`in_point`) | 新ラベル / 採否 | 4視点映像からの確認根拠 |
| --- | --- | --- | --- | --- |
| `serve_overhead` | `clip_000` `e001` f=132 | `shot/serve` (`true`) | `shot/serve_overhead`（採用） | cam2・cam3で白帽・灰色シャツの選手がトスし、f132付近で頭上まで腕を伸ばして振り下ろす |
| `serve_underhand` | `clip_039` `e001` f=0 | `shot/serve` (`true`) | `shot/serve_underhand`（採用） | 水色短パンの左利き選手が膝の高さからラケットを下から振り上げ、cam2・cam3で低い打点からネット方向へ発射（全16件が同一選手で一貫） |
| `groundstroke_forehand` | `clip_000` `e003` f=198 | `shot/groundstroke` (`true`) | `shot/groundstroke_forehand`（採用） | cam0のf189–208で左利き選手がバウンド後の球を左腰側から反対肩側へ振り抜く |
| `groundstroke_backhand` | `clip_002` `e019` f=820 | `shot/groundstroke` (`true`) | `shot/groundstroke_backhand`（採用） | cam3のf808–835で右利き選手が胸の左側に構え、片手バックハンドで打ち抜く |
| `volley_forehand` | `clip_001` `e010` f=495 | `shot/volley` (`true`) | `shot/volley_forehand`（採用） | cam0・cam3のf480–510で赤シャツ前衛が右腕を伸ばし右肩上の面でノーバウンド接触 |
| `volley_backhand` | `clip_002` `e023` f=1014 | `shot/volley` (`true`) | `shot/volley_backhand`（採用） | cam0・cam3のf1007–1023で左利き選手が胸の前から右肩側（バック側）で空中球をとらえる |
| `volley_center` | `clip_000` `e004` f=225 | `shot/volley` (`true`) | `shot/volley_center`（採用） | Pass 2監査で確認。cam0・cam2・cam3でf224–226に頭の前上方で短く面を合わせて受けるブロックで、明確なフォア／バック側方スイングがないため正面に分類 |
| `smash` | `clip_002` `e026` f=1144 | `shot/smash` (`true`) | `shot/smash`（採用） | ロブに対する頭上オーバーヘッドスマッシュ（既存1分類をそのまま引き継ぎ） |
| `service_let` | `clip_009` `e002` f=106 | `net/let_cord` (`true`) | `net/service_let`（採用） | cam2のf99–126でサーブ球が白帯接触後に失速して越え、cam0で回収後にf606で同一選手が打ち直し |
| `service_let` 直後初回バウンド | `clip_009` `e003` f=132 | `bounce/None` (`false`) | `bounce/bounce`（採用へ変更） | サービスレット後の相手サービスエリアへの初回着地。旧 `in_point=false` から規則統一で採用（`clip_035` `e003` f=279 も同様に `false→採用`、`clip_012` f=152・`clip_023` f=460 は `true→採用` を維持） |
| `net_in` | `clip_026` `e006` f=281 | `net/let_cord` (`true`) | `net/net_in`（採用） | f253のラリー打球がcam2のf280–282で白帯に当たって越え、f298（通常バウンド採用）とf333で2バウンドして得点 |
| `net_fault` | `clip_000` `e007` f=326 | `net/net_fault` (`true`) | `net/net_fault`（採用） | cam0・cam1でf326付近にネット下部へ当たり越えず、f349（除外）に打者側へ落下 |
| `bounce`（通常） | `clip_000` `e002` f=170 | `bounce/None` (`true`) | `bounce/bounce`（採用） | サーブ球がレシーバー側コート面へ着地し下降から上昇へ反転（機械変換） |
| 旧ポイント中 `other` ① | `clip_035` `e041` f=2045 | `shot/other` (`true`) | `shot/volley_forehand`（採用） | Pass 2監査で確認。前衛接触後に上がった球を同側後衛が地面より上で右前へフォア面を伸ばして捕球したポイント終了原因の連続接触のため採用 |
| 旧ポイント中 `other` ② | `clip_056` `e004` f=262 | `shot/other` (`true`) | 除外（`excluded_events`） | Pass 2監査で確認。第1サーブ `e001`（f115）が `e002`（f197）でフォルト着地し、同一サーバーがアドサイドからf488で第2サーブを打ち直し。f221返送・f262前衛止球・f268落下バウンドはいずれもフォルト後の球扱いのため除外 |
| フォルト後返送列（除外） | `clip_016` `e003`–`e007` f=160–307 | `shot`/`bounce`/`net` (`true`) | 除外（`excluded_events`） | Pass 2監査で確認。第1サーブ `e001`（f96）が `e002`（f135）でロングフォルトとなりf675で第2サーブ。その間の返送打球・バウンド・ネット接触（f282）を除外 |
| `net_fault` 後落下バウンド（除外） | `clip_011` `e003` f=188 等15件 | `bounce/None` (`true`) | 除外（`excluded_events`） | 旧注釈では `net_fault` 後落下バウンドの `in_point` が15件 `true`・12件 `false` と混在していたため、ポイント終了後の自陣落下としてすべて除外に統一 |

---

## 3. Codex 確認対象台帳と元イベントの1対1照合

`multiview-v3/manifest.json`（SHA-256: `790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a`）と `clip_000`–`clip_068` の v1 注釈から凍結台帳（全69クリップ・全913イベント）を構築し、全件を役割別に仕分けた。

| 台帳区分 (`review_role`) | 件数 | 内訳・処理方式 |
| --- | ---: | --- |
| `primary_review`（Codex 初期確認対象） | 355 | サーブ94、グラウンドストローク183、ボレー44、ネット接触32（`net_fault` 27・`let_cord` 5）、ポイント中 `other` 2（`clip_035` f2045、`clip_056` f262）。8チャンク（各43–45件）並列で Codex `gpt-6.2-sol` (`xhigh`) が4視点映像を判定 |
| `smash_carryover` | 11 | 既存スマッシュ1分類をそのまま `shot/smash` へ引き継ぎ |
| `bounce_in_point` | 274 | 機械的に `bounce/bounce` 変換する候補。うち境界確認で `net_fault` 後落下バウンド15件とフォルト後返送列（`clip_016` 2件・`clip_056` 1件）の計17件を除外し257件を採用（+1件の `clip_016` f307 はネット後落下） |
| `out_of_point` | 273 | 原則除外。うちサービスレット直後の初回バウンド2件（`clip_009` f132、`clip_035` f279）を境界規則統一で採用に変更 |
| **合計** | **913** | **採用 620件 + 除外 293件 = 913件（重複0、未処理0、元クリップID・元イベントID・0始まり元フレーム番号と1対1完全一致）** |

### 対象の増減理由（`v1 in_point = 640` → `v2 adopted = 620`、差 `-20`）

- **増加（+2件）**: サービスレット直後の初回バウンドで旧 `in_point=false` だった2件を採用へ変更（`clip_009 e003@132`、`clip_035 e003@279`）。既採用だった2件（`clip_012 e003@152`、`clip_023 e008@460`）と合わせ、サービスレット4件すべての初回バウンドを統一採用した。
- **減少（-22件）**:
  - `net_fault` 直後に打者側コートへ落ちたバウンドの統一除外: **-14件**（`clip_011`, `013`, `014`, `015`, `019`, `050`, `055`, `056`, `057`, `058`, `060`, `061`, `063`, `064`）
  - `clip_016` 第1サーブ・ロングフォルト後の返送列除外（Pass 2 監査）: **-5件**（`e003@160` 打球, `e004@235` バウンド, `e005@259` 打球, `e006@282` ネット接触, `e007@307` 落下バウンド）
  - `clip_056` 第1サーブ・フォルト後の返送・止球列除外（Pass 2 監査）: **-3件**（`e003@221` 打球, `e004@262` 旧 `other` 止球, `e005@268` バウンド）

### 旧ラベル → 新ラベル 全913件遷移表

| 旧 (`kind / subtype / in_point`) | 新（採用 / 除外） | 件数 |
| --- | --- | ---: |
| `shot / serve / true` | 採用 `shot / serve_overhead` | 78 |
| `shot / serve / true` | 採用 `shot / serve_underhand` | 16 |
| `shot / groundstroke / true` | 採用 `shot / groundstroke_forehand` | 156 |
| `shot / groundstroke / true` | 採用 `shot / groundstroke_backhand` | 24 |
| `shot / groundstroke / true` | 除外（フォルト後返送: `clip_016`×2, `clip_056`×1） | 3 |
| `shot / volley / true` | 採用 `shot / volley_forehand` | 24 |
| `shot / volley / true` | 採用 `shot / volley_backhand` | 19 |
| `shot / volley / true` | 採用 `shot / volley_center` | 1 |
| `shot / smash / true` | 採用 `shot / smash` | 11 |
| `shot / other / true` | 採用 `shot / volley_forehand`（`clip_035` f2045） | 1 |
| `shot / other / true` | 除外（`clip_056` f262） | 1 |
| `net / let_cord / true` | 採用 `net / service_let` | 4 |
| `net / let_cord / true` | 採用 `net / net_in`（`clip_026` f281） | 1 |
| `net / net_fault / true` | 採用 `net / net_fault` | 26 |
| `net / net_fault / true` | 除外（`clip_016` f282 フォルト後返送列） | 1 |
| `bounce / none / true` | 採用 `bounce / bounce` | 257 |
| `bounce / none / true` | 除外（`net_fault` 後落下14件 + フォルト後3件） | 17 |
| `bounce / none / false` | 採用 `bounce / bounce`（サービスレット初回着地: `clip_009`, `clip_035`） | 2 |
| `bounce / none / false` | 除外 | 181 |
| `shot / groundstroke / false` | 除外 | 9 |
| `shot / other / false` | 除外 | 78 |
| `net / net_fault / false` | 除外 | 1 |
| `net / other / false` | 除外 | 2 |
| **合計** | **採用 620 / 除外 293** | **913** |

---

## 4. 種別別集計・split 構成・全69クリップ検証結果

### 採用イベント集計（全620件）

- **大分類**: `shot` 330 / `bounce` 259 / `net` 31 / `unknown` 0
- **詳細分類**:
  - `serve_overhead`: 78 / `serve_underhand`: 16
  - `groundstroke_forehand`: 156 / `groundstroke_backhand`: 24
  - `volley_forehand`: 25 / `volley_backhand`: 19 / `volley_center`: 1
  - `smash`: 11
  - `service_let`: 4 / `net_in`: 1 / `net_fault`: 26
  - `bounce`: 259 / `unknown`: 0
- **確度 (`status`)**: `certain` 595 / `probable` 24 / `uncertain` 1
- **判定経路 (`review_source`)**: `codex_visual_review` 350 / `mechanical_bounce` 255 / `carryover_smash` 11 / `service_let_first_bounce` 4
- **Split 分布**（`multiview-v3` の固定47/11/11 split・seed 42・独立テスト6本を完全継承）:
  - `train`: 47クリップ / 414イベント（`shot` 219 / `bounce` 174 / `net` 21）
  - `val`: 11クリップ / 111イベント（`shot` 60 / `bounce` 45 / `net` 6）
  - `test`: 11クリップ / 95イベント（`shot` 51 / `bounce` 40 / `net` 4）

### 全69クリップ実データ検証結果（[verification.json](verification.json)）

[verify_multiview_v4.py](verify_multiview_v4.py) により生成済み `multiview-v4` の全69クリップを独立検証し、以下すべてを通過した。

1. **`SHA256SUMS` 完全検証**: 全21,341エントリ（`multiview-v3` ハードリンク共有 18,786件 + `multiview-v4` 新規生成・再ハッシュ 2,555件）で重複0件・欠損0件・ハッシュ不一致0件。
2. **既存非イベント配列の byte-for-byte 一致**: 全69クリップで `verify_replayed_scene(..., expected_version=4)` が成功し、`scene.npz` 内の全非イベント配列（カメラ・コート・人物3D・SMPL・ボール3D等）が不変の v2 `scene_assembly` とバイト単位で完全一致することを確認。
3. **ラベル読み戻し・元フレーム一致・禁止コード解消**: 全69クリップの `scene.npz` が `scene_schema_version == 4`、`ball_event_in_point is None` を満たし、`EVENT_CLASSES_V4` / `EVENT_SUBTYPES_V4` の列挙順で読み戻した全620イベントが v2 注釈および v1 元フレームと100%一致（`other`・`None` は0件）。
4. **サービスレット初回バウンド・旧 `other`・下流読込**: 4件のサービスレット初回バウンド（`clip_009@132`, `clip_012@152`, `clip_023@460`, `clip_035@279`）、`clip_026` の `net_in` (`@281`) + バウンド (`@298`)、`clip_035@2045`（採用 `volley_forehand`）、`clip_056@262`（除外）、`clip_000@225`（`volley_center`）、および `src/tasks/ball_refiner_3d/post_training` の `MeijiBenchmark.load`・`Annotations.from_scene` による全69クリップ読み戻しがすべて一致した。

全69クリップのクリップ別新旧差分（`v1_total`, `v1_in_point`, `v2_adopted`, `v2_excluded`, 増減数、全13詳細ラベル件数）は [clip-diffs.csv](clip-diffs.csv) に収録した。
