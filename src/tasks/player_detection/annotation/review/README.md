# Player bbox のレビューと教師採用

`PlayerReviewAdapter` / `PlayerReviewServices` / `PlayerReviewWorker` は共通の
[RevisionStore](../../../base/annotation/editing/README.md)、native renderer、worker pool・
service broker・画像入力 telemetry を使う。work 名は `player_bbox_review / v1 / review`。
production composition への登録、tracking/ID review、pose、dataset publisher はこのモジュールに含まない。
実運用のラベル定義・採用値は未設定であり、この実装だけで production を有効にしない。

## 固定方針と状態

`PlayerReviewPolicy` は対象選手の定義、bbox の定義、元予測の一括確認可否、対象 frame の
確認率・解決率、bbox 詳細画像と全画面画像の最低倍率、`observed_only` / 未確認除外を
すべて明示入力で受け取る。数値や可視領域/amodal 等の意味を本番用の既定値にしない。

revision は全 native frame/PTS と対象範囲を保持し、初期状態は未確認・coverage 不明。
検出ゼロを負例にしない。画像確認と対象一覧の完全性を別に保存する。

- 各候補は `player / non_player / duplicate / unresolved`。bbox は実観測・推定・未解決を区別する。
  duplicate は同 frame の残す player を参照する。元候補は全件、確認後の判断に対応付ける。
- 元候補の observation ID は変更せず、追加は `manual_observation_id(clip_id, frame, manual_id)`。
  除去は `non_player` / `duplicate` 等の明示的な判断として残し、追加後の観測 ID も黙って落とさない。
  元予測・古い revision・解消前の note は不変のまま残る。
- `model_confirmed` は元の bbox・score と完全一致し、方針が許可した実観測のみ。
  修正・追加・除去は `manual / score_origin=missing / confidence=0`。
  このゼロは `ScoreOrigin.MISSING` に対応するマスク値であり、追跡時の検出 score として使わない。
  元 confidence は prediction の参照先に残り、変更後 bbox へ移さない。

## 編集と画像証拠

`replace` は frame hash と判断全体を明示し、修正・追加・除去・保留・note 解消を行う。
`confirm_predictions` は全候補が正しい対象選手で漏れもない未確認 frame だけに使う。
元候補が空の frame は `confirm_empty` で別に確認する。候補がある frame を空にするには
各候補を明示的に除去する。いずれも batch 全体の検証後に CAS で現行版を更新する。

完全な対象一覧・確認済み空 target には全画面が必要。実観測/推定 bbox と除去する元 bbox は、
全体が所定倍率の crop 内に必要。全画面の最低倍率と bbox の最低倍率は独立なので、
全画面と拡大 crop を併用できる。renderer の clip・予測・revision 履歴・描画 bbox を照合する。
手動追加した観測の除去も、同じ観測 ID の直近 bbox を revision と編集履歴から取得し、
編集時と採用時に詳細倍率・全域包含を検証する。再追加・再除去では新しい bbox を使う。
過去 bbox がない除去は証拠不足として拒否する。別 worker の過去の画像だけでは代用できない。
画像生成のみでは確認にならず、採用時には各判断をした worker に画像 bytes が実際に入力された
telemetry を照合する。別 attempt の証拠、未提示画像、他 clip、捏造 overlay、古い head を拒否する。

## 教師採用と後続への引継ぎ

`teacher_rows()` が frame ごとに `positive / negative / ignore` と観測 ID を導出する。
対象範囲内で確認済み、対象一覧 complete、未解決・blocking note・推定 bbox がない frame のみ
教師候補になる。対象選手ゼロは負例、それ以外は採用された実観測 bbox を正例にする。
未確認・不完全な対象一覧・未解決・推定を含む frame は全体を除外し、漏れを負例化しない。
前後の context frame も教師から除外する。全対象が除外なら保留し、採用しない。

`prepare_adoption()` は全 attempt の証拠・明示 policy・導出 mask を検証して receipt を保存する。
`seal()` の前までは同じ観測 ID と期待 revision を指定して bbox を修正できる。
後続 ID review は bbox revision と観測 ID を入力として固定し、その後 bbox が変われば
古い ID review を適用しない接続が必要。seal 後の修正は共通 store が拒否するため、
**後続からの修正が残る間に seal しない**ことは将来の composition の責務である。
現 worker は完了時に bbox 教師を seal する単体 adapter であり、ID/pose を含む全工程接続済みとは扱わない。

操作上限と再試行は共通 [review services](../../../base/annotation/workers/review/README.md)、
worker の手順は [worker instructions](worker-instructions.md) を参照。
少数 frame の `review_rows` は現行判断と同じ frame の元候補 ID・bbox・score を返すため、
修正時に全 clip の予測 JSON を worker へ投入する必要はない。
CPU 試験は合成 VFR 動画、実 renderer と合成 rollout を使う。実テニス映像の品質や
実 Codex の確認、production 方針値をこの試験で承認したことにはならない。独立 QA は追加しない。
