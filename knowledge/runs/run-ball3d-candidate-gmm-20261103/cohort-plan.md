# cross-1: 保存済みLOVO予測の可用性による誤差分解

対象はレビューrun `parent-1554-5647a023173b48feac926b069d0a207c`、HEAD
`1f6b9752940fa6ad1350d7ac5001cab119d9e389` のGMM 4条件。方式・設定・src・基盤採点は変更しない。

## 実装前の検証設計

- 契約: v2の目録SHA、固定bundleとGTの既知SHAを照合し、`load_run` と
  `error_samples(run, bundle, annotations)` だけで誤差点を得る。キーは
  `(held-out camera ID, source frame)`、共通点は同じ参照kindを持つ。
  区間は[0,505)、[505,1010)、[0,1010)。各cameraとobserved/estimatedも分ける。
- 検出したい故障: frameだけでjoinして別cameraを混ぜる、片方の点順で対応させる、
  境界505を前半へ含める、>20pxを>=20pxにする、重複で点を水増しする、空集合を0pxと書く。
- 層: 比較専用scriptの集合演算を軽いunitで検証する。非対称camera/frame、
  手計算可能な誤差[11,31]・[20,41]・[5,9]を使う。scriptはこのrun内に置く。
- 既存nodeの再利用:
  `tests/unit/tennis_scene/evaluation/ball3d/test_scoring.py::test_lovo_heldout_pixels_missingness_and_time_splits_match_hand_calculation`
  と `::test_common_lovo_intersection_compares_identical_camera_frame_pairs` を実行する。
  既存は追加/喪失集合の分布を検証しないため、script向けの3件を追加する。
- コスト: 新規3case、各5秒・setup1秒以内、実データ不要、`-n 0`。
  `--collect-only` と `--durations=0 --durations-min=0` で実測する。
  削除・配置変更はない。故障注入はcameraをキーから落とす変異で確認する。

## 実データでの独立確認

v2の同じ保存baselineを使い、各条件/区間/camera/kindで
`共通+方式のみ=方式全採点数`、`共通+baselineのみ=baseline全採点数` を検算する。
camera別・kind別・前後半の和も検算し、不一致は例外で止める。
全点の件数と分布は既存v2の `scores.json` と照合する。
点ごとのcamera/frame/kind、方式・baselineの誤差と集合名もCSVへ残す。
新しい `candidate_gmm_cohort_v1/` にJSON/CSV/MarkdownとSHA目録を保存し、v2は変更しない。
knowledgeノードとPRには主評価の4条件×共通(双方)/追加/喪失を示す。
