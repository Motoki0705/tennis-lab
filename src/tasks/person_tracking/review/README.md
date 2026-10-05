# Person Tracking の保存データレビュー

raw ID の観測区間、元画像/crop、選別後の実観測を同じframeで点検する読取専用UI。
追跡・特徴抽出・生成・自動レビューは実行しない。静的アセットはtask内にあり、frontend buildは不要。

## 起動

コードのあるworktreeの直下から実行する。`ROOT`は元repo rootの絶対パス。
`--data-root` / `--artifact-root`と入力パスは絶対パスを要求し、各rootの外へ出るsymlinkも拒否する。

```bash
ROOT=/home/kamimura/projects/tennis-lab
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python" \
  -m src.tasks.person_tracking.scripts.review_dataset \
  --data-root "$ROOT/data" --artifact-root "$ROOT/outputs" \
  --campaign "$ROOT/outputs/chat_annotation/player_pose/ball-mix-v2-20261002" \
  --pose-dataset "$ROOT/data/ball_detection/ball-mix-v2-player-pose-v1" \
  --store "$ROOT/outputs/tennis_scene/evaluate/i964-qualification-r14-20261001/store" \
  --reference "$ROOT/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/annotations/player_association/labels.json" \
  --port 8896
```

`http://127.0.0.1:8896`を開く。`--campaign` / `--pose-dataset`は両方指定する。
componentだけを読む場合はこの2引数を省略できる。`--store` / `--reference`は複数指定可能。
参照labelは指定したstoreのclipに対応するものを明示し、対応clipのないlabelや同じclipへの重複指定は停止する。
公開Ball storeをcampaignのRGBに差し替える引数はない。campaignの固定snapshotだけを読む。

## 操作と表示の意味

- データの役割・source・split・clipを選ぶ。未生成/skipped/旧schemaは理由付きで列挙する。
- box上の`R`はraw tracker ID、`G`は保存GSIの未観測box。boxをクリックすると右のcropが切り替わる。
- slider・frame番号・左右キー・4fps再生で確認する。時刻は保存/復号PTS由来で、FPSから作らない。
- timelineは実観測・保存GSI・box欠損を区別する。クリックすると選択したID/frameへ移動する。
  保存ID区間は下段に表示し、匿名選手IDごとに色を分ける。右の区間ボタンは根拠frameへ移動する。
- 「実観測の途切れ」の前/欠測/後ボタンで連続性を点検する。未保存boxのcropは作らず、ラベルも付けない。
- 「選別後」はBallPoseの採用player観測、またはcomponentの保存コート候補maskだけを表示する。
  コートgroup IDとraw tracker IDを同一視しない。欠損・GSIを選別へ加えない。
- 部分参照boxは別のoverlayとして表示し、raw IDへ自動対応させない。
  labelは保存観測boxの人物レビューで、未検出人物を被覆しない。独立RGB hashがlabelにないことも明記する。

BallPoseのapprovedは匿名選手ID区間の採用であり、bbox/pose全体の精度保証ではない。
raw archiveにはGSI boxが保存されていないため、rawに存在しないboxは補間として描かない。
componentでは`person_tracks v5`の保存reconstruction maskを使う場合だけGSIを表示する。
照合cost・detector scoreがtrack観測に保存されていなければ、値を作らない。

## 入力契約と検証

BallPose readerはcampaign config/planの固定hash、RGB snapshot metadata/indexと選択clipのshard hash、
raw generationのfile hash・clip identity、frame/PTS・寸法を照合する。
採用済みclipではmanifestのraw/review/pose hashと区間の被覆を確認し、元rawを再対応した配列と採用artifactを比較する。
未採用clipは全raw観測を表示し、非選手や空poseへ暗黙変換しない。
生成済みclip一覧はcatalogを開いた時点のsnapshotであり、全clipの品質保証ではない。

component readerはscene indexの採用reference・依存lineage・descriptorと配列hashを照合し、
宣言されたsource動画のhash、復号frame数・寸法・実PTSを確認する。
source動画はdata root内の宣言pathに限定し、別動画を探索して代用しない。
選別maskは同じraw artifactを参照するv2だけを受け付ける。
旧`person_tracks` schemaは履歴として列挙し、現行形式に変換しない。
モデル比較用の旧benchmark NPZはこのUIのreader対象外。方式と既定profileは[task README](../README.md)が正本。

```bash
"$ROOT/.venv/bin/python" -m pytest -n 2 \
  tests/unit/tasks/person_tracking/review/test_review.py --no-cov
```

テストfixtureは契約検査だけに使い、データレビュー画面の証拠には使わない。
