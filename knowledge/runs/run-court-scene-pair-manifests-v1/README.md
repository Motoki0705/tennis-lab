# 新5ペアと4シーン共通評価のmanifest

#1581 / #1583の入力作成run。既存の`court_scene_diversity_6000_v1`から、
再レンダリングせずに`court_dataset_subset_v1`を作る。
結果と制約は[run node](../../nodes/synthetic_data_generation/000029-run-court-scene-pair-manifests-v1.md)、
照合の数値は本bundleのJSONを正本とする。

## 選択と検証

`manifests.py`は[元の生成script](../run-court-scene-diversity-6000-v1/generate.py)の
`nested_order`関数のASTだけを実行する。元script全体のimportは、常設srcから除いた
実験固有の生成APIを要求するため実行しない。順位の計算方法は複製しない。
`finalize`と同様、固定済みtrain/val軌道で採択画像を分け、scene順に順位のprefixを連結する。
最初の`check-p01`でB00+B01のsample列を既存scene-2と完全照合し、不一致なら作成を止める。
`create`でも同じgateを通り、既存8 manifestのprefixと新11 manifestの件数・一意性・
順序を検査する。共通評価は各sceneのval pool全体を同じ順位で連結する。
全9 trainに対するsample、trajectory、学習consumerのtrajectory groupの重複を検査する。

`transfer.py`はtennis-drive CLIのinventoryで同名pathの曖昧さとDrive ID・内容を確認する。
CLIに選択的な転送オプションがないため、同skillの`RcloneBackend`から、
必要pathに対応する確認済みDrive IDを`backend copyid --immutable`でローカルへ取得する。
深い保存先の名前探索を避けるため、[rcloneのcopyid](https://rclone.org/drive/#copyid)を使う。
最大16組の重ならないI/Oバッチで取得し、全バッチの失敗を収集して伝播する。
公開は必要pathを列挙した`copy --files-from-raw --immutable --no-traverse`で行い、
いずれもinventoryの内容digestで全転送ファイルを照合する。
再開時は取得済みの各ファイルも内容照合し、破損があれば停止する。
入力はpublication、generation receipt、軌道割当、既存8 subset、4 membership、
452 storeのdescriptorと圧縮indexだけ。JPEGを取得せず、indexもstore単位で読む。
アップロード前に独立した`verify`の成功と各manifestの内容を照合する。
既存名を拒否し、前後のinventoryですべての既存entryのID・型・サイズ・内容・更新日時が
同一で、追加が新manifest11個だけであることを確認する。

## 後続の学習・評価へのインターフェース

保存先はすべて`gdrive:tennis_lab/data/court_scene_diversity_6000_v1/`。
Colabでは既存pool全体をDATA rootの`court_scene_diversity_6000_v1/`へ取得し、
以下のファイルを同じディレクトリに置く。manifestのsource参照は既存publicationへの同階層相対pathである。

| 条件 | train | val |
|---|---|---|
| P02 | `pair-b00-b02.train.subset.json` | `pair-b00-b02.val.subset.json` |
| P03 | `pair-b00-b03.train.subset.json` | `pair-b00-b03.val.subset.json` |
| P12 | `pair-b01-b02.train.subset.json` | `pair-b01-b02.val.subset.json` |
| P13 | `pair-b01-b03.train.subset.json` | `pair-b01-b03.val.subset.json` |
| P23 | `pair-b02-b03.train.subset.json` | `pair-b02-b03.val.subset.json` |

学習sourceの指定例（P02。その他の学習設定は既存scene-2と同一）:

```text
data/source=synthetic_court_dataset_subset
data.source.manifest=court_scene_diversity_6000_v1/pair-b00-b02.train.subset.json
data.source.validation_manifest=court_scene_diversity_6000_v1/pair-b00-b02.val.subset.json
```

各ペアはtrain 5,500枚（各scene 2,750）、val 500枚（各scene 250）。
P01は既存`scene-2.{train,val}.subset.json`をそのまま使う。
共通評価は`court_scene_diversity_6000_v1/eval-4scene.val.subset.json`で、
`split: val`、2,439枚（B00 1,056 / B01 643 / B02 462 / B03 278）。
評価入口の指定オプションは#1582のPRを参照する。
JSONは`schema, source_manifest, source_manifest_sha256, selection_seed, split, samples`を持ち、
`samples`は`{scene_id, sample_id}`の順序付きリスト。seedは42で、IDにはscene prefixを付けない。

## 再現と軽量テスト

認証済みrclone設定を`RCLONE_CONFIG`で明示し、新しいgit管理外の絶対pathを
`PAIR_WORK_ROOT`に指定して、repo rootから`bash knowledge/runs/run-court-scene-pair-manifests-v1/repro.sh`を実行する。
初回の公開時だけ`PAIR_PUBLISH=1`を指定する。既存の同名ファイルには上書きしない。
選択済みmanifestは保存基準のサイズを超えるためGitへ複製せず、Driveに保持する。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest -n 0 \
  knowledge/runs/run-court-scene-pair-manifests-v1/test_manifests.py \
  tests/integration/tasks/court_detection/test_dataset_subset.py::test_subset_preserves_order_and_supervision \
  tests/integration/tasks/court_detection/test_dataset_subset.py::test_subset_rejects_invalid_membership \
  tests/integration/tasks/court_detection/test_dataset_subset.py::test_subset_rejects_disjoint_frames_on_the_same_trajectory \
  --durations=0 --durations-min=0
```

bundle内の新規15件は、このrun固有の件数とscene境界、P01 gate、prefix・重複の拒否、
異なるsampleの軌道漏れ、sceneを含むidentity、既存出力の保護、公開前の検証済み内容との一致、
取得の再開時の破損拒否、全I/Oバッチの失敗伝播、Drive CLIの失敗理由の保持を検査する。
既存9件は小さなCPU fixtureでconsumerの順序・教師・入力拒否・軌道漏れを検査する。
`verify`は実際の新11 manifestを同階層に配置し、実consumerの`read_subset`も通す。
画像を伴う全pool reader・学習・GPU・Blender・Colabの実行は、この入力作成runには含めない。

## 基底PRと全pool読込の確認範囲

[作成開始時のCPU再現](consumer-compatibility.json)では、実Drive publicationのmethod
`saved_train_trajectories`を当時のmainの`TrajectoryMethod`が拒否した。
親の指示に従い、この回帰を修復する[PR #1594](https://github.com/Motoki0705/tennis-lab/pull/1594)の
指定headを基底として本branchを積み直した。
[enum復元の確認](consumer-on-reader-base.json)と
[基底上のmanifest再照合](manifest-verification-on-reader-base.json)を保存している。

`read_subset`で確認したのはschema・source参照・配置であり、画像shardを必要とする
全publication readerの成功とは区別する。#1591で旧学習コードと修復版の両方に
ラップトップの数値環境による幾何の末尾桁差が観測されている。
全poolの厳密な読込は、[親の指示](https://github.com/Motoki0705/tennis-lab/issues/1581#issuecomment-6534951394)に従い、
担当親が前回と同じColab環境で実行手順の最初に確認する。
画像placeholder、比較の緩和、metadata-only consumerモードは追加しない。
