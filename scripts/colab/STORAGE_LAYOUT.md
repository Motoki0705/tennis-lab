# Colab / Drive storage layout

この文書を、Colabで使う入力と学習成果物の配置に関する正本とする。jobごとの正確な
入出力は `workflows/jobs/*.toml` が機械可読な正本であり、この文書へ同じ一覧を複製
しない。

## 固定契約と育てる規約

このstorage layoutは完成した固定規格ではない。データ種別や実験運用が増えたときは、
既存分類へ無理に押し込めず、利用者またはrepository maintainerの監修を受けて構造を
拡張し、この文書と該当job manifestを同じPRで更新する。ここにある推奨パスを未知の
artifactへ機械的に当てはめてはならない。

一方、次は変更時にmigrationが必要な固定契約である。

- 学習中の書き込み先はColab VMのlocal filesystemとし、LightningModuleは保存先や
  rcloneを知らない。
- `BaseTrainingRunner`がlocal output rootを所有し、選択された`ArtifactStore`へ同期する。
- Drive入力は原則としてrepository相対pathをVM上へそのままmirrorする。
- credentialをconfig、Hydra override、log、成果物へ保存しない。
- 入力と完了済みrunは暗黙に上書きしない。曖昧さやupload失敗はerrorにする。

## 推奨する出発点

新しいDrive rootの出発点として、次の分類を推奨する。ただし名称をコード上の普遍的な
列挙値にはせず、`--drive-root`、job manifest、run requestから実際の場所を解決する。

```text
<drive-root>/
├── data/                  # repositoryの data/ をmirrorする入力
├── ckpt/                  # repositoryの ckpt/ をmirrorする入力checkpoint
├── colab-live/<run-id>/   # 実行中・失敗時にも残す可変mirror
│   └── training/          # runnerが同期するconfig/log/checkpoint/artifact
├── colab-runs/<run-id>/   # workflowが検証後にatomic publishする完了bundle
└── staging/               # 人間が検証してpromoteする前の候補
```

現在の`colab-live` / `colab-runs`はworkflowの既定規約であり、プロジェクト全体の永続的な
domain taxonomyではない。将来taskやteam単位の階層が必要になった場合は、互換性、既存
runのread path、migration手順をレビューしてから拡張する。agentが独断で`latest`、
`current`、task名などの新しい階層を追加してはならない。

local側は次を基準にする。

```text
<repository>/
├── data/                  # dataset / runtime input
├── ckpt/                  # model checkpoint input
├── outputs/               # run output
├── .cache/                # 再生成可能。Driveへpublishしない
├── assets/                # git管理するREADME用の軽量素材
└── third_party/           # 外部source tree。通常はDrive正本にしない
```

## 学習成果物の責務

全taskのtraining configは`run.artifact_store`を明示する。

```yaml
artifact_store:
  mode: local
  remote: null
  remote_root: null
  sync_interval_seconds: null
```

通常のlocal学習はこのままlocalにだけ保存する。Colabのrclone modeはworkflowが次を
予約済みoverrideとして注入し、credential pathだけを一時環境変数`RCLONE_CONFIG`で
渡す。

```yaml
artifact_store:
  mode: rclone
  remote: gdrive
  remote_root: <drive-root>/colab-live/<run-id>/training
  sync_interval_seconds: 60
```

`remote_root`は絶対的な推奨taxonomyではなく、そのrun requestが選んだ出力先である。
checkpointはLightningの通常の`ModelCheckpoint`がlocalへ保存した直後に一時名へuploadし、
remote上でmoveして公開する。TensorBoard、resolved config、予測、可視化などは周期的に
tree copyする。同期に失敗した場合はlocal-onlyへ切り替えずrunを失敗させる。

## 入力データ

通常のdataset/checkpoint inputはDriveとlocalで同じrepository相対pathを使う。

```text
Drive: <drive-root>/data/tennis/tracknet/...
local: <repository>/data/tennis/tracknet/...

Drive: <drive-root>/ckpt/court_detection/model.ckpt
local: <repository>/ckpt/court_detection/model.ckpt
```

外部libraryが自身のdirectory配下だけを読む場合は、job manifestで`source`から
`third_party/...`への明示的なstage adapterを宣言できる。この例外は入力の正本を移す
意味ではなく、暗黙symlinkやruntime fallbackを作ってはならない。

datasetを発展させるときは既存directoryへ直接追記せず、`staging/`またはrun artifactへ
新しい候補を生成し、内容・manifest・利用条件を人間が確認してから新しいversionとして
promoteする。version命名やmanifest schemaは全domainへ一律に固定せず、必要になった時点で
監修のもとこの文書へ追加する。最低限、由来run、作成日時、file count/size、content digest、
immutableかどうかを追跡できるようにする。

Google Driveは同名folderを許すため、canonical inputの同一pathに複数候補が見つかった
場合は自動選択しない。Drive IDとdigestをinventoryし、人間が正本を決めてからjobを実行
する。既存のlegacy配置は新runの途中で移動せず、read-only inventory、copy、digest検証、
job manifest切替の順にmigrationする。

## 新しい構造を追加するレビュー項目

1. 既存分類で意味を損なわず表現できるか。
2. repository local pathとのmirror関係または明示的stage adapterが一意か。
3. mutable inputを上書きせずversion/promotionで扱えるか。
4. credentialや個人情報を含まないか。
5. duplicate folder、partial upload、同じrun idの再利用をfail closedにできるか。
6. retentionとlegacy migrationを誰が担当するか。

承認後、この正本文書と機械可読job manifestだけを更新し、task READMEにはリンクだけを
置く。配置説明を複数READMEへコピーしない。
