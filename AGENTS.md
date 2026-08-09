# AGENTS.md

## プロジェクト概要

このプロジェクトは、テニスシーンの3次元再構成をAIによって解くことを目的とする。入力はマルチカメラの動画であり、各カメラにおけるボール位置・プレーヤーpose検出という2次元検出から始まり、それらを2D → 3Dへ再構築するモデルが最終的に3D空間へ写像する。

### パイプライン構成

| モジュール | 役割 |
|---|---|
| `src/tasks/ball_detection` | 2Dボール検出 |
| `src/tasks/court_detection` | 2Dコート検出 |
| `src/submodules` | 2Dプレーヤーpose検出 および 3Dプレーヤーpose推定（GVHMR 移植版。重みは `ckpt/` → `third_party/GVHMR/inputs/checkpoints` の symlink） |
| `src/tasks/blcs` | 2D ball + 2D court から3Dボール軌道を推論 |
| `src/tasks/plcs` | 2D pose + 2D court から3Dプレーヤーの位置・回転を推論 |

最終的に、GVHMRの3D poseとPLCSの3D位置・回転を統合し、コート座標系におけるプレーヤーの軌跡を3D上で再構築する。

## 開発環境

- Pythonの実行には `.venv/bin/python` を使用する。
- テストは `pytest` で実行する。`-n auto` が設定されているため並列実行される。
- コミット時には `.pre-commit-config.yaml` により以下が実行される。
  - **ruff**: `select = ["E", "F", "UP", "B", "SIM", "I"]`, `ignore = ["F405", "F403", "E501"]`
  - **mypy**: `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`, `no_implicit_optional`, `warn_return_any`, `warn_unused_ignores`, `warn_unreachable`, `strict_equality`
  - **task-script-reviewer**: `**/scripts/**` にスクリプトを作成する際の規約（モジュールdocstringの強制）

## 開発スタイル

### Repository Architecture Contract

`src/**`、`scripts/**`、configuration、path、import、module責務、model I/O、dataset / artifact schema、実行境界を変更する前に、[Repository Architecture Contract](docs/architecture/repository-contract.md)を読むこと。この文書を恒久的なarchitecture ruleの単一の正本とする。

作業開始時に次を行う。

1. 対象packageのREADMEを読み、現在のcanonical owner、public API、composition rootを確認する。
2. 変更に適用されるcontractのrule IDを特定する。
3. static importだけでなく、Hydra target、registry、CLI、subprocess、checkpoint、serialized schemaを含むconsumerを調査する。
4. Issue本文や過去のpathではなく、現在のsourceとcanonical auditを正とする。

計画とPR本文には、適用rule ID、canonical owner、削除・移行する旧surface、実行した検査を記載する。

### Configurationとpath

runtime defaultはcanonical config declarationだけが所有する。taskは完全なraw configをexact-validateし、shared baseへは必要fieldだけを閉じたtyped contractとして渡す。shared baseがtask固有mappingを再parseしてはならない。

pathは`project / data / checkpoint / artifact / output / cache / external_asset`のroleから解決する。CWD、directory名、旧layout、別拡張子からroleやpathを推測しない。non-Hydra boundaryもrole、direction、kind、existenceを明示して副作用前に検証する。

configurationまたはpathに変更を加えた場合、少なくとも次を実行する。

```bash
.venv/bin/python scripts/audit_configuration.py src
```

`.get`等の構文自体を機械的に置換するのではなく、configuration fallbackか、明示されたoptional data semanticsかを区別する。audit exemptionが必要な場合は、module、qualified name、line、`AuditRule`、stable reason codeを持つexact source findingとして登録する。

### Model I/Oとcompute hot path

model固有のinput key、shape、dtype、mask、座標系、output decodeは各taskの`model_io/**`が所有する。`src/tasks/base/model_io/**`は`ModelCall`、binding、build → execute → decodeという共有mechanicsだけを所有する。

- Dataset / DataModuleはcanonical domain batchを返し、model profileへの変形を行わない。
- modelとadapterはcomposition rootで一度だけ選択・検証する。
- training、inference、evaluation、visualization、統合pipelineで同じtask adapter contractを利用する。
- runnerやpipelineへmodel名、model class、raw output keyによる分岐を追加しない。
- checkpoint復元時もcomposition configからmodelとadapterを再構成し、state dictから推測しない。

repository-owned `nn.Module.forward`、loss `forward`、そこから到達するrepository helperは検証済みtensorの計算に限定する。shape / type validation、mask準備、backend選択、matching、output decode、logging、I/O、module生成、state mutationはconstructor、factory、adapter、`prepare_inputs`等のboundaryへ置く。

### 責務と配置

- task非依存のalgorithm、primitive、schema、device、geometry、rendering、model componentは `src/utils` に置く。
- 複数taskで共有するdata / training / inference / visualizationのlifecycle mechanicsは `src/tasks/base` に置く。
- task固有のtensor、dataset、loss、metric、pipeline semanticsは該当task packageに置く。
- vendor内部へproject固有logicを追加せず、repository-owned wrapper / adapterで接続する。

共有化はimport数ではなく責務と依存方向で判断する。同じ責務の旧実装を残したまま新しいhelperを追加してはならない。

### Compatibilityと静かなfallbackの禁止

旧key、旧path、assignment alias、compatibility re-export、pass-through wrapper、deprecated shim、dual-read / dual-write、checkpoint key migration、runtime schema upgradeをactive production pathへ残さない。repository-owned consumerを同じ変更でcanonical surfaceへ移行し、旧形式は明示的に失敗させる。

不足したdependency、data、checkpoint、metadata、track ID、visibility、fps、frame count等をdummy、zero tensor、連番、推測defaultで補わない。`auto`のようにavailability selectionを明示した値だけが環境に応じて選択できる。明示CUDA / GPU / backendが利用不能な場合はmodel、Trainer、subprocess構築前に失敗させる。

### テスト

新しくmoduleを実装・改善した際は、意味のあるtestを作成・改善する。特に`src/utils`、`src/tasks/base`、model I/O、schema / archive、path boundaryの変更では必須。

正常系だけでなく、関連するmissing / unknown / old key、wrong type / shape / dtype / mask、path escape、model-adapter mismatch、unavailable explicit device、malformed outputを検証する。不正inputではmodel callやfilesystem / GPU / subprocess side effectが発生していないことも確認する。

恒久的なarchitecture checkはcurrent source treeを直接検証し、過去base SHAや一回限りの削除一覧だけに依存させない。

### 不明点は積極的に質問する

人間による方針の明確化が必要だと判断した場合は質問する。あいまいなまま動くだけのcode、silent compatibility、意味を変えるfallbackを作らない。

### ドキュメントの二重管理禁止

同じ事柄を2つの場所に記述しない。

- 恒久的なarchitecture rule: `docs/architecture/repository-contract.md`。
- agent向け導線: `AGENTS.md`。
- 現在のpackage構成、public API、data flow、実行方法: 該当README。
- 設計変更と移行evidence: Issue / PR。

canonical owner、public API、entrypointを変更した場合は該当READMEを同じPRで更新する。

### READMEを起点とした探索

大きなdirectoryにはREADMEを設置し、具体的な実装を簡潔にまとめている。実装に取り組む際は、READMEを探索起点とし、その後にcanonical source、factory、adapter、testを確認する。

### リファクタリング優先

機能追加により責務の重複、model-specific branch、fallback、flat packageが増える場合は、canonical ownerを確立するrefactoringを先に行う。新旧surfaceを併存させる段階移行を既定戦略にしない。

### Colabでの学習

ユーザーの指定がある場合、学習はローカルGPUを用いずColabで実行する。その場合は`scripts/colab`にshell scriptを実装し、Colabではそのscriptだけを実行する（Driveのmountは別途行う）。指定がない場合はローカルGPUを用いる。必ず`.training_queue/`経由で実行する。
