# Repository Architecture Contract

## 1. 文書の位置づけ

この文書は、`tennis-lab`のrepository-owned codeに適用する**規範的なアーキテクチャ契約**です。Issue #688 / PR #705およびIssue #691 / PR #709で実際に導入・移行・削除された実装を基礎とし、merge後も維持する設計不変条件を定義します。

本書は次の事項に対する単一の正本です。

- runtime configurationと既定値のauthority。
- filesystem pathのrole、解決、検証、引き渡し方法。
- module、package、public API、composition rootの責務境界。
- model、adapter、training / inference / evaluation / visualization間のI/O契約。
- `nn.Module.forward`およびそこから到達するtensor hot pathの責務。
- compatibility、fallback、entrypoint、検査、例外の扱い。

`AGENTS.md`は本書への導線、package READMEは現在のmodule構成・public API・data flow・実行方法、Issue / PRは設計変更と移行証拠の履歴です。本書とREADME、code comment、過去Issueが競合する場合、恒久的なarchitecture ruleについては本書を優先します。

本書は特定のbase SHA、変更ファイル一覧、Issue間の一時的な依存関係、workflowの完了手順を規約化しません。それらは移行時の証拠であり、current-state architectureではありません。

## 2. 適用範囲と非規定事項

### 2.1 適用範囲

本契約は次に適用します。

- `src/**`のrepository-owned production code。
- `scripts/**`、`src/**/scripts/**`、package `__main__.py`。
- Hydra config、factory、registry、dynamic target、CLI、subprocess boundary。
- repository-owned wrapper、adapter、vendor接続層。
- schema、manifest、checkpoint、dataset、artifactのreader / writer。
- 本契約を強制する`tests/**`、audit、lint、type check、CI。
- canonical ownerやpublic surfaceを説明するREADMEとarchitecture document。

### 2.2 原則として対象外

- `third_party/**`。
- `src/submodules/vendor/**`のvendor内部実装。
- runtimeが生成する`data/**`、`outputs/**`、checkpoint、cacheそのもの。

vendor codeを呼び出すrepository-owned wrapper、request / result contract、configuration、pipeline componentは対象です。

### 2.3 本書が直接決めない事項

本書はmodel architecture、lossの研究上の定義、datasetの内容、学習hyperparameter、domain固有のstage DAGそのものを固定しません。ただし、それらの**authority、構成、I/O、失敗境界、配置、public API**は規定します。

`src/synthetic_data_generation/**`のdomain固有architectureは、その領域のcanonical設計が所有します。本書はそこへもrepository-wideなconfiguration、path、API、entrypoint、fail-closed規則だけを適用し、domain固有のstage構造を先回りして固定しません。

## 3. Authorityと依存方向

### OWN-001 — one responsibility, one owner

一つの責務には、一つのcanonical ownerと一つのcanonical implementationだけを置きます。共有実装を追加した後もtask-local複製を残す、同じschemaをreaderごとに再定義する、同じmodel出力を複数adapterで別々にdecodeする、といった並行所有を禁止します。

### OWN-002 — canonical placement

配置は責務と依存方向で決めます。

- task非依存のalgorithm、primitive、schema、device、geometry、rendering、model component: `src/utils/**`。
- 複数taskで共有するdata / training / inference / visualizationの**lifecycle mechanics**: `src/tasks/base/**`。
- task固有のtensor semantics、dataset semantics、model I/O、loss / metric、domain pipeline: `src/tasks/<task>/**`。
- tennis scene全体の統合schema、archive、pipeline composition: `src/tennis_scene/**`。
- vendor接続: repository-owned wrapper / adapter。project固有logicをvendor内部へ混入させない。

単に複数箇所からimportされるという理由だけで共有層へ移しません。task固有のkey、shape、座標系、単位、identity、visibility semanticsを知る処理はtask側が所有します。

### OWN-003 — shared baseは完全なtask設定を解釈しない

各taskは、自身の完全なraw configurationをexact-validateし、task固有のtyped runtime contractを構築します。`src/tasks/base`へは、その共有lifecycleに必要なfieldだけを閉じた型または明示的なprojectionとして渡します。

共有baseが任意のtask mappingを再parseすること、task固有keyを`.get`で探索すること、taskごとに意味の異なるfallbackを持つことを禁止します。

### OWN-004 — shared value objectはruntime composition moduleより下位へ置く

複数のcomposition boundaryが同じ型を必要とする場合、一方のruntime moduleから他方をreverse-importせず、両者が一方向に依存できる下位のcontract moduleへvalue objectを置きます。consumer間の循環importやimport順依存を、re-exportや遅延importで隠してはなりません。

### OWN-005 — architecture layerを分離する

次の責務は、moduleまたはsubpackageとして分離します。

- schema / contract / archive I/O。
- data loading、collate、augmentation。
- model / component。
- model I/O adapterとcomposition factory。
- training / loss / metric。
- inference / pipeline orchestration。
- rendering / visualization。
- executable entrypoint。

巨大moduleやpackage rootへ複数layerを混在させません。package rootは意図的なpublic APIまたはcomposition rootに限定します。

## 4. Configuration contract

### CFG-001 — runtime値と既定値はcomposition-owned

runtimeで選択可能な値、型、必須・任意区分、既定値、値域、排他条件は、canonical Hydra / YAML declarationまたは明示的なschema declarationだけが所有します。同じ既定値をPython constructor、runner、adapter、CLI wrapperへ複製してはなりません。

code内に置くことができるのは、ユーザー選択肢ではないalgorithm invariantやformat constantです。runtime behaviorを変える値を「内部定数」として隠してはなりません。

### CFG-002 — composed valueだけを実行authorityとする

設定の優先順位はcomposition boundaryまでに解決し、内部moduleには選択済みの値だけを渡します。実行中に次の経路を追加してはなりません。

- Python側default。
- 旧key alias。
- `dict.get(key, fallback)`による補完。
- `getattr(value, fallback)`による補完。
- `setdefault`によるmutation。
- model class、file存在、環境状態からの暗黙推論。

任意fieldは「欠落のまま保持する」または「boundaryで`None`へmaterializeする」のどちらかをschemaで宣言します。consumerが独自にabsence semanticsを決めてはなりません。

### CFG-003 — exact recursive validation

raw configurationはcomposition rootまたはruntime entrypointで一度だけ、完全なkey setとexact typeを再帰的に検証します。

少なくとも次を副作用前に拒否します。

- 必須keyの欠落。
- 未知key、タイプミス、deprecated key。
- `bool`と`int`を含むexact typeの不一致。
- sequence / mappingのrank、要素型、長さの不一致。
- enum、値域、有限性、単位、相互依存、排他条件の違反。
- 明示device / backend / dependencyの利用不能。

検証済みmappingはconsumerが再解釈できない形に閉じ、typed dataclassはruntime defaultを持ちません。

### CFG-004 — schema authorityとsemantic authorityを分離して明示する

fieldの存在・型を検証するschema authorityと、field間の意味制約を検証するsemantic authorityを明示します。boundaryごとに、どのschema / adapter / semantic validator / path authorityが実行を支配するかをsourceから追跡可能にします。

### CFG-005 — source-derived catalogを正本とする

configuration contractの網羅性は、手書き一覧ではなくrepository sourceからのdiscoveryを基準にします。canonical implementationは次の責務分割を持ちます。

- `src/utils/configuration/discovery.py`: runtime boundary discoveryの唯一のowner。
- `src/utils/configuration/contracts.py`: dataclass、strict schema、non-Hydra path boundaryのinspectable contract生成。
- `src/utils/configuration/catalog.py`: source declarationとruntime boundaryを結び付けるcatalog。
- `src/utils/configuration/audit.py`: source、catalog、inventory、exemptionの整合検査。
- `src/utils/configuration/inventory.py`: migration / boundary inventory。
- `scripts/audit_configuration.py`: canonical operational entrypoint。

catalog、audit、discovery間にback edgeを作らず、下位のdiscoveryへ一方向に依存します。

### CFG-006 — 構文を一律禁止せず、configuration semanticsを検査する

`.get`、`getattr`、mapping lookupなどの構文自体をrepository全体で一律禁止しません。永続dataの明示的なoptional fieldや純粋algorithm用途など、configuration fallbackではない使用は存在し得ます。

ただし、audit findingを除外する場合は、canonical exemption dataに正確なsource位置、finding kind、理由、ownerを持たせます。広いpath単位、曖昧な説明、将来の未知findingまで覆うblanket exemptionを禁止します。

### CFG-007 — generated audit dataの更新をfail-closedにする

migration / exemption dataを書き換える操作は、immutableなsource revisionを要求し、新規の未分類constructがある場合は失敗させます。auditを通すために生成dataを無条件で更新してはなりません。

### CFG-008 — repository-owned consumerをatomicに移行する

設定key、schema、default、typed adapterを変更する場合、static importだけでなくHydra dynamic target、registry、factory、CLI、subprocess、checkpoint restore、tests、documentationを同じ変更で移行します。旧key reader、dual-read、dual-write、runtime migrationを残しません。

## 5. Path contract

### PATH-001 — seven typed path roles

runtime path authorityは、次の7 roleへ明示的に分類します。

- `project_root`。
- `data_root`。
- `checkpoint_root`。
- `artifact_root`。
- `output_root`。
- `cache_root`。
- `external_asset_root`。

`RuntimePathRoots`は7 rootを完全に持ち、各rootをresolved absolute `Path`として検証します。filesystem root自体をruntime authorityとして与えてはなりません。

### PATH-002 — roleはpath文字列から推測しない

pathの意味はdirectory名、prefix、拡張子、存在場所から推測せず、callerが`PathRole`として宣言します。同じ文字列を異なるroleとして解釈する実装や、`data/`、`outputs/`等をchild fragmentへ重ねてroleを選び直す実装を禁止します。

### PATH-003 — configured pathはrole-relative fragmentとして解決する

Hydra等で構成されるderived pathは、選択済みrole rootに対する非空のrelative childとして`PathResolver`から解決します。

- absolute childを拒否する。
- `.`、`..`、root-prefixed / legacy fragmentを拒否する。
- symlink解決後もrootまたは指定parentからのescapeを拒否する。
- current working directoryへ依存しない。
- already-resolved pathを再利用する場合も`validate(role, path)`でrole containmentを確認する。

### PATH-004 — non-Hydra boundaryは完全なtyped declarationを持つ

argparse、Python callable、subprocess等でpathを受けるboundaryは`NonHydraPathBoundary`相当の明示契約を持ちます。各fieldは少なくとも次を宣言します。

- field name。
- `PathRole`。
- input / output direction。
- file / directory / any kind。
- input existence requirement。
- scalar / non-empty sequence。
- role root自体を許可するか。

non-Hydra path値はabsolute pathとして受け取り、未知field、欠落field、重複、escape、kind不一致をfilesystem mutation前に拒否します。

### PATH-005 — persisted pathも使用直前に再検証する

manifest、run state、checkpoint metadata等へ保存したpathは、読み出しただけでtrustedにしません。roleとabsolute pathを保持し、使用時に現在のresolverでcontainment、existence、kindを再検証します。

### PATH-006 — subprocessとexternal executableも同じ境界に置く

external repository、checkpoint、executable、input / output artifactをsubprocessへ渡す場合も、role、kind、existence、containmentをspawn前に検証します。subprocess wrapperがCWDやPATH探索で別実装を暗黙選択してはなりません。

### PATH-007 — 候補探索は有限で明示的なpublic contractにする

複数候補pathや拡張子探索が正当な機能である場合、候補集合、優先順位、曖昧時の失敗条件をpublic contractとして定義します。旧layoutを偶然発見して処理を継続する無制限探索やcompatibility fallbackを禁止します。

## 6. Canonical APIとmigration

### API-001 — one canonical public surface

public class、function、schema、factory、entrypointには一つのcanonical import pathを定めます。package `__init__.py`から公開する場合、そのre-export自体を意図的なpublic APIとし、内部pathを並列のpublic surfaceとして扱いません。

内部subpackageの`__init__.py`へ便宜的にimplementationを大量re-exportし、ownerを不明確にしてはなりません。特にvendor wrapperはrepositoryが定めた単一のpublic package surfaceから利用します。

### API-002 — compatibility-only surfaceを禁止する

次を、旧consumerを動かす目的だけで追加・維持してはなりません。

- assignment alias。
- compatibility re-export。
- pass-through wrapper / property。
- deprecated shim。
- module-level `__getattr__` forwarding。
- 旧pathから新pathへのforwarder。
- 旧signatureを受け取るsilent adapter。
- checkpoint keyのload-time migration。
- old schemaのruntime upgrade。

独立した責務とtyped contractを持つadapter / wrapperは許可します。古い名前を新実装へ転送するだけの層は許可しません。

### API-003 — atomic repository migration

canonical APIを変更する場合、repository-owned consumerを同じPRで移行し、旧module / symbol / dynamic targetを削除します。新旧package treeの併存を通常の移行戦略にしません。

repository外consumerの移行が必要な場合は、active production pathへcompatibility layerを入れるのではなく、明示的なoffline converter、migration document、別versioned boundaryを設計し、本書の例外手続きを通します。

### API-004 — 削除済み契約は明示的に失敗させる

旧moduleは`ModuleNotFoundError`、旧symbolはabsence、旧artifact / checkpoint / schemaは具体的なcontract errorとして失敗させます。別実装へ暗黙転送して成功させません。

### API-005 — static / dynamic consumerを両方検査する

migration inventoryには少なくとも次を含めます。

- Python static import。
- package re-export。
- Hydra `_target_`。
- registry / factory。
- `python -m` CLI。
- subprocess invocation。
- serialized module / class name。
- checkpoint restore path。
- tests、examples、README。

## 7. Model I/O contract

### IO-001 — task adapterがtensor semanticsを所有する

model固有のinput key、dtype、device、rank、shape、値域、mask、padding、座標系、単位、output inventoryはtask-local model I/O adapterが所有します。`src/tasks/base`はこれらのdomain semanticsを知りません。

### IO-002 — shared layerはlifecycle mechanicsだけを所有する

canonical shared mechanismは次の責務に限定します。

- `ModelCall`: 検証済みのtensorまたは明示的な`None`だけを持つimmutable call。
- `ModelIOAdapter`: task adapterが実装する`build_call` / `decode_output` contract。
- `BoundModelIO`: build → execute → decodeの順序。
- `bind_model_io`: modelとadapterをcomposition時に一度だけbindingする境界。
- `TensorSpec` / `require_tensor`: task adapterが利用する共有のtensor検証primitive。

shared layerへtask固有key、出力名、camera / player / ball semanticsを追加しません。

### IO-003 — modelとadapterをcomposition rootで一度だけ選択する

model variant、input profile、output head、attention backend、task adapterの選択はfactory / composition rootで完了させます。modelとadapterのclass、dimension、profile、output strategyが整合することをmodel call前に検証します。

training / inference / evaluation / visualization loopの途中で、model名、`isinstance(model, ...)`、raw output keyにより実装を選び直してはなりません。

### IO-004 — canonical domain batchをmodel profileから独立させる

Dataset / DataModule / collateはdomainのcanonical batchを返します。frame、sequence、multiview、axial、track-query等のmodel profileへのselection、flatten、broadcast、mask生成はtask adapterが行います。

DataModuleがmodel名を読み、modelごとのbatch shapeを生成する構造を禁止します。

### IO-005 — adapterはforward前の全準備を所有する

adapterまたは明示的なexecution boundaryは、model entry前に次を完了します。

- required fieldとexact tensor typeの検証。
- dtype、device、rank、fixed / relational shapeの検証。
- normalized coordinate、finite、mask、padding、identity等のsemantic検証。
- canonical batchからmodel profileへの変換。
- attention mask、valid-state、empty-row policyの構築。
- DINO等のdynamic external responseの型・key・shape検証。
- model callのimmutable化。

不正inputではmodel call countが0であることをtest可能にします。

### IO-006 — raw outputはadapterがexactにdecodeする

raw outputは、必須key、未知key、dtype、shape、finite、semantic rangeをadapterで検証し、typed predictionへdecodeします。training、predictor、pipelineがraw dictionaryを直接解釈してはなりません。

presence threshold、physical unit conversion、pixel / normalized座標変換等を複数consumerで再実装せず、選択済みcontractの一箇所で行います。

### IO-007 — 同じadapter contractを全lifecycleで利用する

training、inference、evaluation、visualization、統合pipelineは、同一taskのcanonical model I/O contractを利用します。評価専用adapter、可視化専用のmodel batch converter、pipeline内の出力key分岐を並行実装してはなりません。

### IO-008 — checkpointはbinding再構成に必要なauthorityを保持する

checkpointからmodelを復元する場合、modelとadapterのcompositionを厳密に再構成できるconfig / metadataを必須とします。state dictのkey、tensor shape、class名の推測だけでvariantを選択しません。

旧checkpoint keyをload hookで書き換えて受理してはなりません。必要ならactive loader外の明示的なoffline migrationを用意します。

### IO-009 — mutable / vendor boundaryでは使用直前に再検証する

frozen dataclassや`ModelCall`に格納されていても、tensorや外部object自体はmutableであり得ます。buildから実行までにmutation可能なrequest、遅延実行、vendor callでは、downstream entry直前にshape、dtype、device、finite、semantic consistencyを再検証します。

## 8. Compute hot path contract

### FWD-001 — `forward`はtensor計算に限定する

repository-owned `nn.Module.forward`と、そこから到達するrepository-owned helperは、constructor / factory / adapterで解決済みのmodule stateと検証済みtensorに対する計算だけを行います。

### FWD-002 — Python validationをhot pathへ置かない

入力契約確認を目的とする次を`forward`およびtransitive helperへ置いてはなりません。

- Python `assert`、`raise`。
- `isinstance`、`type`、`hasattr`、`getattr`によるruntime type / implementation selection。
- `.shape`、`.ndim`、`len`、`.item()`等を使うPython validation branch。
- `validate_*`、`require_*`、`ensure_*`等のvalidation helper call。

static hyperparameterはconstructor / factory、external batchはadapter / data boundary、raw outputはdecode boundaryで検証します。

algorithmそのものに必要なtensor control flowは許可します。標準PyTorch layerのtraining-mode挙動も本項の違反ではありません。

### FWD-003 — side effectとstate mutationを禁止する

hot pathで次を行ってはなりません。

- filesystem / network / subprocess I/O。
- logging、print、artifact保存。
- config / global / module state mutation。
- module、parameter、optimizerの生成。
- backend、model variant、loss term、adapterの選択。
- 不正inputの補正や別algorithmへのfallback。

### FWD-004 — implementationはconstruction時に固定する

MoE、time-local attention、output head、fusion、optional branch、loss penalty等の実装候補はconstructor / composition時にcallableまたはmoduleとして固定します。明示CUDA backendが利用不能ならcomposition時に失敗し、tensor実行中にreference backendへ切り替えません。

### FWD-005 — loss `forward`もprepared tensorだけを合成する

repository-owned loss moduleは、matching、registry dispatch、shape validation、term constructionを`prepare_inputs`等のboundaryで完了し、`forward`ではprepared tensor termを計算・合成します。empty mask等の正当なtensor-domain caseはfiniteな定義済み結果を返しますが、missing inputを補完してはなりません。

## 9. Fail-closed runtime contract

### FAIL-001 — explicit requestの意味を変えない

`"auto"`のようにavailability selectionをpublic contractとして明示した値だけが環境に応じて選択できます。明示されたCUDA / GPU count / backend / checkpoint / detector / model variantが利用不能な場合、model、Trainer、backbone、subprocessの構築前に失敗させます。

### FAIL-002 — missing dependencyやmetadataを合成しない

不足したdependency、dataset field、scene metadata、checkpoint config、track ID、visibility、fps、frame count、camera、eventを次で隠してはなりません。

- dummy / no-op implementation。
- zero tensor、空配列、連番ID。
- guessed default。
- shapeからの旧schema推定。
-別file / extension / directoryの暗黙探索。
- geometryからのsemantic event推定。

optionalである場合は、schemaでoptional性とabsence semanticsを明示します。

### FAIL-003 — fallbackは同一semantic contractを保存する場合だけ許可する

明示的なfallback policyを設ける場合、入力・出力のshape、dtype、座標系、単位、identity、artifact ownership、品質意味を保存し、選択条件をdeterministicかつ観測可能にします。

簡略化された物理、別camera、別detector、別backend等へ切り替えて「動作だけ継続する」fallbackは禁止します。

### FAIL-004 — destructive side effectより先に検証する

file作成、directory mutation、GPU確保、model / Trainer構築、subprocess spawn、dataset走査の前に、当該boundaryで確認可能なconfiguration、path、dependency、schema、device、request整合性を検証します。

### FAIL-005 — errorは契約違反箇所を特定する

例外は、field / tensor / path / model-adapter pair、期待値、実際値、失敗boundaryを特定できる内容にします。広い例外をcatchして別経路へ進む実装や、warningだけで継続する実装を禁止します。

## 10. Data、schema、artifact contract

### DATA-001 — persistent schemaはcanonical reader / writerを一組だけ持つ

manifest、annotation、scene archive、dataset indexには、一つのcanonical schema ownerとreader / writerを定めます。consumer taskがforeign schemaやerror classを再export・複製してはなりません。

### DATA-002 — required / optionalを明示する

required fieldやarrayが欠落した場合は明示的に失敗します。optional fieldはschemaで宣言し、readerが旧layoutやshapeから存在を推測しません。

frame count、camera ID、track ID、visibility、fps、dtype、shape、relative media path、manifest digest等の相互整合性をreader boundaryで検証します。

### DATA-003 — path traversalとformat ambiguityを拒否する

artifact内のrelative pathはowner directoryからescapeできません。format suffix、metadata sidecar、JSON object等をpublic contractとして固定し、pickle fallback、別suffix補完、missing sidecarの推測を禁止します。

### DATA-004 — test fixtureをproduction APIにしない

synthetic fixture、dummy dataset builder、test-only defaultは`tests/support/**`等のtest領域へ置き、production packageのpublic surfaceにしません。fixtureは可能な限りcanonical production writerを合成して作成し、別schemaを発明しません。

### DATA-005 — provenanceとsemantic completenessを検証する

artifactを受理する条件はbyte存在だけではありません。必要に応じてsource manifest、schema version、shape、dtype、finite、frame / identity対応、camera / coordinate convention、required arraysを検証します。

## 11. Entrypointとlibrary boundary

### ENTRY-001 — executable surfaceを限定する

`if __name__ == "__main__":`は次に限定します。

- repository rootの`scripts/**`。
- `src/**/scripts/**`。
- packageの`__main__.py`。

通常のlibrary moduleへdemo、print、乱数入力、save / load、assertを含むmain guardを置きません。

### ENTRY-002 — library importをside-effect freeにする

library importでmodel load、filesystem mutation、network、subprocess、実行検証を開始してはなりません。smoke検証は`tests/**`、運用機能は明示的entrypointへ置きます。

### ENTRY-003 — operational entrypointを一つにする

同じ運用責務に複数のroot script、`python -m` path、library `main()`を並行公開しません。configuration auditのcanonical operational entrypointは`scripts/audit_configuration.py`です。

## 12. Documentation ownership

### DOC-001 — 規約を重複させない

- 本書: 恒久的なrepository architecture rule。
- `AGENTS.md`: agentが本書を必ず読むための導線と最小限の作業規則。
- package README: 現在のowner、module、public API、data flow、実行方法。
- Issue / PR: 意思決定、移行scope、evidenceの履歴。

同じ規約全文を複数READMEへ複製しません。

### DOC-002 — current implementation mapを同時更新する

canonical owner、package tree、public import、entrypoint、model I/O factoryを変更した場合、該当READMEと本書のenforcement / implementation mapを同じPRで更新します。

### DOC-003 — ruleと歴史的evidenceを分離する

過去base SHA、削除module一覧、特定Issueだけのallowlistはmigration evidenceです。current-state invariantをそれらだけで強制してはなりません。

## 13. Mechanical enforcement

### ENF-001 — canonical enforcement surface

| 対象 | Canonical implementation / check |
|---|---|
| exact configuration schema | `src/utils/configuration/schema.py` |
| typed config / boundary inspection | `src/utils/configuration/contracts.py` |
| runtime boundary discovery | `src/utils/configuration/discovery.py` |
| source-complete catalog | `src/utils/configuration/catalog.py` |
| audit / inventory / exemptions | `src/utils/configuration/audit.py`, `inventory.py`, `migration_data.py`, `exemption_data.py` |
| path roles / resolver / non-Hydra paths | `src/utils/configuration/paths.py` |
| operational configuration audit | `scripts/audit_configuration.py` |
| model I/O lifecycle mechanics | `src/tasks/base/model_io/**` |
| task tensor semantics | 各`src/tasks/<task>/model_io/**` |
| strict device selection | `src/utils/device.py` |
| repository architecture checks | `tests/e2e/development/test_architecture_boundaries.py` |
| audit entrypoint checks | `tests/e2e/development/test_configuration_audit.py` |
| style / typing / regressions | canonical ruff、mypy、pytest、CI checks |

### ENF-002 — source discoveryとcatalog parityを検証する

新しいconfig dataclass、strict schema、non-Hydra path boundary、runtime entrypointを追加した場合、source discovery、catalog、boundary authority、inventoryが一致することを検証します。手書きcatalogだけ更新してsource completenessを装ってはなりません。

### ENF-003 — negative caseをfirst-class testにする

正常系だけでなく、少なくとも関連する次のcaseを検証します。

- missing / unknown / legacy configuration。
- exact type、value range、semantic conflict。
- path escape、wrong role、missing input、wrong kind。
- unavailable explicit device / backend。
- model-adapter mismatch。
- missing tensor、wrong dtype / rank / shape / mask / coordinate range。
- unknown / missing / malformed model output。
- old import、old symbol、old checkpoint / schema。
- side effectまたはmodel callが失敗前に実行されていないこと。

### ENF-004 — permanent checkはcurrent-state invariantにする

恒久規約の検査は現在のsource treeとruntime contractを直接検証します。特定base revisionとの差分、過去の削除一覧、Issue固有allowlistだけに依存させません。

`tests/e2e/development/test_architecture_boundaries.py`に残るone-time migration evidenceは、その目的を明示して恒久invariantと区別します。新規規約を追加する際はcurrent-state testを追加します。

### ENF-005 — forwardはtransitive call graphまで検査する

`forward`本体だけでなく、そこから呼ばれるrepository-owned method / functionもcompute-only contractを満たすことを検査します。validationをprivate helperへ移動しただけでは準拠になりません。

### ENF-006 — test通過をcompatibilityの根拠にしない

既存testが旧alias、fallback、parallel implementationを前提としている場合、canonical architectureへ合わせてtestを更新または削除します。ただし、algorithmic behaviorや外部public contractの意図しない変更を隠すためにassertionを弱めてはなりません。

## 14. Exception procedure

本契約から逸脱する場合は、実装前または同じPRで次を満たします。

1. 対象rule IDを明記する。
2. 対象path、symbol、runtime boundaryを最小範囲で特定する。
3. 契約どおりに実装できない技術的理由を示す。
4. canonical ownerと、逸脱が拡大しない検査を示す。
5. removal condition、期限またはtracking Issueを示す。
6. 恒久的に正しい設計変更なら、例外ではなく本書自体を更新する。

configuration auditのsource findingを除外する場合は、本節の一般例外ではなく`CFG-006` / `CFG-007`に従い、canonical exemption dataへexact findingとして登録します。

現在、一般architecture exceptionは登録されていません。

## 15. Change and review checklist

architectureに影響する変更は、少なくとも次を確認します。

- [ ] 適用rule IDとcanonical ownerを特定した。
- [ ] package README、source discovery、既存factory / adapterを先に確認した。
- [ ] runtime default、path role、public API、tensor semanticsを複製していない。
- [ ] task完全schemaとshared projectionの依存方向を守った。
- [ ] static import、dynamic target、CLI、registry、checkpoint等の全consumerを調査した。
- [ ] modelとtask adapterをcomposition rootで一度だけbindingした。
- [ ] DataModule / runner / pipelineへmodel variant分岐やraw output decodeを追加していない。
- [ ] validation、mask preparation、dynamic external response検査をhot path前へ置いた。
- [ ] old key、old path、alias、shim、dual route、runtime migrationを残していない。
- [ ] explicit requestが利用不能な場合に副作用前に失敗する。
- [ ] negative caseと「失敗前にmodel / side effectが実行されない」testを追加・更新した。
- [ ] package READMEと本書のimplementation / enforcement mapを更新した。
- [ ] `scripts/audit_configuration.py src`とrelevantなruff、mypy、pytest、CIが成功した。
- [ ] 逸脱がある場合、Exception procedureを完了した。

## 16. Provenance

本契約は、次の完了済みmigrationのIssue本文だけでなく、各PRの全変更を領域別に確認し、最終`main`のcanonical implementationと照合して作成しています。

- Issue #688 — 設定解決とパス管理を単一の正本へ統合する。
- PR #705 — 設定・パス契約を統一する。
- Issue #691 — 責務境界・モデルI/O契約を再構成する。
- PR #709 — アーキテクチャの責務境界とモデルI/O契約を再構成する。

PR #705で導入されたconfiguration / path foundationのうち、PR #709で移動・統合・削除されたsurfaceについては、後者と現在の`main`を正としています。たとえば、configuration auditの運用entrypointはroot script、model I/Oの共有ownerは`src/tasks/base/model_io/**`、task tensor semanticsのownerは各taskの`model_io/**`です。
