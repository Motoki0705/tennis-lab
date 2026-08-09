# Repository Architecture Contract

## 1. 文書の位置づけ

この文書は、`tennis-lab` のrepository-owned codeに対する**規範的なアーキテクチャ契約**です。設定、パス、module責務、canonical API、model I/O、実行境界を変更する場合は、本契約を単一の正本として扱います。

- **必須**: 例外登録がない限り遵守しなければならない。
- **禁止**: repository-owned production pathへ導入してはならない。
- **原則**: 逸脱には具体的な技術的根拠と、本書の例外登録が必要である。

IssueやPRは対象作業について本契約を強化できますが、暗黙に弱めることはできません。本契約とpackage README、code comment、過去Issueの記述が競合する場合は、本契約を優先します。`AGENTS.md`は本契約への入口であり、規約本文の複製先ではありません。

本契約はIssue #688と#691で確立した設計判断を恒久化したものです。各Issueに固有だったbase revision、変更可能範囲、依存Issue、移行対象一覧、完了workflowは履歴情報であり、本契約には含めません。

## 2. 適用範囲

本契約は、次のrepository-owned領域に適用します。

- `src/**`。ただし、明示的に隔離されたvendor implementationを除く。
- `scripts/**`、`src/**/scripts/**`、package `__main__.py`。
- Hydra config、registry、factory、dynamic target、CLI、subprocess invocationなど、runtimeを構成する定義。
- 本契約を検証する`tests/**`、audit、lint、type check、CI設定。
- canonical owner、public API、実行方法を説明するREADMEおよびarchitecture document。

次は原則として適用対象外です。

- `third_party/**`。
- `src/submodules/vendor/**`の内部実装。
- `data/**`、`outputs/**`、checkpoint、cacheなどの生成物。

vendorをrepositoryへ接続するwrapper、adapter、configuration、pipeline componentはrepository-owned codeであり、本契約の対象です。本契約はmodel architecture、loss、dataset内容、学習hyperparameterなどの研究上の選択自体を規定しません。ただし、それらを構成・実行する境界は規定します。

## 3. 設定契約

### CFG-001 — 設定と既定値の単一正本

各設定値、型、必須・任意区分、既定値、許容範囲、排他条件は、一つのcanonical schemaまたはconfig declarationだけで定義します。同じ既定値や意味制約を複数のPython module、YAML、CLI wrapperへ複製してはなりません。

### CFG-002 — 実行境界で一度だけ解決する

raw configはcomposition rootまたはruntime entrypointで一度だけ解決し、型検証・意味検証済みのtyped contractへ変換します。内部moduleには、未検証のmappingではなく、解決済みcontractを渡します。内部moduleが設定の優先順位を再解釈したり、別の既定値を追加したりしてはなりません。

### CFG-003 — fail-fast validation

次は、model構築、dataset走査、GPU確保、file出力などの副作用より前に拒否します。

- 必須keyの欠落。
- 未知key、タイプミス、deprecated key。
- 型、rank、shape、値域、enum、単位の不一致。
- 排他的な設定の同時指定や、依存設定の欠落。
- 利用不能なbackend、device、external dependencyを明示指定した状態。

エラーは、問題のfield、期待契約、実際値を特定できる形で報告します。

### CFG-004 — 暗黙補完の禁止

必須設定を補うための`.get(key, fallback)`、連鎖`.get(...)`、`getattr(..., fallback)`、`setdefault`、truthinessによる代替値選択をproduction codeで使用してはなりません。任意値のdefaultはcanonical schemaで宣言し、実行境界で明示的にmaterializeします。

複数の実装候補や`auto`選択は、それ自体がtyped public contractとして定義され、選択条件と失敗条件が検証可能な場合に限り許可します。利用不能な明示指定を別実装へ静かに切り替えてはなりません。

### CFG-005 — repository内consumerの一括移行

設定keyやschemaを変更する場合は、repository-owned consumerを同じ変更で移行します。旧keyのalias、旧schema reader、dual-read、dual-write、silent migrationを移行手段として追加してはなりません。旧形式は明確なcontract errorとして失敗させます。

## 4. パス契約

### PATH-001 — path roleの一元化

`data_root`、`checkpoint_root`、`artifact_root`、`output_root`、`cache_root`、external asset rootなどのpath roleは、canonical typed path contractとresolverから解決します。各task、runner、datamodule、model factory、scriptが同じroleを独自に組み立ててはなりません。

### PATH-002 — 解決基準を明示する

repository root、workspace root、scene rootなど、相対pathの基準はcontractで一意にします。current working directory、呼び出し元moduleの位置、偶然存在するdirectoryへ依存して意味を変えてはなりません。CWD基準がCLIの意図したpublic contractである場合だけ、明示的に使用できます。

### PATH-003 — 推測による探索を行わない

存在しないpathに対して、別root、旧directory、別拡張子、過去artifactを無制限に探索して処理を継続してはなりません。複数候補の探索が正当な機能である場合は、候補集合、優先順位、曖昧時の失敗条件をpublic contractとして定義します。

### PATH-004 — path変更も破壊的移行として扱う

canonical pathを変更する場合は、全consumer、config、dynamic target、test、documentationを同じ変更で更新し、旧path forwarding wrapperやsymlink前提のcompatibility routeを残しません。外部assetの配置契約を変更する場合は、必要な移行手順を明示します。

## 5. 責務と配置

### OWN-001 — one responsibility, one owner

同じ責務には一つのcanonical ownerと一つのcanonical implementationだけを置きます。共有helperの追加だけで既存実装を残したり、同じ意味のclass・function・schemaをtaskごとに複製したりしてはなりません。

### OWN-002 — 配置規則

canonical placementは次を原則とします。

- task非依存の汎用責務: `src/utils/**`。
- 複数taskで共有するdata、training、inference、visualizationなどのlifecycle責務: `src/tasks/base/**`。
- task固有のdomain責務: `src/tasks/<task>/**`。
- 外部実装との接続責務: repository-owned wrapperまたはadapter。vendor内部へproject固有logicを混入させない。

共有化は、import回数ではなく責務と依存方向で判断します。特定taskのschemaや意味へ強く依存する実装を、見かけ上の再利用性だけで`src/utils`へ移してはなりません。

### OWN-003 — architecture layerを分離する

schema/contract/IO、data、model/component、training、inference/pipeline、rendering/visualization、entrypointは、責務別のmoduleまたはsubpackageへ分離します。複数layerの実装を一つの巨大moduleや無秩序なpackage直下へ集約してはなりません。

package rootは、canonical public APIまたはcomposition rootに限定します。内部実装を無差別に再exportして、実体のownerを不明確にしてはなりません。

### OWN-004 — consumerを確認してから統合・削除する

未使用・重複判定では、static importだけでなく、Hydra dynamic target、registry、factory、CLI、subprocess、plugin、package entrypoint、test fixtureを確認します。正当なconsumerまたは明示されたpublic APIを持たないrepository-owned implementationは削除します。

## 6. Canonical APIと互換性

### API-001 — one canonical name and import path

public class、function、schema、entrypointには、一つのcanonical nameとimport pathを定めます。package rootから公開する場合、そのpathをcanonical APIとし、内部pathを並列のpublic APIとして扱いません。

### API-002 — compatibility-only surfaceの禁止

旧import、旧symbol、旧call signatureを維持することだけを目的とした次の実装を禁止します。

- 代入alias。
- compatibility re-export。
- pass-through wrapper。
- deprecated shim。
- 旧pathから新pathへのforwarder。
- 意味を変える暗黙adapter。

正規のpublic abstractionとして独立した責務を持つwrapperやadapterは許可します。単に古いconsumerを動かすだけの層は許可しません。

### API-003 — atomic repository migration

canonical APIを変更する場合は、static import、dynamic target、registry、factory、CLI、subprocess invocation、tests、documentationを同じ変更で移行します。新旧treeを併存させる段階移行を既定戦略にしません。

repository外consumerへの互換性が必要な場合は、そのAPIが外部向けpublic contractであること、互換期間、削除条件を本書の例外として登録します。

### API-004 — 旧形式は明示的に失敗させる

削除済みpathやsymbolを利用した場合は、通常のimport errorまたは明確なcontract errorにします。別実装へ暗黙転送して成功させてはなりません。

## 7. Model I/Oとpipeline境界

### IO-001 — typed model I/O contract

model固有の入力と出力にはtyped contractを定義します。入力tensorのkey、dtype、device、rank、shape、mask semantics、単位、座標系、出力型を、model実行前に検証可能にします。

### IO-002 — adapterがmodel固有I/Oを所有する

batch構築、key変換、tensor整形、mask生成、model call構築、raw output decodeはmodel-specific adapterが所有します。training loop、inference loop、pipeline orchestratorへこれらを直接実装してはなりません。

### IO-003 — composition rootで一度だけbindingする

modelと対応adapterの選択・整合性検証は、外部factoryまたはcomposition rootで一度だけ行います。loopや`forward`の途中でmodel名、class、variant、output keyに基づいて実装を選択してはなりません。

### IO-004 — pipeline independence

pipelineは、binding済みの抽象I/O contractだけを利用します。具体的なmodel variant、batch layout、raw output dictionaryを認識してはなりません。modelを追加する際は、既存pipelineへvariant分岐を追加するのではなく、contract implementationとfactory registrationを追加します。

### IO-005 — validationはloopとforwardの前に完了する

model/adapter不整合、必須input欠落、dtype・rank・shape・semantic constraint違反は、training/inference loopのmodel call前に失敗させます。同じvalidationを各layerで繰り返してはなりません。

## 8. `nn.Module.forward`契約

### FWD-001 — forwardはtensor計算に限定する

repository-owned `nn.Module.forward`は、検証済みtensorとconstructor/factoryで解決済みのmodule stateに対する計算だけを行います。

### FWD-002 — forward内validationの禁止

入力契約確認を目的とするPython `assert`、`isinstance`、`raise ValueError`、`raise TypeError`、Python scalarへの変換を伴うshape/rank/device検査を`forward`へ置いてはなりません。

- static hyperparameterとmodule構成はconstructorまたはfactoryで検証する。
- external batchはdataset、collate、model I/O adapterなどのboundaryで検証する。
- output contractはadapterのdecode boundaryで検証する。

数値algorithmそのものに必要なtensor control flowはvalidationではありません。Dropout、stochastic depthなど、標準的なtraining-mode stochastic layerも許可します。

### FWD-003 — Python side effectと実装選択の禁止

`forward`は次を行ってはなりません。

- filesystemまたはnetwork I/O。
- logging、print、artifact保存。
- config mutation、global state mutation。
- module、parameter、optimizerの生成。
- backend、model variant、adapter、loss implementationの選択。
- 不正入力を補正して処理を継続するfallback。

## 9. 実行境界と失敗規則

### ENTRY-001 — executable entrypointを限定する

`if __name__ == "__main__":`は、次に限定します。

- repository rootの`scripts/**`。
- `src/**/scripts/**`。
- packageの`__main__.py`。

通常のlibrary moduleへdemo、print、乱数入力、assert、save/loadを含むmain guardを置いてはなりません。

### ENTRY-002 — smokeと運用処理を分離する

検証目的のsmoke処理は`tests/**`へ置きます。対話的または運用上必要な機能だけを明示的なscript entrypointとして提供します。library importにより処理やI/Oが開始されてはなりません。

### FAIL-001 — missing dependencyを隠さない

不足したdependency、dataset、checkpoint、config、contractをdummy、no-op、zero tensor、空結果、代替modelで隠してはなりません。機能としてoptionalな要素はtyped contractで明示し、無効時の意味を定義します。

### FAIL-002 — fallbackは意味を保存しなければならない

明示的なfallback policyを設ける場合は、入力と出力のsemantic contractを保存し、選択条件がdeterministicかつ観測可能でなければなりません。精度、座標系、shape、identity、artifact ownershipなどの意味を変えるfallbackは禁止します。

## 10. 文書と正本

### DOC-001 — 規約を重複させない

恒久的なarchitecture ruleは本書だけに記述します。`AGENTS.md`は本書を必読文書として参照し、package READMEは現在のowner、module、data flow、実行方法を説明します。READMEやcode commentへ本書の規約全文を複製してはなりません。

### DOC-002 — 実装案内を同時更新する

canonical owner、public API、entrypoint、package treeを変更した場合は、該当READMEと本書のenforcement mapを同じPRで更新します。READMEが存在しない大きなpackageを新設する場合は、探索起点となるREADMEを追加します。

### DOC-003 — 設計判断と移行履歴を分離する

IssueとPRは、なぜ変更したか、どの範囲を移行したか、どのevidenceで完了したかを保持する履歴です。本書は、merge後も存続する設計不変条件だけを保持します。一時的なbase SHA、削除対象一覧、作業順序を恒久規約へ混入させません。

## 11. 機械的な強制

### ENF-001 — 文書と検査を対応付ける

現在のcanonical enforcement surfaceは次です。pathを変更する場合は、本書と参照元を同じPRで更新します。

| 対象 | Canonical enforcement |
|---|---|
| 設定・path contract | `scripts/audit_configuration.py` |
| 設定audit entrypoint | `tests/e2e/development/test_configuration_audit.py` |
| module責務・canonical API・model I/O・forward・entrypoint | `tests/e2e/development/test_architecture_boundaries.py` |
| style・型・回帰 | repositoryのcanonical ruff、mypy、test、CI checks |

### ENF-002 — negative caseを検証する

正常系だけでなく、missing key、unknown key、旧key、旧import、不正shape、model/adapter mismatch、明示device/backendの利用不能などがfail-closedになることを検証します。

### ENF-003 — 恒久検査はcurrent-state invariantにする

新しい恒久規約の機械検査は、現在のsource treeとruntime contractを直接検証します。特定の移行base SHA、過去の削除一覧、one-time diffだけに依存させません。一回限りのmigration evidenceが必要な場合は、恒久invariantと責務を分離し、目的を明示します。

### ENF-004 — テストを規約の代替にしない

既存testが通ることは、暗黙fallback、重複owner、compatibility shimを正当化しません。本契約に反する既存testは、canonical architectureへ合わせて更新または削除します。ただし、algorithmic behaviorや外部public contractの意図しない変更を隠すためにtestを弱めてはなりません。

## 12. 例外手続き

本契約からの逸脱は、実装前または同じPRで次を満たす場合に限り許可します。

1. 対象rule IDを指定する。
2. 対象path、symbol、runtimeを最小範囲で特定する。
3. 契約どおりに実装できない技術的理由を示す。
4. ownerと、違反が拡大しない検証方法を示す。
5. removal condition、期限または追跡Issueを示す。
6. 下表へ登録する。

永続的に正当な設計変更である場合は、例外を追加するのではなく本契約自体を更新します。PR本文やcode commentだけの例外宣言は、merge後の正本にならないため無効です。

### Registered exceptions

| ID | Rule | Scope | Owner | Reason | Removal condition / tracking |
|---|---|---|---|---|---|
| — | — | — | — | 現在登録された例外はありません。 | — |

## 13. PR review checklist

architectureに影響するPRは、少なくとも次を確認します。

- [ ] 適用されるrule IDとcanonical ownerを特定した。
- [ ] 設定、path、APIの正本を複製していない。
- [ ] staticおよびdynamic consumerを同時に移行した。
- [ ] 旧key、旧path、alias、shim、dual routeを残していない。
- [ ] validationをcomposition/data/model-I/O boundaryへ置いた。
- [ ] pipelineと`forward`へmodel固有I/OやPython validationを混入させていない。
- [ ] 失敗系を含むmeaningful testまたはauditを追加・更新した。
- [ ] relevantなruff、mypy、test、CI checkが成功した。
- [ ] package READMEとenforcement mapが実装に一致している。
- [ ] 逸脱がある場合、本書のRegistered exceptionsへ登録した。

## 14. Provenance

本契約の初期版は、次の完了済みarchitecture migrationを恒久規約へ抽出したものです。

- [Issue #688 — 設定解決とパス管理を単一の正本へ統合する](https://github.com/Motoki0705/tennis-lab/issues/688)
- [PR #705 — 設定・パス契約を統一する](https://github.com/Motoki0705/tennis-lab/pull/705)
- [Issue #691 — 責務境界・モデルI/O契約を再構成する](https://github.com/Motoki0705/tennis-lab/issues/691)
- [PR #709 — アーキテクチャの責務境界とモデルI/O契約を再構成する](https://github.com/Motoki0705/tennis-lab/pull/709)
