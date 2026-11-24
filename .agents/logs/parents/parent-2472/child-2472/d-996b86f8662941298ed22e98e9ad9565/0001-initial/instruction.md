# Issue #2472 ワーカー指示書（担当: `memory-infra` — Pillar V 共有Memoryカタログ 1ページ ＋ Pillar VI 研究モデル・基盤 3ページ）

## 1. 役割と絶対制約

あなたは親 Issue `#2472` の `memory-infra` ワーカーとして、GitHub Wiki（`Motoki0705/tennis-lab.wiki.git`）に掲載する **Pillar V の残り 1 ページ（`Shared-Memory-Catalog`）および Pillar VI の残り 3 ページ（`Research-Models-PLCS-BLCS-SLCS`, `Installation-Storage-and-Outputs`, `Knowledge-Library-and-Annotation`）の計 4 ページ** を日本語 Markdown で執筆・検証します。

### 厳守事項（違反禁止）
1. **本体リポジトリの変更・commit・push・PR作成・Issueコメント投稿は一切行わないこと**。
   - 出力ファイルは必ずこの worktree 内の **`tmp/wiki_pages/<Slug>.md`**（`.gitignore` 対象ディレクトリ）にのみ書き出してください。
   - 終了時点で `git status --porcelain` が完全に空であることを確認してください。
2. **正本照合の徹底（静かなフォールバック・推測禁止）**:
   - 下の設計仕様は事前調査に基づく設計図です。執筆前に必ず worktree 内の正本ファイル（`.agents/memory/SUMMARY.md`, `.agents/memory/{workflow,decision,environment,gotcha}[local path]`, `.agents/memory/ops-report/SUMMARY.md`, `src/tasks/{plcs,blcs,slcs}[local path]`, `knowledge/reports/slcs-real-rgb.md`, `knowledge/summary.md`, `knowledge/README.md`, `knowledge/registry/{tasks,relations,deployments}.yaml`, `docs/installation/README.ja.md`, `LICENSES.ja.md`, `src/tasks/OUTPUTS.md`, `.agents/ops/storage_tier/{README.md,config.toml}`, `src/tasks/annotation/README.md`, `.agents/skills/{knowledge-control,annotation-workflow}[local path]`）を実際に読んで数値・件数・パス・仕様を裏取りしてください。
   - 万一、設計仕様の数値（例えば `.agents/memory/` の現在の件数や `knowledge/nodes/` の件数など）と repo 正本の状態が食い違う場合は **repo 正本の実測値を優先**し、最終報告に差分を明記してください。
3. **二重管理の回避**:
   - 共有 memory 全25件の本文丸ごとコピー、設定ファイル全文コピー、32文字以上の生ハッシュ値（SHA-256 や 40桁 git commit SHA）の記載は禁止です（資産のハッシュ値はコード管理が原則）。詳細は正本への `https://github.com/Motoki0705/tennis-lab/blob/main/<path>` リンクへ誘導してください。
4. **秘匿・ローカル情報の混入禁止**:
   - リテラル文字列としての `[local path]`・`[local path]`・`file:[local path]`・メールアドレス・トークン・秘密鍵を一切含めないでください（`[local path]` は repo 公式文書 `storage_tier/README.md` の標準マウントパスなので記載可ですが、`[local path]` や `[local path]` は例示であっても絶対に書かないこと）。

---

## 2. 共通フォーマット・リンク・Mermaid 規約（既存公開ページと統一）

各ページの先頭は以下の形式で統一してください（Pillar V は `Shared-Memory-and-AutoDream` へ、Pillar VI の 3 ページは `Research-and-Infra-Overview` へ戻るリンクを置く）:

```markdown
# <日本語ページタイトル>

| 項目 | 内容 |
|---|---|
| **想定読者** | ... |
| **検索キーワード** | `...`, `...` |
| **正本ファイル** | [`...`](https://github.com/Motoki0705/tennis-lab/blob/main/...), ... |

> ← [Pillar V: 共有 Memory と AutoDream の仕組みへ戻る](Shared-Memory-and-AutoDream)
（※Pillar VI の3ページでは: > ← [Pillar VI: 研究モデル・基盤リファレンス入口へ戻る](Research-and-Infra-Overview)）

---
```

- **Wiki 内部リンク記法**: `[[Slug]]` や `Slug.md` ではなく、標準 Markdown の **`[表示テキスト](Slug)`**（アンカー付きなら `[表示テキスト](Slug#anchor)`）を使ってください。
- **使用可能な全 32 Wiki スラッグ一覧**（これ以外の内部リンク先を作らないこと）:
  - `Home`, `Glossary-and-Search-Traps`
  - Pillar I: `Tennis-Scene-Pipeline-Overview`, `Tennis-Scene-History-and-Optimizations`, `Component-Person-Detection-and-Tracking`, `Component-CourtSide-Association-Alignment`, `Component-3D-Reconstruction-and-Tooling`
  - Pillar II: `Court-Detection-Overview`, `Court-Model-Evolution-and-Backbones`, `Court-Geometry-Postprocessing-and-Video-Shared`, `Synthetic-3DGS-Alignment-and-Generation`, `Synthetic-Scaling-Diversity-and-Pair-Studies`, `NHT-Rendering-and-Foreground-Compositing`
  - Pillar III: `Ball-Detection-Overview`, `Ball-2D-Detector-Baseline-and-Domain-Adaptation`, `Ball-Refiner-2D-Failure-and-3D-Transition`, `Ball-3D-Smoothing-and-Scene-Evaluation`, `Ball-Long-Sequence-and-Architecture-Exploration`, `Ball-Datasets-Annotation-and-Test-Contracts`
  - Pillar IV: `AI-Ops-Overview`, `Parent-Session-Lifecycle`, `Issues-Adoption-and-Dispatcher`, `Worker-Delegation-and-Waits`, `Review-and-Merge-Gate`, `Communication-Visibility-and-Audit-Logs`, `Host-Ops-and-Recovery`
  - Pillar V: `Shared-Memory-and-AutoDream`, `Shared-Memory-Catalog`
  - Pillar VI: `Research-and-Infra-Overview`, `Research-Models-PLCS-BLCS-SLCS`, `Installation-Storage-and-Outputs`, `Knowledge-Library-and-Annotation`
- **Mermaid 規約**:
  - 使用可能種別は `flowchart LR` / `flowchart TD` / `sequenceDiagram` / `stateDiagram-v2` のみ。
  - Mermaid ブロック内に `<br>` 等の HTML タグを絶対に入れないこと（改行は `"1行目\n2行目"` のように二重引用符内で `\n` を使う）。括弧 `()` `[]` や `:` `[local path]` を含むラベルは必ず `"..."` で囲むこと。
- **GitHub Alert 規約**: `> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!TIP]` を効果的に使い、**連続配置やネストはしない**こと。

---

## 3. 作成する 4 ページの詳細仕様

出力先ディレクトリ: `tmp/wiki_pages/`

### (1) `tmp/wiki_pages/Shared-Memory-Catalog.md`
- **タイトル**: `現行共有 Memory 要約カタログ（テーマ別ガイド）`
- **正本確認対象**: `.agents/memory/SUMMARY.md`, `.agents/memory/{workflow,decision,environment,gotcha}[local path]`, `.agents/memory/{workflow,decision,environment,gotcha}[local path]`, `.agents/memory/ops-report/SUMMARY.md`
- **必須セクションと内容**:
  1. **このページの読み方と現在の件数サマリ**:
     - 正本は `.agents/memory/SUMMARY.md` であり、共有 memory は「規則」ではなく「検証待ちの観察」であること（仕組みは [Shared-Memory-and-AutoDream](Shared-Memory-and-AutoDream) へリンク）。
     - 現在の実測件数（4層合計：例 `workflow` 1 / `decision` 7 / `environment` 4 / `gotcha` 13 = 全25件、`applies_to` の内訳、および隔離された `ops-report` 専用領域の件数）を worktree 実ファイルと照合して明記。
  2. **テーマ別 4 グループ完全カタログ表**（全エントリを漏れなく収録し、各エントリファイルへの `blob/main/.agents/memory/<layer>[local path]<slug>.md` リンク・層区分・要点・背景となった事象/人間が気をつけるポイントを1表ずつ整理）:
     - **グループ 1: 開発規律・PR・命名・文書方針**: `focused-pull-requests`, `one-off-experiment-code-not-in-src`, `semantic-implementation-names`, `public-docs-english-default`, `gitignore-generic-rules-hide-new-paths`, `precommit-mypy-follow-imports-skip`（他に追加分があれば含める）。
     - **グループ 2: AI 協調・指示・監視運用**: `broadcast-to-parents-check-targets`, `ask-only-open-decisions`, `agents-ops-local-failures-not-regressions`, `claude-background-monitor-dies-with-session` (`applies_to: claude`), `codex-followup-timer-script-path`, `codex-worker-defers-without-numeric-budget`。
     - **グループ 3: WSL2・GPU・共有 venv・NHT 実行環境**: `wsl2-host-ram-silent-kill`（24GB→40GB引上げ経緯）, `wsl2-vram-overflow-freezes-host`（16GB GPU `batch_size=4`）, `wsl2-nvidia-smi-polling-stalls-gpu`（`#1757` 1秒周期照会タイムアウト）, `wsl2-distro-restart-keeps-boot-id`, `shared-venv-lags-lockfile`, `src-imported-from-cwd`, `tests-dir-not-a-package`, `main-submodule-checkout-lags-pointer`, `setup-nht-binds-global-tool-to-checkout`。
     - **グループ 4: 資産整理・Colab・長時間ジョブ運用**: `ckpt-force-delete-policy`, `colab-single-session-sequential`, `colab-idle-vm-reclaimed`, `bash-script-edit-while-running`。
  3. **人間向けメンテナンス・昇格ガイド**:
     - どのような発言が `decision/` に記録されるか（人間本人の発言のみ）、規則（`AGENTS.md` / skill / `README`）への昇格と削除のサイクル、`ops-report` 継続観察の確認方法。

### (2) `tmp/wiki_pages/Research-Models-PLCS-BLCS-SLCS.md`
- **タイトル**: `研究用3D推定モデル（PLCS・BLCS・SLCS）の現在地`
- **正本確認対象**: `src/tasks/plcs/README.md`, `src/tasks/blcs/README.md`, `src/tasks/slcs/README.md`, `knowledge/reports/slcs-real-rgb.md`, `knowledge/summary.md`, 該当する代表ノード（`plcs/000067,000087,000090,000111,000122,000123`, `blcs/000043,000068,000069`, `slcs/000009,000066,000069,000071,000075,000077,000079,000082,000139`）
- **必須セクションと内容**:
  1. **全体位置づけ比較表（標準 `tennis_scene` との境界と廃止済み系統）**:
     - **重要注意**: PLCS・BLCS・SLCS の 3 モデルはいずれも**現行の標準 `tennis_scene` パイプライン（[Tennis-Scene-Pipeline-Overview](Tennis-Scene-Pipeline-Overview)）には組み込まれていない研究用モデル**であること。
     - 3 モデル比較表（入力視点、学習データ、ノード数、現行維持契約 vs 廃止済み契約［`multi_object`・追跡・残差補正・`camera_view_v2` 専用データ廃止］、現在の到達ステータス）。Mermaid 図で入力モダリティと出力対象を対比。
  2. **PLCS（複数視点プレーヤー 3D 位置・向き・姿勢）**:
     - `PLCSMultiViewAxialModel` と `single_object` / `physical_v1` 契約、ACCAD/SMPL-H 変換。
     - **位置と回転の Pareto 構造（`plcs/000067`）**: 位置最良（`S=6/H=0 + auxoff + posw8`: `0.166 m`）と回転最良（`S=4/H=4 + posw4`: `8.01°`、`#560` で `7.54°`）で最適容量配分が分かれる知見。
     - Canonical temporal 分解（`000087`: centered 相関 `0.795` で平均姿勢退化を脱出するも高周波 jitter `0.391` vs GT `0.069` が過剰）、reprojection loss（`000090`）、廃止された残差補正（`000111`）・固定 track Re-ID（`000122`–`000123`）、次の課題。
  3. **BLCS（複数視点ボール 3D 軌道・物理損失）**:
     - `BLCSMultiViewAxialModel` と `blcs_generated_dataset_v3`（`blcs/000043`）。
     - **`#1044` 段階的物理損失比較（`000068`）**: 候補 B が主対照比 3 seed 平均 test 軌道 RMSE `1.146 → 0.930 m`（確認 seed43/44 `1.199 → 1.021 m`）へ改善した一方、短 clip・6view・rollout 速度残差・欠損 seed44 で退行層があり、ODE 積分出力は不安定（GT区間 `4.23–4.34 m`）。
     - **128-frame 非重複窓境界の誤差集中**: 差分速度 SSE の **80.55%**・加速度 SSE の **93.87%** が窓境界に集中している構造的ボトルネック、推論重み契約 `*-inference-v1.ckpt` 分離（`000069`）、次の課題。
  4. **SLCS（単眼実 RGB 時系列統合による選手・ボール同時 3D 推定）**:
     - Entity/Temporal 交互 attention + 10 frame 毎スパース DINOv3 tokens + RoPE、`real_rgb_v1`（Meiji 56 + broadcast 5 clip, 疑似3D教師）。
     - 実 RGB 総括（`slcs-real-rgb.md`, `slcs/000075`）: full ball `2.5210 m` が定数 baseline (`7.6922 m`) と `no_rgb` (`2.6813 m`) を上回り実 RGB の寄与を確認した一方、p95 `7.6934 m`・最大速度 `481.77 m/s` が残り頑健性未達。
     - 5 つの改善施策比較表（`gap48`, `velocity`, `missing_ball_court`, `TemporalContext`, `DomainBalanced`, `OneSidedContext` — いずれも他 domain 退行等で全面置換見送り）、固定 test 239窓の温存、`clip_000` v2 dataset 再生成（`000139`）と今後の課題。

### (3) `tmp/wiki_pages/Installation-Storage-and-Outputs.md`
- **タイトル**: `環境構築・16資産ライセンス・出力規約・[local path]`
- **正本確認対象**: `docs/installation/README.ja.md`, `LICENSES.ja.md`, `src/tasks/OUTPUTS.md`, `.agents/ops/storage_tier/README.md`, `.agents/ops/storage_tier/config.toml`
- **必須セクションと内容**:
  1. **動作環境とクリーン導入フロー（Mermaid `flowchart LR`）**:
     - 前提環境（Linux x86-64 / WSL2, Python 3.11, `uv sync --locked`, CUDA toolkit 13.0.88 + driver 580+ / PyTorch cu130, 所要容量 約 20.87 GB, RTX 5060 Ti 16GB 実測ピーク VRAM `4,928 MiB`）。
     - `third_party/{DINO,dinov3}` submodule 初期化、`build_dino_extension.sh`（`PYTHONPATH` / `TENNIS_LAB_DINO_EXTENSION`）、モデル重み・サンプルデータ・第三者16資産配置、`preflight` 検査、サンプルパイプライン実行までの流れ（コマンド詳細は `docs/installation/README.ja.md` へリンク）。
  2. **外部 16 資産のライセンス区分と取得トラブル対処**:
     - ライセンス区分要約表（非商用限定：GVHMR / SMPL-X / Ball 重み、申請制：DINOv3 / SMPL-X、利用条件未確認・暫定研究使用・再配布禁止：`AFLink_epoch20.pth` 等、正本 `LICENSES.ja.md` への誘導、※SHA-256 生ハッシュ値は書かずコード検査 `preflight` へ誘導）。
     - symlink 配置禁止契約、GVHMR Google Drive Quota 超過時の検査手順と任意 HF ミラー（`camenduru/GVHMR`）からの復旧案内。
  3. **論理出力パス規約（`src/tasks/OUTPUTS.md`）**:
     - 4 つの root（`OUTPUT`, `ARTIFACT`, `DATA`, `CHECKPOINT`）の役割表、`<task>[local path]<purpose>[local path]<experiment>[local path]<run-id>[local path]` 4階層命名、`resume`（完全状態再開）vs `init_weights`（重みのみ読込）の厳格分離、UI からの実行要求 IPC（`.training_queue/ui_requests/`）。
  4. **物理 [local path] ローカルストレージ階層（`.agents/ops/storage_tier/`）**:
     - **D:（WSL ext4 `relatime`・高速作業実体）** と **E:（`[local path]` HDD ext4 `noatime`・正本ミラー）** の二層構造図（Mermaid）。
     - 管理区分表：`resident_units`（永久保持 6 単位）、`dataset_containers` / `dataset_units`、`unmanaged_data_links`（E: 直リンク）、`ckpt/`（削除禁止）、`.worktree-outputs` の E: 集約。
     - Worktree 作成時の初期化手順（`prepare-worktree`）、14日 `atime` + `F_SETLEASE` + 実プロセス照合による安全 `evict`、クラッシュ時 `recover`、`d_capacity`（空き 20 GiB 門番）との関係。

### (4) `tmp/wiki_pages/Knowledge-Library-and-Annotation.md`
- **タイトル**: `` `knowledge/` 実験知識ベースと注釈運用基盤 ``
- **正本確認対象**: `knowledge/README.md`, `knowledge/registry/{tasks,relations,deployments}.yaml`, `.agents/skills/knowledge-control/SKILL.md`, `src/tasks/annotation/README.md`, `.agents/skills/annotation-workflow/SKILL.md`
- **必須セクションと内容**:
  1. **`knowledge/` の全体構造と 3 文書の役割分担**:
     - 全体構成（16 トピック［15 active + 1 retired `ball_refiner`］、現在のノード総数・Papers・`citation_map`・`reports` の実測値）とディレクトリ地図。
     - **`summary.md`（現在の判断のみ・20,000字上限）vs `history/summary-2026-11-05.md`（凍結された過去経緯）vs `nodes/`（1 run = 1 ノードの正本）** の読み分けと `summary_impact: updated | none` 契約。
  2. **`registry/` 3 正本・ストレージ上限・検索と Web UI**:
     - `tasks.yaml` / `relations.yaml`（8関係語彙）/ `deployments.yaml`（本番パイプライン配備 ckpt と根拠ノードの 1対1 強制照合）。
     - `runs/` 容量ガード（通常ファイル 256 KiB / 図 512 KiB / 1 run 合計 2 MiB、ckpt/NPZ/動画禁止、`artifact-manifest.json` による削除追跡）。
     - 横断検索 CLI `kg_query.py`（`find`, `show`, `metrics`, `tasks`, `deployments` の実用コマンド例）と Next.js / React Flow 製 Web UI（`knowledge/webui/`）、`Papers/`（人間採録）と `citation_map/`（自動引用探索）の分離、`Knowledge library` CI 検査。
  3. **注釈・レビュー・継続学習基盤（`src.tasks.annotation`）**:
     - `base/annotation` 共通基盤と `ball` / `player` adapter 分離（旧 `chat_annotation` 撤去）のアーキテクチャ Mermaid 図。
     - **系統 1: Web 取得キャンペーン**（選別 → 準備 → 予測 → `prediction_preview` → Codex レビュー `worker-decision.toml` → snapshot 公開 → 追加学習；**現在は `ball/frames` のみ production 対応**で人物 bbox/pose は拒否される契約）。
     - **系統 2: ローカル動画の白紙注釈登録と自動継続**（`configure_local_videos` + `LocalAnnotationController`：1 attempt **5,400 秒**、進捗増分検証による自動 `authorize_retry`、停止種別［`timed_out` / `invalid_saved_progress` / `quota` / `failed` / コード更新時］ごとの運用対応表）。

---

## 4. 完了前の機械自己検証（必須実行）

執筆後、`.venv/bin/python` で以下の検査を実行し、**エラー 0 件**を確認してください:
1. `tmp/wiki_pages/` 配下に上記 4 つの `.md` ファイルがすべて存在し、十分な分量と構造を持つこと。
2. 全 `.md` 内の `https://github.com/Motoki0705/tennis-lab/blob/main/<path>` リンク（`#` 以降を除いたパス）が worktree 内にすべて実在すること。
3. 全 `.md` 内の内部 Wiki リンク先が上記 32 スラッグのいずれかに完全一致すること（`.md` 拡張子付きや `[[...]]` が残っていないこと）。
4. Mermaid コードブロック内に `<br>` が無く、連続/ネストした `> [!` アラートが無いこと。
5. リテラルの `[local path]`・`[local path]`・`file:[local path]`・メールアドレス・トークン・32文字以上の16進文字列（`\b[0-9a-fA-F]{32,}\b`）が **0 件**であること。
6. `git status --porcelain` の出力が空であること。

最終報告には、作成した 4 ファイルの絶対パス・行数・検証結果・正本照合で確認した特記事項を簡潔に記載し、最終行に必ず次の 1 行だけを置いて終了してください:

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}
