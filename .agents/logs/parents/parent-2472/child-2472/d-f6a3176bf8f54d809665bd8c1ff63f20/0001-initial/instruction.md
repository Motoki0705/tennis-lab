# Issue #2472 ワーカー指示書（担当: `ball` — Pillar III ボール検出・平滑化・3D Refiner 詳細 5ページ）

## 1. 役割と絶対制約

あなたは親 Issue `#2472` の `ball` ワーカーとして、GitHub Wiki（`Motoki0705/tennis-lab.wiki.git`）に掲載する **Pillar III（ボール検出・平滑化・3D Refiner）の詳細 5 ページ** を日本語 Markdown で執筆・検証します。

### 厳守事項（違反禁止）
1. **本体リポジトリの変更・commit・push・PR作成・Issueコメント投稿は一切行わないこと**。
   - 出力ファイルは必ずこの worktree 内の **`tmp/wiki_pages/<Slug>.md`**（`.gitignore` 対象ディレクトリ）にのみ書き出してください。
   - 終了時点で `git status --porcelain` が完全に空であることを確認してください。
2. **正本照合の徹底（静かなフォールバック・推測禁止）**:
   - 下の設計仕様は事前調査に基づく設計図です。執筆前に必ず worktree 内の正本ファイル（`src/tasks/ball_detection/`, `src/tasks/ball_refiner_3d/`, `src/tennis_scene/configs/{pipeline,pipeline_x3d_dpt}.yaml`, `src/tennis_scene/evaluation/README.md`, `knowledge/summary.md`, `knowledge/registry/{deployments,tasks}.yaml`, 該当する `knowledge/nodes/{ball_detection,ball_refiner,ball_refiner_3d,tennis_scene,court_side,annotation,player_pose_annotation}[local path]`）を実際に読んで数値・設定キー・ファイルパス・経緯を裏取りしてください。
   - 万一、設計仕様の数値やパスと repo 正本の記述が食い違う場合は **repo 正本を優先**し、最終報告に差分を明記してください。
3. **二重管理の回避**:
   - 設定 YAML 全文コピー、CLI `--help` 全量貼り付け、32文字以上の生ハッシュ値（SHA-256 や 40桁 git commit SHA）の記載は禁止です（モデルファイル名や短い run ID は可）。詳細は正本への `https://github.com/Motoki0705/tennis-lab/blob/main/<path>` リンクへ誘導してください。
4. **秘匿・ローカル情報の混入禁止**:
   - リテラル文字列としての `[local path]`・`[local path]`・`file:[local path]`・メールアドレス・トークン・秘密鍵を一切含めないでください（例示でも `[local path]` 等は書かず「マシン固有のホームパス」等と表現すること）。

---

## 2. 共通フォーマット・リンク・Mermaid 規約（既存公開ページと統一）

各ページの先頭は以下の形式で統一してください:

```markdown
# <日本語ページタイトル>

| 項目 | 内容 |
|---|---|
| **想定読者** | ... |
| **検索キーワード** | `...`, `...` |
| **正本ファイル** | [`...`](https://github.com/Motoki0705/tennis-lab/blob/main/...), ... |

> ← [Pillar III: ボール検出・再構成 概要と評価契約マップへ戻る](Ball-Detection-Overview)

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
- **実験比較表の必須列**: 実験比較表には必ず **「結論 / 採否（採用・棄却・opt-in採用・見送り・未完 等）」列** を設け、異なる評価契約（分母・解像度）を混同しないよう評価データセット・母数を明記してください。
- **Mermaid 規約**:
  - 使用可能種別は `flowchart LR` / `flowchart TD` / `sequenceDiagram` / `stateDiagram-v2` のみ。
  - Mermaid ブロック内に `<br>` 等の HTML タグを絶対に入れないこと（改行は `"1行目\n2行目"` のように二重引用符内で `\n` を使う）。括弧 `()` `[]` や `:` `[local path]` を含むラベルは必ず `"..."` で囲むこと。
  - Mermaid で `click` を使う場合は相対パスではなく `https://github.com/Motoki0705/tennis-lab/wiki/<Slug>` の完全 URL を使うこと。
- **GitHub Alert 規約**: `> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!TIP]` を効果的に使い、**連続配置やネストはしない**こと。

---

## 3. 作成する 5 ページの詳細仕様

出力先ディレクトリ: `tmp/wiki_pages/`

### (1) `tmp/wiki_pages/Ball-2D-Detector-Baseline-and-Domain-Adaptation.md`
- **タイトル**: `2Dボール検出ベースラインと明治ドメイン適応`
- **正本確認対象**: `src/tasks/ball_detection/README.md`, `knowledge/registry/deployments.yaml`, `knowledge/nodes/ball_detection/000001`〜`000022`, `knowledge/nodes/ball_refiner/000012`〜`000015`, `000028`, `knowledge/nodes/tennis_scene/000009`〜`000010`, `knowledge/nodes/court_side/000001`〜`000003`
- **必須セクションと内容**:
  1. **初期バックボーン探索と失敗（`#551`, `#579`）**:
     - DINOv3+RoPE（F1 `0.080–0.094` で失速）と Webmix 多段階学習（T=1 崩壊 `0→0.035` は防ぐが多フレーム `0.106→0.048` で退行）の教訓。
  2. **`conv_next_unet` + サブピクセル精密化（`#618`）と「offline F1 最高値だけで採用しない」原則**:
     - 72×128 格子量子化律速（F1 `0.4111`）→ `subpixel_refine`（`0.5764`）→ `ft-e13`（TrackNet Test F1 **`0.7218`**, P `0.7357` / R `0.7084`, dist `2.18 px`）への到達表。
     - **重要教訓**: `scratch` ep25 は Test F1 **`0.7692`** で `ft-e13` を上回ったが、実動画 `tennis_clip` で coverage 低下と **`179.9 px` テレポート**を起こしたため却下。
  3. **3DGS×BLCS 合成データ混合実験（`ball_detection/000011`〜`000016`）**:
     - TrackNet game9 val F1 統制 `0.6729` vs 1/6 混合 `0.6631` (`-0.0098`) vs 1/12 半率 s731 `0.6914` (`+0.0185`, 追加検証未完)。
  4. **ImageNet 正規化バグ修正と `#934` → `#935` 指標逆転による現行既定 `i935-mixed-ft-s42-epoch09` 採択**:
     - 正規化漏れ修正で Meiji `cam0` 欠測率 `84.1% → 62.5%`（`tennis_scene/000009`–`000010`）。
     - `#934`（閾値 F1 で `ep0` 選択も p95 `346.08→386.30 px` 悪化で見送り）から `#935`（閾値なし候補 **`recall@8`** `<=20px` へ切替え、`ft-e13` `74.25%` → `e0` `86.96%` → **`e9` `90.84%` [20,900/23,007]** 発見、閾値 F1 は `0.475` へ低下）への転換比較表と Mermaid フロー。
     - `clip_000` 反実仮想（`court_side/000003`）: `ft-e13+gate`（margin `0.1185` 停止、`cam0` recall `0.2568`, p90 `312.7 px`）vs `e9+gate`（margin `0.4070` 通過、`cam0` recall `0.7895`, p90 `11.30 px`）と `2026-10-01` 採用判断（根拠ノードが廃止トピック `knowledge/nodes/ball_refiner/` に格納されている検索上の注意点も明記）。

### (2) `tmp/wiki_pages/Ball-Refiner-2D-Failure-and-3D-Transition.md`
- **タイトル**: `2D Ball Refiner 廃止の経緯と 3D Refiner 転換`
- **正本確認対象**: `src/tasks/ball_refiner_3d/README.md`, `knowledge/registry/tasks.yaml`, `knowledge/nodes/ball_refiner/000001`〜`000047`, `knowledge/nodes/ball_refiner_3d/000001`〜`000015`, `knowledge/nodes/court_side/000004`〜`000006`
- **必須セクションと内容**:
  1. **2D Temporal MDN Refiner（`ball_refiner`）の推移と限界**:
     - `r4`（p95 改善の一方で中央値 `10.3 → 25.3 px` 悪化・HDR95 `85.9%` 過信）→ `anchored_12k`（`000018`: Meiji obs med `3.11 px`）→ LOCO 共分散倍率 `1.8125`（`000019`: HDR50 `~63.7%` 過大被覆・TrackNet/gap NLL 悪化）→ Seed 再現 **9/10 FAIL**（`000023`）。
  2. **2 つの決定的棄却ゲート：Source B Gate と `court_side` 安全ベンチ**:
     - **Source B Gate（`000025`–`000026`）**: 720p JPEG90 キャッシュ vs 元 MP4 直推論で `clip_010` p90 **`94.35 → 140.69 px (+46.34 px)`** 悪化（`cam2` `55.05 → 158.18 px`）。圧縮微小差が MDD + Top-K ランキングで非線形増幅する構造を特定。
     - **`court_side` 安全ベンチ（`court_side/000004`–`000006`）**: Confidence filter（`p>=0.9, area<=30000`）が 11,200 試行で誤判定 `0 → 3` / `0 → 2` を起こし `2026-10-02` 廃止。
  3. **座標 Refiner 比較と `2026-10-06` 2D Refiner 正式廃止の原則**:
     - BLCS `single_object` 15条件（`000040`）と RoPE+GAN 検証（`000044`, `000047`: GAN 単独は 2D `289.49 px` / 3D `12.313 m` へ崩壊）を経て確立された原則：**「三角測量前に 2D 幻覚補完を混ぜず、欠損は実観測 mask のまま 3D 側で扱う」**（Mermaid 図で旧2D補完経路 vs 現行3D経路を対比）。
  4. **`ball_refiner_3d` の到達点と標準パイプライン未統合の理由（`#1014`, `#1032`）**:
     - `#1014`（RoPE 8L + Event Head, `ball_refiner_3d/000011`）: test RMSE `0.158 m`、missing **`0.428 m`**（線形 `0.896 m` 比 -52%）、event Brier `0.00324`。しかし**正確な観測点も約 `6.8 cm` 改変**し、速度振動 `3.01 m/s`・OOD ノイズ `1.952 m` 崩壊が残る。
     - `#1032` 物理 Head 3 方式比較表（`000012`–`000015`: A coords-only / B params head / C ODE再構成+整合性勾配）：A が直接誤差 `0.162 m` / F1 `0.980` 最良だが非物理加速度率 `40.6%`（missing `98.9%`）・jerk比 `2,343`；B (`0.265 m`)・C (`0.257 m`, 予測区間抽出失敗 68–70%) は直接誤差も F1 も悪化 → 既定採用見送り・現在 `tennis_scene` 未統合。

### (3) `tmp/wiki_pages/Ball-3D-Smoothing-and-Scene-Evaluation.md`
- **タイトル**: `ボール3D平滑化（LOESS）と凍結LOVO 6方式評価`
- **正本確認対象**: `src/tennis_scene/evaluation/README.md`, `src/tennis_scene/README.md`, `knowledge/nodes/tennis_scene/000031`〜`000041`
- **必須セクションと内容**:
  1. **CPU `ball_points` v4（局所2次コンセンサス＋ロバスト2次 LOESS）の仕様とトレードオフ（`#1554`, `tennis_scene/000031`）**:
     - アルゴリズムの流れ（`trajectory_gate` → 局所2次コンセンサスで `108/2,423` 点棄却 → 信頼度重み付き2次 LOESS）と最重要不変条件：**局所補間点は診断保存のみで三角測量には渡さない**。
     - 定量効果（有効3D `821 → 859 / 1010`、3Dステップ p95 `0.688 → 0.412 m`）と代償（生2D再投影中央値 `3.554 → 4.810 px`、バウンド/ヒット前後の微小悪化、view 切替時の最大ジャンプ `1.077 m` 残存）。
  2. **凍結 `clip_000` 3D LOVO 評価基盤の設計原則（`#1556`, `000032`–`000033`）**:
     - `FixedInputBundle`、前半 `[0, 505)` 調整 / 後半 `[505, 1010)` 主評価の Leave-One-View-Out (LOVO) 再投影設計、および **Coverage 見かけ増加の罠を防ぐ `paired_error_summary`（共通集合比較）必須ルール**。
  3. **誤差分解から見えた真のボトルネック（`#1558`, `000034`）**:
     - detector → annotation oracle 改善（mean `+1.777 px`, p90 `+6.432 px`）、argmax 失敗 222 点中 **57.2% (127点)** が top-8 内にある一方、固定 mask 下の candidate8 oracle 改善は p90 `+0.187 px` に留まる事実、時刻同期ずれとカメラ校正誤差の交絡。
  4. **後処理 6 方式 LOVO 統合比較表（`#1559`〜`#1568`, `000035`–`000041`）と `2026-11-05` Baseline 維持決定**:
     - 全方式比較表（Baseline LOESS有 `9.340 / 27.078 px` [cov `56.43%`]、`without_loess` `9.798 / 29.810 px`、`annotation_oracle` `7.859 / 20.722 px` [`92.00%`]、Heatmap decoding `#1559`、Top-K 多仮説 `#1560` [cov `36.46%` 崩壊]、Candidate GMM `#1563` [all-view `420→62` 崩壊]、Dense heatmap `#1564` [cov `99.73%` も新規633点が p90 **`313.39 px`**・共通825点も悪化]、Segment 物理+時刻offset `#1565`、Court+Ball BA `#1567` [court holdout 悪化]、統合 20+4 条件 `#1568`）。
     - 結論：PR `#1672`〜`#1682` 不採用クローズ・後処理 `#1683` 取消、検出器本体の改善へ回帰。

### (4) `tmp/wiki_pages/Ball-Long-Sequence-and-Architecture-Exploration.md`
- **タイトル**: `長系列 720p 5ch・X3D/CSN・mHC・MSDA 探索`
- **正本確認対象**: `src/tasks/ball_detection/README.md`, `src/tennis_scene/configs/pipeline_x3d_dpt.yaml`, `knowledge/nodes/ball_detection/000023`〜`000060`, `000068`〜`000079`, `000081`〜`000094`, `000096`, `knowledge/nodes/tennis_scene/000043`, `000052`, `000065`
- **必須セクションと内容**:
  1. **720p 5ch（RGB+MDD）化と学習 I/O 工学（`#986`, `000023`〜`000051`）**:
     - CPU JPEG+MDD `0.57–1.85 win/s` から、GPU MDD + BF16 `torch.compile` + nvJPEG + `preadv`（mmap page fault `0.175s → 0.044s`）+ 8 worker/prefetch 4 により **`14.7 win/s`**（reader 待ち `0.4%`）へ至るボトルネック解消過程。
  2. **2 段階学習（DPT → Query）への転換と Camera Stress 脆弱性（`#1049`[local path]`#1050`, `000052`〜`000060`）**:
     - Scratch query 崩壊（`~237 px`）→ DPT 事前学習＋Query 後段で Clean Common `13.49 px` / Full `15.38 px` 到達も、**Camera Stress `32.21 px`・1/4 FPS `41.11 px`・合成遮蔽 `32.94 px`** で脆弱性が露呈（※この評価プロトコルは `#1514` 以降と分母・教師枠が異なること、また `#1231` `000092` は原本 JSON 喪失のため順位除外することを注記）。
  3. **同一 Validation（85,473 observed / 31,885 absent）統制比較表（`#1514`, `#1675`, `#1899`, `#1859`）**:
     - 全主要実験を 1 つの比較表に整理：
       - `#1514`: `convnext common-dpt-T32` (`42.58 px`, `<=20px 84.03%`) / `dpt-T64` (`36.56 px`, **`<=20px 86.58%`**) / `query-T32` (med `24.20 px` の未解明系統ズレ) / **`query-T64`（mean `28.20 px`, med `7.66 px`, P95 `116.89 px` 最良だが `<=20px 76.29%` かつ presence head 欠落）**。
       - `#1675`: **`x3d_csn common-dpt-T32`（mean `34.43 px`, med `8.87 px`, `<=20px 86.14%` → `pipeline_x3d_dpt.yaml` 採用）**；X3D + 4-query mHC 7条件（12k `32.48–35.38 px`、最良 `d512-l8` `32.48 px`、12L/24L 保留）。
       - `#1899` 1要素統制 12k（`000089`–`000094`）: ConvNeXt+4q mHC (`36.42 px`)、X3D+1q (`33.98 px`)、ConvNeXt+1q T64→T64 (`28.53 px`, L4→5060Ti 再開) — いずれも `#1514` `28.20 px` を更新せず。
       - `#1859` 構造効率化・MSDA（`000079`, `000091`, `000096`）: CNN配置最適化 C2 `-22.4%` (`4.265→3.308 ms/f`)、MSDA 4-level OOM・3-level T1/T2 は学習 `~1.95×` 低速、RTX 5060 Ti で T1 12k `35.34 px`、C3 再開中（未完）。
  4. **明示 opt-in `pipeline_x3d_dpt.yaml` の接続と BF16 決定性特例（`#2150`, `tennis_scene/000043`, `000065`）**:
     - `meiji_4cam` `clip_000` (cam1-3) での `i935` 停止突破（`123/356` 有効3D）と `video_shared` `court_side` 全69 clip 解決、および BF16 最大差 `16.38 px`（人間承認済み特例）・`video_batch` 補助9配列超過の注意点。

### (5) `tmp/wiki_pages/Ball-Datasets-Annotation-and-Test-Contracts.md`
- **タイトル**: `ボールデータセット・注釈・独立Test契約の現状`
- **正本確認対象**: `src/tasks/ball_detection/README.md`, `src/tasks/annotation/README.md`, `knowledge/nodes/ball_detection/000061`〜`000067`, `000080`, `000095`, `knowledge/nodes/annotation/000001`〜`000002`, `knowledge/nodes/player_pose_annotation/000001`〜`000003`
- **必須セクションと内容**:
  1. **全フレーム教師 128-frame 共通パックと Colab 可搬配布（`#1367`, `#1369`, `#1374`, `#1375`）**:
     - 中央16枠案棄却 → 128-frame 共通区間全フレーム教師（`T32×4 = T64×2 = T128×1` 等価バッチ契約、Chat notes 付き空ラベル `14,347` frame 除外）。
     - `train-val-v1`（1,537 clip / 793,017 common frames / 157.03 GB / 2,926 files）の Drive 配布と Colab VM 跨ぎ再開実証（`max_loss_diff 8.97e-7`）。Mermaid 図でデータフローと Split 境界を図解。
  2. **【最重要警告】`#1476` 独立 Test（225 clips）の断絶と Validation 不在ラベルの Chat バイアス**:
     - `#1476`（`000080`）: 225 test clips（143,074 common / 127,202 scored frames）契約確定後、Drive 転送途中停止で同名 `shards` フォルダ3個に分裂（4 payload のみ、`distribution.json` 欠落）、PR `#1489` 未マージクローズ、ローカル元 snapshot 欠損により **`#1514` 以降の全モデルが独立 test 未評価**である現状。
     - Validation 不在 `31,885` frame の 99.5% 超が `chat_annotation` 由来であり、**Meiji クリップには不在ラベルが 0 件（Meiji FPR = `null`）** という評価バイアス。
  3. **Store 版進化（`v1`→`v4` `active_ball`）と Meiji v3 移行（`#1714`, `000095`）**:
     - `meiji-active-ball-v3-20261113` への移行状況（`123/171` camera 完了、`63,540` frames；不整合・未完 `48` camera 保留）。
  4. **人間注釈キャンペーンとの連携（`#1088`, `#988`）**:
     - `#1088`（`annotation/000001`–`000002`）: ボール[local path] 専用本番注釈ループ（YouTube 898f パイロット `87.3s` 疎通、継続学習2サイクル未完；人物 bbox/pose は未対応拒否）。
     - `#988`（`player_pose_annotation/000001`–`000003`）: `review_then_pose.v1` による 1,172/1,300 clip 進行状況（基盤仕様の詳細は [Knowledge-Library-and-Annotation](Knowledge-Library-and-Annotation) へ相互リンク）。

---

## 4. 完了前の機械自己検証（必須実行）

執筆後、`.venv/bin/python` で以下の検査を実行し、**エラー 0 件**を確認してください:
1. `tmp/wiki_pages/` 配下に上記 5 つの `.md` ファイルがすべて存在し、十分な分量と構造を持つこと。
2. 全 `.md` 内の `https://github.com/Motoki0705/tennis-lab/blob/main/<path>` リンク（`#` 以降を除いたパス）が worktree 内にすべて実在すること。
3. 全 `.md` 内の内部 Wiki リンク先が上記 32 スラッグのいずれかに完全一致すること（`.md` 拡張子付きや `[[...]]` が残っていないこと）。
4. Mermaid コードブロック内に `<br>` が無く、連続/ネストした `> [!` アラートが無いこと。
5. リテラルの `[local path]`・`[local path]`・`file:[local path]`・メールアドレス・トークン・32文字以上の16進文字列（`\b[0-9a-fA-F]{32,}\b`）が **0 件**であること。
6. `git status --porcelain` の出力が空であること。

最終報告には、作成した 5 ファイルの絶対パス・行数・検証結果・正本照合で確認した特記事項を簡潔に記載し、最終行に必ず次の 1 行だけを置いて終了してください:

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}
