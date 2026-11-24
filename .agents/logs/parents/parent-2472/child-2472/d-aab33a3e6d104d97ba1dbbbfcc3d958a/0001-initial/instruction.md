# Issue #2472 ワーカー指示書（担当: `primary` — Pillar II コート検出・3DGS合成データ 詳細 5ページ）

## 1. 役割と絶対制約

あなたは親 Issue `#2472` の `primary` ワーカーとして、GitHub Wiki（`Motoki0705/tennis-lab.wiki.git`）に掲載する **Pillar II（コート検出 ＆ 3DGS合成データ）の詳細 5 ページ** を日本語 Markdown で執筆・検証します。

### 厳守事項（違反禁止）
1. **本体リポジトリの変更・commit・push・PR作成・Issueコメント投稿は一切行わないこと**。
   - 出力ファイルは必ずこの worktree 内の **`tmp/wiki_pages/<Slug>.md`**（`.gitignore` 対象ディレクトリ）にのみ書き出してください。
   - 終了時点で `git status --porcelain` が完全に空であることを確認してください。
2. **正本照合の徹底（静かなフォールバック・推測禁止）**:
   - 下の設計仕様は事前調査に基づく設計図です。執筆前に必ず worktree 内の正本ファイル（`src/tasks/court_detection/`, `src/tennis_scene/pipeline/video_court/`, `src/synthetic_data_generation/`, `knowledge/summary.md`, `knowledge/registry/deployments.yaml`, `knowledge/reports/`, 該当する `knowledge/nodes/court_detection/*`・`knowledge/nodes/synthetic_data_generation/*`, `paper/court_robustness/evidence/` 等）を実際に読んで数値・設定キー・ファイルパス・経緯を裏取りしてください。
   - 万一、設計仕様の数値やパスと repo 正本の記述が食い違う場合は **repo 正本を優先**し、最終報告に差分を明記してください。
3. **二重管理の回避**:
   - 設定 YAML 全文コピー、CLI `--help` 全量貼り付け、32文字以上の生ハッシュ値（SHA-256 や 40桁 git commit SHA）の記載は禁止です（12文字以下の短い識別子 `b863df1f01f0`, `dd3a39` 等のみ可）。詳細は正本への `https://github.com/Motoki0705/tennis-lab/blob/main/<path>` リンクへ誘導してください。
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

> ← [Pillar II: コート検出 概要・9フェーズ年表へ戻る](Court-Detection-Overview)

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
- **実験比較表の必須列**: `knowledge/` では仮説棄却も `status: done` になるため、実験比較表には必ず **「結論 / 採否（採用・棄却・部分成功・暫定維持 等）」列** を設けてください。また現行有効な契約と廃止済み仕様（`adaptive`, `paired_edges_v1` 等）を視覚的に区別してください。
- **Mermaid 規約**:
  - 使用可能種別は `flowchart LR` / `flowchart TD` / `sequenceDiagram` / `stateDiagram-v2` のみ。
  - Mermaid ブロック内に `<br>` 等の HTML タグを絶対に入れないこと（改行は `"1行目\n2行目"` のように二重引用符内で `\n` を使う）。括弧 `()` `[]` や `:` `[local path]` を含むラベルは必ず `"..."` で囲むこと。
  - Mermaid で `click` を使う場合は相対パスではなく `https://github.com/Motoki0705/tennis-lab/wiki/<Slug>` の完全 URL を使うこと。
- **GitHub Alert 規約**: `> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!TIP]` を効果的に使い、**連続配置やネストはしない**こと（間に通常段落や見出しを挟む）。

---

## 3. 作成する 5 ページの詳細仕様

出力先ディレクトリ: `tmp/wiki_pages/`

### (1) `tmp/wiki_pages/Court-Model-Evolution-and-Backbones.md`
- **タイトル**: `コートモデル進化・SSL・DINOv3 バックボーン比較`
- **正本確認対象**: `src/tasks/court_detection/README.md`, `src/tasks/court_detection/configs/`, `knowledge/nodes/court_detection/000001`〜`000006`, `000025`〜`000027`, `000034`〜`000044`, `knowledge/runs/group-i983-court-dinov3-shared1024-s42/`, `knowledge/runs/group-i983-court-throughput-s42/`, `paper/court_robustness/evidence/`, `knowledge/registry/deployments.yaml`
- **必須セクションと内容**:
  1. **アーキテクチャ遷移（KP14単体 → Line専用 → 4ヘッド+Pose DINOv3+Transformer+DPT）**:
     - 初期階層 KP14（`kp512`）と Line モデルから、統合 4 dense heads（`kp`, `seg`, `line`, `semantic_line`）＋ Pose 監視への進化。
     - 学習 config（`dinov3_dpt.yaml` 等）と現行配備 `b863df1f01f0.ckpt` の実設定比較（凍結 DINOv3-B + Transformer 8L `d=768` vs `#983` 共通後段 `d=1024` 128.6M params、DPT512、Residual head `depth3`×4 vs `depth2`、multiscale `[256..512]`、オンライン subpixel 教師等）。Mermaid 図でモデル構成を図解。
  2. **Phase 1: ドメイン特化 SSL（`#524`）と初代 KP512 配備（`#621`）**:
     - Wikimedia テニスコート上の DINOv3 ViT-B/16 LoRA SSL merge による `val/miou 0.517 → 0.800`（`000001`〜`000005`）。
     - PR `#613` constrained-retry 修正後の `kp512` 再開（2 epoch で `val/mean_dist 13.45 → 2.23 px`、epoch 21 最良 `1.708886 px`、手動 GT `tennis_clip` `0.76 px` vs ep17 `5.91 px`）と初代配備（`000006`）。
     - 負の知見: `test_dataloader` が実際には `data_val.json` を読んでいた不具合の発見。
  3. **Phase 4: LoRA+Depth2（`dd3a39...`, `#867`）から凍結 Multiscale-Depth3（`b863df1f01f0`）への逆転と来歴欠落**:
     - `#867`（`000025`→`000026`）は KP `15.01→3.44 px`・Line Dice `0.461→0.542` 改善の一方、Seg `0.859→0.802`・Pose `6.51→15.82 px` 悪化（広幅線・depth2・pose監視の複合変更で因果分離不能）。さらに Meiji `clip_000` 全画面で共通校正 `0/1010 frame` 失敗（`cam0` 平均 `701 px`）。
     - 対して凍結 DINOv3-B + multiscale + depth3 の `b863df1f01f0.ckpt` は領域探索と併用し全 `3,030 frame` 通過（手動 GT 比 `8.37–13.13 px`、`court:000032`）。
     - **検索の罠・来歴欠落の説明**: `knowledge/nodes/court_detection/` に `b863df1f01f0.ckpt` 本体の学習ノードが未登録であり、`paper/court_robustness/evidence/`（`training_provenance.json`, `checkpoint_config.json`, `checks.json`、旧予想ハッシュとのバイト不一致注記）と `#1031` にのみ来歴が残る事実（※生64桁SHAは書かず短縮表記＋正本ファイルリンクにすること）。
  4. **Phase 5: 凍結 DINOv3 バックボーン 4 サイズ比較（`#983`）と L4 スループット診断（`000034`〜`000044`）**:
     - `S / S+ / B / L` 比較表（共通 128.6M 後段、seed 42、全サイズ偶然 `epoch 17 / step 29,844` 最良）：Synthetic test (886) vs Real val (2,211) の主要指標（B が Syn KP `3.986 px` / SEG `0.9521` 最良、L が LINE `0.6501` / Semantic LINE `0.4619` / Pose `9.580 px`・`0.528 m`・`5.47%` 最良、Real KP は `1.343–1.368 px` で拮抗、S の Pose `184.28 px` 外れ値が `B01:court-sample-002077` 無効深度2点 = 総和の `91.65%` による解剖、有効点のみ `15.39 px`）。
     - L4 スループット診断表（`000040`〜`000042`）: `S+` batch `8 → 16` で `+3.03%`（`10.06 → 10.36 img/s`）に対し VRAM `9.1 → 16.2 GB`。真の律速は Drive I/O ではなく **128.6M 後段 backward（約 `517 ms/step`）とブロッキング checkpoint 同期（約 `43 GB/run`）**。

### (2) `tmp/wiki_pages/Court-Geometry-Postprocessing-and-Video-Shared.md`
- **タイトル**: `コート幾何後処理・#2190失敗分析・#2191動画共有`
- **正本確認対象**: `src/tennis_scene/pipeline/video_court/README.ja.md`, `src/tasks/court_detection/geometry/video_shared/README.ja.md`, `src/tasks/court_detection/geometry/video_drift/README.ja.md`, `src/tasks/court_detection/README.md`, `src/tennis_scene/configs/pipeline.yaml`, `knowledge/nodes/court_detection/000028`〜`000033`, `000045`〜`000047`, `000068`〜`000078`
- **必須セクションと内容**:
  1. **単フレーム幾何の進化：TCD argmax → Hybrid KP+LINE PROSAC+LM と「frame 0 固定」**:
     - TCD 固定12点・argmax 廃止 → 確信度順 PROSAC（`#1207` `X_REFLECTED` 向き検査追加、`000045`〜`000047`）＋最大8点ハードトリミング＋双方向2値 LINE 共同 LM 最適化（指定4枚 `0/4 → 4/4`、`BiOnysEJdYE_2200` `16.40 → 1.73 px`）。
     - 画像ベース固定グリッド領域探索の役割と、CPU 後処理 `~2s/frame`（SciPy `least_squares` 17回呼出、3,030 frame で約116分、`000033`）から `2026-09-25` **固定カメラ frame 0 のみ推論**決定への経緯。
  2. **`#2190` 単フレーム採否・RGB白線精密化の失敗解剖と `legacy` 完全復元（`000068`〜`000075`）**:
     - `meiji_4cam` 276枚で `legacy` が **124/276 棄却**（inlier<8: 79, 画面外のみ: 15, 両方: 30）した原因（14点全画面内制約）と、旧手指定参照の `9–48 px` 誤差発見 → 15フレーム×4カメラ承認済み白線参照 **`court_paint_reference_v2`**（許容 mean<=`10` / max<=`25 px`）策定。
     - 4系統比較表（`legacy` [105 FP / 47 TP] → 初回 `quality_v1` [116 FP + Court val **114/2211 破綻**、支持点集合 `S` が `667/684` で振動] → 固定支持+`paired_edges_v1` RGB [val回復・FP 0 だが **TP 5/221 = 2.3%**, `cam1–3` 全滅] → `candidate_subsets`+`semantic_consensus_v2` [TP `59/181 = 32.6%` だが **34 FP**: `cam1` ネット上端誤認33件 + `cam0 clip_023` 前後反転 `831 px`, `cam3 0/69`]）。
     - Revision 4 合意（`000075`, PR `#2200`）: 単フレーム解決を断念し **`clip` 既定を `legacy` へ完全復元**（3,545画像 / 8,530領域で不一致0）。明示 opt-in 部品として残したもの vs 削除したもの（`adaptive`, `paired_edges_v1`）。
  3. **`#2191` 動画共有コート（`video_shared`）と 4-DoF 相似線 offset drift（`000076`〜`000078`）**:
     - `clip` vs `video_shared` の判定・状態遷移 Mermaid 図（`stateDiagram-v2`）。
     - 共通 H の設計原則（単フレーム採否で棄却された有限 H も初期種として保持、`endpoints` + `centre_service` でネット誤認排除、H 行列の係数平均を禁止し `3 px` 等価域内最小残差を選択）。
     - Run 77（`meiji_4cam` 4/4 カメラ `0.38–3.36 px`, **FP 0** 達成 vs 局所全 H 再fit drift が `169/276` 観測不能で 0/69 全滅）→ Run 78 4-DoF 相似線 offset drift（`min_lines=4` 等、dev `56/56`, syn `24/24` 合格；Meiji 厳格 `fail` 時 `155/276` 停止の事後分析で 384/437 が `0.09–4.66 px` 微小雑音・実移動 `>10/25 px` **0件**を確認 → `2026-11-17` 人間判断で **`unobservable_policy: warn` 既定・69/69 (276/276) 通過**、`#1752` 接続）。
  4. **運用設定・Web UI・歴代 Checkpoint 早見表**:
     - `clip` vs `video_shared` の使い分け、Court Alignment Studio (`8765`)・Court Review UI (`8905`[local path]`8906`)・Dataset/Inference UI (`8774`[local path]`8775`)、歴代 checkpoint の役割整理（生64桁SHAは書かない）。

### (3) `tmp/wiki_pages/Synthetic-3DGS-Alignment-and-Generation.md`
- **タイトル**: `3DGS アライメント・合成視点生成・外観・SfM誤差`
- **正本確認対象**: `src/synthetic_data_generation/README.md`, `src/synthetic_data_generation/dataset/court/README.md`, `src/synthetic_data_generation/alignment/`, `src/synthetic_data_generation/appearance/`, `knowledge/reports/20260829-b01-b03-alignment.md`, `knowledge/nodes/synthetic_data_generation/000001`〜`000023`, `knowledge/nodes/court_detection/000007`〜`000024`
- **必須セクションと内容**:
  1. **B00〜B03 多コート 3DGS アライメント（`#618`, `#878`, PR `#836`）**:
     - KP14 単体の挫折（B00 `0/363`）→ Line モデル投影（距離重み付き RANSAC 地面投影、現行 `production.yaml` は `b863df1f01f0` に統一）→ NHT 描画 RGB 入力化（`#878`, `syn:000011`: 48視点中央値差 `4.271/255`、3DGS 内部姿勢空間と一致）。
     - 大域 subset 再fit ドリフト（`9.08 m / 89.94°`, `000001`）と holdout 棄却→`user_override` v2 公開（`000003`–`000004`）。
     - PR `#836` の 128-frontier 重み付き探索＋**`boundary_lattice_assisted`**（2面の厳密格子から B01 第3コート救済）と B00〜B03 アライメント成績表（面数・scale・被覆率・RMS `0.111–0.186 m`）。
  2. **SfM 外周制約（`sfm_bounded`）と Canonical Dataset v1 → v2 → v3（8,433枚）**:
     - 無制約軌道の外挿破綻（B01 `3,769/4,800` 逸脱）→ 4種 XY 包絡、HFOV 拡大、1m セル空間均一選択（占有セル `747 → 1,124` vs 形状のみ `632`）、B02 非対称凸包バグ修正（`sfm_complex_center_on_hull=true`）。
     - `v1`[local path]`v2`（Y軸反転中心）から **`canonical v3`**（`CAMERA_VIEW_HALF_TURN_INDEX` による厳密 2D 半回転 `Rz(π)` 左右正規化、`scope: target_court`）への進化と 4シーン枚数表（B00 `2,176` / B01 `2,052` / B02 `2,112` / B03 `2,093` = **8,433枚**）。途中遭遇した GPU 混在・実行中コード更新の教訓も簡潔に触れる。
  3. **SfM 地面高さドリフト検証（`syn:000022`〜`000023`）**:
     - 登録フレーム前後半の共通 0.5m セル地面高さ中央値差表（B00 `+8.8 mm`, B01 `+0.9 mm`, B02 `-13.8 mm`, B03 `+15.5 mm`、初期4分割 `+3.19 cm`）＝合成 GT 自体の **1〜3 cm 級幾何不確実性**。
  4. **外観バリエーション実験（B00 クレー変換、`syn:000012`〜`000021`）**:
     - Flare vs Sunburst、`50@7k` / `50@30k` / `100@30k` 比較表（`100視点[local path]` で PSNR `24.25 dB`, SSIM `0.6588`, LPIPS `0.1809`）と、視点間不整合による近距離ネットぼやけ・白線細りの限界。

### (4) `tmp/wiki_pages/Synthetic-Scaling-Diversity-and-Pair-Studies.md`
- **タイトル**: `合成データスケーリング・多様性・全6ペア比較`
- **正本確認対象**: `knowledge/reports/court-scale-trajectory-design.md`, `src/tasks/court_detection/scaling/README.md`, `knowledge/nodes/synthetic_data_generation/000024`〜`000029`, `knowledge/nodes/court_detection/000048`〜`000067`
- **必須セクションと内容**:
  1. **スケーリング軌道設計と D1〜D6 合意（`#1350`, `#1354`, 2026-10-22）**:
     - 旧 `0.5x` 刻みのネスト崩壊（32→56 で共通8点のみ）・連番リークの発見 → 2倍入れ子弧長＋決定論ハッシュ ID。
     - 5方式＋2,080視点プローブの知見（`roi_hole=0` でもボケを通す、明線指標は暗線を見落とす）と **`stratified_counter_production_v1`**（HFOV `[75, 92.5, 110]°`, `target_kp>=4 & roi_hole<=0.01`, jitter `1.0m`, `640×360`）採択。
     - D1〜D6 契約表（合成は学習専用、実画像は動画非重複評価専用 **`real-video-v1`**［val 249動画 4,468枚 / test 249動画 4,372枚］、補充なし、Pose 主評価除外）。
  2. **Colab L4 数値一致検証と Drive 耐久分割生成（`#1351`, `#1355`, `syn:000026`〜`000027`）**:
     - RTX 5060 Ti vs Colab L4 で教師・可視性不一致 **0/784 KP**、PSNR 全シーン `>=57 dB`・全視点 `>=55 dB`、`court_scene_assets_v1` 確定。
     - 64候補 cache-shard 再開検証で描画 `137 s` に対し Drive path lookup 起因の再開 `8,622 s`・fetch `3,663 s` を要したインフラ教訓。
  3. **シーン多様性 6,000 枚実験の非単調悪化と A1/A2 大規模実験中止（`#1343`, 2026-10-30）**:
     - 5,500枚合成専用学習（ViT-S, 33,160 steps, seed 42）の固定 real val (2,211) 比較表：**1-scene (`B00`) `53.95 px` → 2-scene (`+B01`) `83.45 px`（悪化）→ 3-scene (`+B02`) `43.70 px` → 4-scene (`+B03`) `23.61 px`**（中央値 `5.81 → 67.92 → 3.68 → 3.42 px`）→ 2026-10-30 に A1/A2 中止。
  4. **同群 vs 異群 全 6 ペア比較実験（`#1581`〜`#1913`, `court:000056`〜`000067`）**:
     - オムニ群（B00/B01）×クレー群（B02/B03）の全6ペア＋単一[local path] **`57.36 px`**（P01 `67.92`, P23 `46.81`）vs 異群ペア平均 **`4.57 px`**（P03 `2.97` 最良, P02 `3.49`, P13 `4.02`, P12 `7.78`）、生差 `+52.80 px`・加法性対比 `-3.24 px`。
     - 留保条件（seed 42 単発、会場・照明・密度との交絡、P03/P13/P23 の途中再開を含む暫定観察）を明確に記載。

### (5) `tmp/wiki_pages/NHT-Rendering-and-Foreground-Compositing.md`
- **タイトル**: `NHT レンダリング契約と動的前景合成の限界`
- **正本確認対象**: `src/synthetic_data_generation/rendering/nht/README.ja.md`, `src/synthetic_data_generation/README.md`, `knowledge/nodes/synthetic_data_generation/000030`〜`000031`
- **必須セクションと内容**:
  1. **NHT 出力契約と「`RGB + alpha + expected_depth` だけでは前景合成できない」証明（`#1728`, `syn:000030`）**:
     - `expected_depth`（視線透過重み付き期待深度）と通常の不透明 z-buffer の違い。
     - 非線形シェーダが zero 特徴でも RGB `0.23–0.70` を出力（線形近似と最大 `0.190` 差）し、半透明ネット領域では全 alpha `0.9999` に対し 6m/10m prefix alpha が `0.050 / 0.402` と大きく乖離する構造的理由（Mermaid 図で視線積分とネット遮蔽を図解）。
  2. **3種 Prefix 近似の比較と `nht_prefix_alpha_closed_srgb_v1` 採択・既知故障モード（`syn:000031`）**:
     - 人間判断「NHT 再学習なしで近似する」に基づく 3 手法比較表と **`nht_prefix_alpha_closed_srgb_v1`（full-alpha closure）** の優位性（両極限誤差 `0.0`、proxy 再スケール差最大 `0.00061`、SSAA2 細ボールで 4–13 サンプル維持）。
     - 既知の破綻条件：大きな Gaussian を奥行 `2 mm` 跨ぐだけで alpha が `0.797`・RGB が `0.617` 跳躍し、ray-peak と平均 z が `388/576` で不一致になる限界。
  3. **主系合成データ生成からの分離状況**:
     - `src/synthetic_data_generation/README.md` の現行スコープ境界（BLCS/PLCS/動的前景合成は主パイプラインから撤去済みであり、C5 本実装は別系統のまま未接続であること）。

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
