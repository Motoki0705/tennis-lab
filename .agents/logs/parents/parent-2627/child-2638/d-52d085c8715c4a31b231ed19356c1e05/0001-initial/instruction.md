# 子Issue #2638 全文執筆（LaTeX 原稿・図表・参考文献・根拠対応表）の実装指示

親Issue #2627 の子Issue **#2638**（`[論文執筆 #2627] 論文の全文をLaTeXで書く`）を担当してください。

## 1. 目標と完了条件

`paper/tennis_scene_architecture/` の既存雛形・設計書（`design.md`）・`.agents/skills/paper-writing/SKILL.md` に従い、DINOv3 技術報告（`paper/reference/arXiv-2508.10104v1` を構成・図表密度の参考とする）に倣った **日本語の技術報告の全文（要旨・第1〜11章・付録・図表・参考文献・`evidence/claims.md`）** を完成させてください。

受け入れ基準（#2638）:
1. 全章（要旨、第1〜11章）と付録の本文・図表・参考文献（`references.bib`、および `main.tex` での `\bibliographystyle`・`\bibliography` 有効化）が LaTeX 原稿として揃っていること。
2. 本文・図表に掲載した数値すべてに出典・評価条件・元データ・ハッシュの対応が `paper/tennis_scene_architecture/evidence/claims.md` に記録されていること。
3. 第5章（コート検出）が既存の `paper/court_robustness/` の画像・図（`figures/`・`images/`）を使用し、コピーした図のパスと SHA-256 ハッシュが記録されていること。
4. `bash paper/tennis_scene_architecture/build.sh --check-tracked` が終了コード 0 で成功すること（未解決の参照・引用 0 件、文字欠落 0 件、`Overfull \hbox/\vbox` 0 件、未追跡 LaTeX 入力 0 件）。

> **重要（役割分担）**: 冗長な留保表現の徹底推敲と最終PDF全頁検査は後続の子Issue #2639 で行います。ただし、この #2638 段階でもできる限り明快で技術報告として自然な日本語で書き切り、実験条件は設定節や表の注にまとめて整理してください。

---

## 2. 必ず守る編集スコープ・人間判断・根拠の扱い

### 2.1 変更対象範囲の厳守（重要）
- **変更・追加するファイルはすべて `paper/tennis_scene_architecture/` 配下のみに限定してください。**
- `knowledge/` や `src/` は変更しないでください（現在 `main` の `knowledge/registry/deployments.yaml` に別親起因の CI エラーがあり、`knowledge/` を触らないことで #2638 は通常の CI でクリーンに検証・マージできます）。
- 共有 `.git/info/exclude` に `[local path]` が含まれているため、追加・変更したファイルは必ず個別パスで `git add -f -- ...` し、`paper/tennis_scene_architecture/build/` の中間生成物・PDF はコミットしないでください。
- `build.sh --check-tracked` は LaTeX が読み込むすべてのリポジトリ内入力が Git 追跡済みであることを要求します。したがって、外部ディレクトリ（`paper/court_robustness/`、マージ済み `knowledge/runs/`、または下記未マージ検証 PR コミット）から図を掲載する場合は、**`paper/tennis_scene_architecture/figures/` にコピー（または同ディレクトリ内の生成スクリプトで再現生成）して `git add -f` で追跡登録**し、元パス・元コミット SHA・コピー先パス・SHA-256 を `evidence/claims.md`（および必要なら `evidence/figures.md`）に記録してください。

### 2.2 人間の編集判断（#2628 合意事項 — 厳守）
1. **対象範囲（#2628#issuecomment-7190240176）**:
   - 標準 `tennis_scene` 3D再構成パイプライン（手法＋実行システム）を対象とする。PLCS / BLCS / SLCS / `ball_refiner_3d` は本論文の対象外（本文に混ぜない。関連・今後の展望で簡潔に触れる場合のみ対象外であることを明記）。
   - 投稿用フォーマット対応や外部公開作業は対象外。
2. **第5章・コート検出（#2628#issuecomment-7205422706）**:
   - 堅牢性の主根拠は既存のシーン数・シーン組合せの多様性実験（`000055`: 1→4シーンで旧TCD実画像val平均KP誤差 53.9→23.6 px、`000067`）に置く。
   - **#2633 の小規模視点分布比較**（PR #2671、下記コミット `d292aa707ab7874119698ad6191e7d7704e0f1e6`、`000083`[local path]`000084`）の数値は改変せずそのまま報告し、限定条件（単一シーン由来・小規模等）での傾向と課題を第10章の今後の課題へ接続する。
   - 既存の `paper/court_robustness/figures/`・`images/`（地面推定 `method_ransac.png`、LINE投影 `method_projection.png`、B00–B03 アライメント、実画像 `local01`–`local04` 比較、SfM ドリフト `sfm_temporal_drift.png` 等）を厳選して掲載し、元ファイルとコピー先の SHA-256 を記録する。
3. **第6章・プレイヤーの3D空間上での追跡（#2628#issuecomment-7193775785）**:
   以下の **4段階構成** で記述すること：
   - **① 2D追跡アブレーションとその限界**: `000012`（StrongSORT++ + pose raw IDF1 0.930→0.955、Deep OC-SORT+pose/CLIP+AFLink/GSI との対比と被覆率）・`000013`（重複 box 統合）。
   - **② コート平面による選手選別の精度向上**: `#2634`（`000015-run-i2634-player-selection-ablation.md`）— 予約未見 3cam で精度 63.25%→100%、IDF1 74.98%→99.71%、4cam で完了クリップ数 0→55/69。
   - **③ 停止クリップの原因は2D側処理全体による情報削減**: `#2636`（`000101-run-meiji-stopped-clip-observations.md`）— DINO・追跡・ReID・2D選手決定が段階的に情報を削ぎ落とす構造を実測で示す（DINO除去 0/8 回復、グループ統合 0/2、次点除外 7/9 通過）。`knowledge/runs/run-meiji-stopped-clip-observations/figures/` の4視点診断図も掲載する。
   - **④ 未マージの3D時間追跡（#1739）による停止解消とトレードオフ**: `#1739` が `meiji_4cam` の停止を 52→69/69 へ解消する一方、Meiji 3cam 予約未見でリグレッション（IDF1 92.07%→80.42%、留保視点再投影中央値 7.89→24.86 px）が生じることを **未マージ枝であることを明記して** 報告する。なお、第4章および第9章の標準パイプライン記述は現行 `main` 経路を維持する。
4. **第7章・ボール検出（#2637 ＋ 新規統制学習 #2697）**:
   - **外部注釈の来歴**: `#2637`（`000097` / `knowledge/runs/run-i2637-ball-image-difficulty/provenance.json`）で確定した来歴をそのまま正確に記述すること（未確認の推測を書かない）。
   - **実映像の難しさの定量化**: `#2637`（`000097`）のカメラ別球像径・ブラー・可視性（`footprint-distributions.png`, `visibility.png`）、容量・長系列比較（`000088`, `000068`, `000043`, `000060`）、境界外診断（4,136/184,912）を評価単位・母数つきで整理する。
   - **時系列入力の必要性と形状ベース手法の限界**:
     - **推論時の時間文脈介入（#2637 / `000098`）**: 144アンカー（observed 88）で X3D T32 通常入力 76/88（86.36%）→ 同一フレーム反復 2/88（2.27%）、i935 T8 通常 73/88（82.95%）→ 反復 0/88、シャッフル劣化、形状ベース対照 7/88（7.95%）。
     - **新規・同一学習条件での単フレーム vs 時系列スクラッチ学習統制比較（#2697 / PR #2719 / コミット `3c0083802aa118d46cd739109d3ba250004c182f` / `000099-run-i2697-ball-temporal-training.md`）**:
       - 同一 X3D/CSN+DPT・512×288・seed 42・30,478学習パック・4,700更新×128枚＝601,600入力フレームで学習。
       - 全validation 117,358件（observed 85,473）の 20 px 以内率: **T=1 RGB 1.66%（平均/中央値 226.75/177.02 px）→ T=8 RGB+MDD 40.90%（134.06/42.48 px）→ T=32 RGB+MDD 44.23%（125.74/31.68 px）**。最適バランス閾値 F1: 74.07% → 82.02% → 82.63%。全3収録元（Chat 1.29→45.54→49.20%、Meiji 3cam 2.66→27.36→29.77%、TrackNet game9 1.37→65.18→69.10%）で一貫改善。
       - `#2637` と同じ 144アンカー（observed 88）の 20 px 以内率: **2/88（2.27%）→ 40/88（45.45%）→ 45/88（51.14%）**。
       - 論文図 `temporal_training_ablation.pdf` を掲載し、T=1との差にはMDDの有無が含まれる点（時間長単独の寄与は T=8 vs T=32 の +3.33 pt / アンカー +5.68 pt）、単一seed・512×288・独立test未評価という範囲を誠実に記述する。
5. **第8章・実行システム & 第9章・実験（既存E2E ＋ 新規実行系4条件統制アブレーション #2696）**:
   - **第8章**: 宣言型パイプライン・component store（69件のCPU事前計画 2,622ノード `000072`）・モデル共有（`000064`）・横断バッチ・TensorRT FP16（`000077`）の設計を図表付きで詳述。
   - **第9章・E2E 3D精度**: ボール LOVO 再投影誤差（`000033`: 共通採点 768/1,462点、中央値[local path] 8.97/25.40 px）、人物有効性被覆（`000083`: 生成55件で関節 96.04%・SMPL 95.79%）、疑似参照整合（`000084`）。絶対3D真値誤差（モーションキャプチャ等）は未計測であり第10章の限界へ明記（#2628#issuecomment-7191344388）。
   - **第9章・処理性能**:
     - 全件69クリップ一括実行（`000083`: 55/69生成、wall time 12時間10分58秒、Peak VRAM 1.76 GiB、Peak RSS 6.16 GiB、停止14件の内訳）。
     - **新規・実行系4条件統制アブレーション（#2696 / PR #2698 / コミット `f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1` / `000107-run-i2696-exec-system-ablation.md`）**:
       - 同一HEAD `2dbd8c355`・同一3クリップ（`clip_000/037/027`、計883フレーム×4カメラ）・固定コート・空store・RTX 5060 Ti で最適化を段階的に切り離して実測：
         - `C1_serial_pt16`: **2,186.31 s**（1.00×）、VRAM 4.49 GiB、RSS 7.61 GiB、モデル読込 18/18回
         - `C2_shared_pt16`: **1,686.15 s**（**1.30×**、−500.17 s）、VRAM 4.49 GiB、RSS 7.76 GiB、6/6回、3D関節差 **0.000 mm（完全一致）**
         - `C3_batched_pt16`: **1,706.81 s**（1.28×、C2比 +20.66 s）、VRAM 4.49 GiB、RSS 8.16 GiB、6/6回、充填率 ≥99.77%、最大3D差 0.079 mm（3クリップ小規模の PyTorch FP16 ではバッチ化単独では速度優位にならず、高い充填率の土台が TRT 実行で効く事実を正確に論じる）
         - `C4_batched_trt16`: **1,319.31 s**（**1.66×**、C3比 −387.50 s）、VRAM **1.76 GiB（−61%）**、RSS **3.77 GiB（−50%）**、6/6回、共通有効3D関節 59,533点で中央値差 **1.81 mm / P95 9.06 mm**（最大 0.698 m）、SMPL有効フレーム 3,496→3,495
       - 論文図 `comparison.pdf` と4条件比較・数値整合表を掲載する。

---

## 3. 未マージ検証済み PR のデータ参照方法

以下の3本の記録専用 PR は親側で独立再現検証済みですが、別親の `deployments.yaml` 待ちでまだ `origin/main` に入っていません。必要な knowledge ノード・CSV・JSON・PDF/PNG はローカル git オブジェクトから `git show <SHA>:<path>` で直接参照・抽出してください（必要なら `git fetch origin` で参照可能）：

1. **#2633（PR #2671, コミット `d292aa707ab7874119698ad6191e7d7704e0f1e6`）**:
   - `knowledge/nodes/court_detection/000083-run-i2633-court-viewpoint-comparison.md`
   - `knowledge/nodes/court_detection/000084-group-i2633-court-viewpoint-diversity.md`
   - `knowledge/runs/run-i2633-court-viewpoint-comparison/{metrics.csv,comparison.json,test-effects.png}`
2. **#2696（PR #2698, コミット `f3ba146f57c07d8f1e168cb03fc6e71b9b9fdac1`）**:
   - `knowledge/nodes/tennis_scene/000107-run-i2696-exec-system-ablation.md`
   - `knowledge/runs/run-i2696-exec-system-ablation/{comparison.csv,metrics.json,parity.csv,parity.json,comparison.pdf,comparison.png}`
3. **#2697（PR #2719, コミット `3c0083802aa118d46cd739109d3ba250004c182f`）**:
   - `knowledge/nodes/ball_detection/000099-run-i2697-ball-temporal-training.md`
   - `knowledge/runs/run-i2697-ball-temporal-training/{comparison.csv,metrics.json,temporal_training_ablation.pdf,temporal_training_ablation.png}`

すでに `origin/main` にマージ済みの根拠ノード・図版・コード正本：
- `#2630` 調査コメント群（統合コメント: `https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548`）
- `src/tennis_scene/{README.md,schema.py,configs/pipeline.yaml,pipeline/README.md,evaluation/README.md}`
- `paper/court_robustness/{report.tex,README.md,evidence/dataset_audit.json,evidence/homography/results.json,tables/,figures/,images/}`
- `knowledge/nodes/court_detection/{000007,000030,000055,000067,000078}*`
- `knowledge/nodes/person_tracking/{000004,000012,000013,000015}*`, `knowledge/runs/run-i2634-player-selection-ablation/`
- `knowledge/nodes/tennis_scene/{000028,000033,000043,000064,000072,000077,000083,000084,000101}*`, `knowledge/runs/run-meiji-stopped-clip-observations/`
- `knowledge/nodes/ball_detection/{000003,000060,000068,000088,000094,000097,000098}*`, `knowledge/runs/run-i2637-ball-image-difficulty/`, `knowledge/runs/run-i2637-ball-temporal-context/`
- `knowledge/Papers/` および `paper/court_robustness/report.tex`・`src/tasks/person_tracking/strongsort_NOTICE.md`（参考文献の一次確認元）

---

## 4. 図表・LaTeX 品質要件

1. **パイプライン依存図・実行システム図（第4章・第8章）**:
   - TikZ または Python（matplotlib/graphviz 等、リポジトリ内 `.venv/bin/python` で再現可能）で明瞭な日本語ベクタ図（PDF）を作成し、生成スクリプトを `paper/tennis_scene_architecture/scripts/` に保存する。
2. **既存・新規の検証図版**:
   - 第5章: `paper/court_robustness/` の厳選図版（RANSAC地面推定、LINE投影、アライメント、実画像定性比較、SfMドリフト）＋ `#2633` の `test-effects.png`（または同CSVからの再作図）。
   - 第6章: `#2636` の4視点停止診断図（`knowledge/runs/run-meiji-stopped-clip-observations/figures/` から厳選してコピー）および `#2634` の比較表。
   - 第7章: `#2637` の球像径分布・可視性図（`footprint-distributions.png`, `visibility.png`）＋ `#2697` の `temporal_training_ablation.pdf`。
   - 第8〜9章: `#2696` の `comparison.pdf`（実行系4条件アブレーション図）＋ 69クリップ一括実行・数値整合表。
   - すべての図・表には番号・日本語キャプション・本文からの `\ref{...}` 参照を付け、単位・母数・条件を明確に記す。
3. **参考文献（`references.bib`）**:
   - リポジトリ内（`knowledge/Papers/`, `paper/court_robustness/`, `strongsort_NOTICE.md`, 実装コメント等）で一次情報を実在確認できた文献のみを登録し、捏造や未確認補完を行わない。
   - `main.tex` の `\bibliographystyle{unsrt}` と `\bibliography{references}` を有効化し、全 `\cite{...}` が解決することを確認する。
4. **組版・追跡チェック**:
   - 新規・更新ファイルすべてを `git add -f` で登録した後、`bash paper/tennis_scene_architecture/build.sh --check-tracked` を実行し、警告・Overfull・文字欠落・未解決参照・未追跡ファイルが **0件** であることを確認する。
   - ビルドした PDF を数ページ抜粋または `pdftoppm` 等で軽くレイアウト崩れがないか確認し、問題があればこの段階で直す。
   - `build/` 配下は絶対に `git add` しない。

---

## 5. 完了時の提出手順（高速並列レビュー対応）

1. すべての変更が `paper/tennis_scene_architecture/` のみであり、`git status` がクリーン、`bash paper/tennis_scene_architecture/build.sh --check-tracked` が HEAD で終了コード 0 で通ることを確認する。
2. ブランチを push し、日本語で PR 本文（`## 変更内容`、`## 受入と検証`、`Closes #2638`、`## 実行証拠` の JSON ブロックに現 HEAD の `head_sha` と `build.sh --check-tracked` の実行結果・ページ数・`claims.md` 照合件数を明記）を書いて **Ready（draft でない）PR** を作成する。
3. **PR の CI 完了は待たず**、Ready PR 作成直後に最終サマリー（PR URL、HEAD SHA、生成PDFページ数、掲載図表数、`claims.md` 行数、`build.sh --check-tracked` 結果）を報告して `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` で終了する（親が直ちにローカル検証と並列 `review-request` を起動します）。
