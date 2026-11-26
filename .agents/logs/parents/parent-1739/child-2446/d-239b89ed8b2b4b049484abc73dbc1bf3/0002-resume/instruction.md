# 親からの続き（担当親 Joseph）: 新経路の下流接続・GPU実行・69 clipの完了確認

## ユーザー判断（2026-11-24）

- #2460（旧経路をダブルス幅で再構成）は、clip_000・clip_020 の試し実行で止める。
- GPUは、この #2446 の新経路の下流接続と、新旧比較の段階2に回す。
- **meiji_4cam 69 clipで、選手（player）とボール（ball）がすべて完了していることを確かめたい。**

## 起点の変更

- 親が、このブランチに次の2つをmergeしてpushした。現在のHEADは `61c0994932fa8d2a44d6a3664ab5222d6e7e6f58`。
  - #2445 の最新HEAD `b5f97c893a26c4961e7daf93bb997db13cf9be23`（衝突なし）
  - #2459 の最新HEAD `a80c7a8c3e4bd31eb192f25a77b34f7bad60cb6a`（下記のとおり衝突を解消）
- #2445 は、断片を選手枠へ統合する処理を追加した。
  - 統合版の clip store は `[local path]` にある（`multiview_pose_tracks` v1）。下流はこの統合版を入力にする。
  - 結果の記録は #2445 worktree の `knowledge/runs/run-i2445-player-slot-integration/README.md`。
- #2459 の `a80c7a8c` は、比較ページの変更で標準sceneキャッシュの識別がずれた問題への修正。`pose_comparison_cache_compatibility.json` を追加した。
  - `src/tennis_scene/pipeline/source_identity.py` で #2445 側の `multiview_tracking_cache_compatibility.json` の行と衝突した。親は両方の行を残した（pose_comparison → multiview_tracking → multiview_pose の順）。
  - 各パッチが前提とするソースのダイジェストは、片側の変更だけを含む状態のもの。両方を含む現在の状態とは一致しないことがある。

## 1. CIの修正（最初に）

PR #2470 の `CPU tests (4)` が、HEAD `e118a608` で2件失敗した。

- 失敗したテスト（`tests/integration/tennis_scene/test_fp32_cache_compatibility.py` の2件）:
  - `test_current_fp32_cache_is_adopted_in_execute_and_load_modes`
  - `test_batched_tensorrt_fp32_compatibility_requires_the_exact_sources`
- 失敗内容は、ソースハッシュの不一致（`'c762d2e7…' == '6022ecaa…'`）。
- 上記のmergeで原因の一部が入った可能性が高い。両方の変更を合わせた状態で、標準sceneのキャッシュ識別が保たれるか確かめる。
  - 保たれない場合は、互換パッチの前提を正しく合成する。
  - 期待値を書き換えるだけで通さない。
  - 理由をPR本文に書く。

## 2. 新経路の下流接続（Issue #2446 の残り）

- 新しいrecipeで、新経路と旧経路を明示的に選べるようにする。
  - 新経路が成り立たないときは停止する。旧経路へ黙って切り替えない。
  - 成果物に、どの方式で作ったかを保存する。
- 3D時間IDから、カメラ別のcrop系列を組み立てる。2D Re-IDで同一性を解き直さない。
- 人物ID・観測対応・品質maskを、`body_view_selection`・`gvhmr`・`body_placement`・`scene_assembly` へ渡す。
- **ボール3Dは、#2458 で `i2363-all-v1/inputs/dataset` の現在の採用sceneに入った `ball_triangulation` を、そのまま引き継ぐ。** 新しく作り直さない。引き継いだ参照とSHA-256を記録する。
- カメラ幾何も、同じ採用sceneのものを使う。
- 既定recipeの実行契約と結果が変わらないことを、回帰確認する。

## 3. GPU実行（69 clip全件）

- GVHMR・HMR2特徴などのGPUの段は、training queue（`TRAINING_QUEUE_DIR=[local path]`）へ `add` だけで投入する。queue workerは、親が常駐サービスとして起動済み（`worker: RUNNING`）なので、自分では起動しない。
- 投入の順序:
  - いま先行している #2444 の holdout ViTPose と、#2460 のパイロットが終わってから順に走る。
  - まず clip_000（成功clip）と clip_020（旧経路で停止したclip）で、パイロットを行う。
  - 結果を読み戻して確かめてから、全69 clipを投入する。
- 出力は、新経路のsceneとして**別の出力ルート**に公開する。
  - `i2363-all-v1/inputs/dataset` の `scene.json`・`annotation.json` の参照は、まだ切り替えない。切り替えは、ユーザーが結果を見てから親が判断する。
  - #2458 と同じ方式（原子的な切替、切替前の参照の記録、restore）で、1コマンドで切り替えられるようにしておく。
- 大きなJSON・npzは、必要な部分だけをPythonで読む（本日、`jq` でWSLがメモリ枯渇した）。
- 並列数は、他作業と共有するメモリ約40 GBを考えて抑える。

## 4. 69 clipの完了確認（ユーザーの要求、最重要）

全69 clipについて、clipごとの完了表（CSV）を作り、knowledge の run 記録に置く。列は次のとおり。

- 選手:
  - 選別されたID数（−Y側・+Y側）
  - 各選手のGVHMR身体があるフレーム数・身体配置が有効なフレーム数
  - 4人そろって身体があるフレーム数と、その割合（分母は元フレーム）
- ボール: 有効な3Dのフレーム数と割合
- 停止または不足の理由（あれば）

「完了」の判定基準は、データの統計を見て決め、根拠を書く。最低でも次の2条件を含める。
- 各側で、選手の人数（`players_per_side`）を満たす。
- 選手とボールの3Dが、両方sceneに入っている。

基準を満たさないclipは、黙って落とさない。clip名と理由を列挙する。

- 既知の事情: #2458 の時点で、clip_064 だけはボールの有効フレームが 982/1,252 だった。
- 旧経路（現在の採用scene）についても、同じ表を作る。両者を並べて比べられるようにする。

## 5. 新旧比較の段階2

- meiji_4cam: 新旧を次の項目で比べる。
  - coverage
  - 停止clip数
  - 未使用視点への再投影
  - 処理時間
  - メモリ
- Meiji 3cam: ラベルで採点する。
- 比較ページ（`outputs/pose-comparison/`）の段階2を、新経路の身体（GVHMR）まで含めた表示に更新できるなら更新する。
  - 更新したら、出力ディレクトリを報告する。
  - 公開は親が行う。

## 終わり方

- テストを設計してから実装する。pre-commit・pytestを通す。
- PR #2470 を更新し、本文を日本語で書き直す。
  - 本文に、積んだ境界と依存PR（#2469・#2461）、mergeした #2445 HEAD を明記する。
- 各GPU待ちの区切りでは、その時点の状態と完了表の途中結果を報告してよい。
- 最後に、PR番号と完全なHEADを宣言する。
