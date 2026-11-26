# 親からの続き（担当親 Joseph）: 試し実行で止める

ユーザー判断（2026-11-24）: #2460 は clip_000・clip_020 の試し実行で止める。GPUは #2446（新経路の下流接続と新旧比較）に回す。

GPUパイロット `meiji-doubles-players-pilot` のqueue jobは終端に達した（状態はqueueの `list` とログで確かめる）。

## 起点の変更

- 親が、#2458 の最新HEAD `c839934f41311027557d1b4e3ac69c6974478002` をこのブランチにmergeし、pushした。
  - `c839934f` は、外部ボール取り込みCLIの入口でパスを検証する修正。
  - 衝突したのは `tests/integration/tennis_scene/test_external_ball_update.py` だけ。両側が別々に追加したテストを、両方とも残した。
  - そのテストファイルと `tests/unit/utils/configuration` の401件は通過した。
  - 現在のHEADは `c06094db5b97fb4e4c2ba6ccd9f0d4116b29fd88`。

## やること

- パイロットの結果を検証する。
  - 対象は clip_000・clip_020。
  - GVHMR・身体配置・scene_assembly の読み戻し、3D選手フレーム数、処理時間を確かめる。
  - 失敗していれば、原因を記録する。
- 次の作業は**行わない**。
  - 全69 clipの身体工程
  - 全件の投入
- `i2363-all-v1/inputs/dataset` の `scene.json`・`annotation.json` は、全69 clipで #2458 版のまま保つ。
  - パイロットで2 clipの参照を切り替えていたら、restore で #2458 版に戻し、バイト一致を確かめる。
  - 新しい scene_assembly 成果物そのものは残してよい。
- knowledge の run 記録に、次の4点を残す。
  - CPU工程69 clipの結果（作成済みの cpu-clips.csv）
  - パイロット2 clipの結果
  - ユーザー判断で止めたこと
  - 再開するときの手順（全件投入のコマンド）
- draft PR を日本語で出す。
  - baseは `main`。
  - 本文1行目は `Refs #2460`。全件は未実施なので、Closes にしない。
  - 本文に、積んだ境界commit `980a902e27f819db7def3bd969a4ff8b8ebdf850` と依存PR #2450・#2462 を明記する。
- Issue #2460 の「作業状況・停止事項」を、試し実行で止めた事実と再開条件に更新する。
- テスト・pre-commit・knowledge検査を通す。
- 最後にPR番号と完全なHEADを宣言する。
