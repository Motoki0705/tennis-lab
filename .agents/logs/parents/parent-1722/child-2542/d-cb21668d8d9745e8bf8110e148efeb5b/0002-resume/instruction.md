# 担当: 3クラスモデルと規則による検出を multiview-v2 に適用し、目視ラベルで照合する（#2542 の統合担当 primary）

担当Issue #2542 が仕様の正本。ただし人間の判断（2026-11-26）で、旧 traj3d_transformer（重みが回収できない）の代わりに、#2567 で学習した ball_refiner_3d の3クラス（shot・bounce・net）モデルを使う。旧重みの探索範囲（ローカル[local path] outputs・ckpt、Driveの深さ5までの `*event*` 検索）と、回収できなかった事実をPRと記録に残す。

## 共通の前提（親 #1722、2026-11-26 夜間作業）

- 人間の指示: イベント検出モデルを meiji_4cam / multiview-v2 に適用した結果を今夜中に出す。モデルをネット接触にも対応させるため、データ生成経路を拡張してデータ生成・学習まで行う。規則による検出も並行して作り、モデルと比較する。代表8クリップでワーカーが目視でイベント時刻を付け、両方の精度を測る。
- 人間の判断（2026-11-26）: 旧 traj3d_transformer の重みはローカル[local path]、現行 `ball_refiner_3d` のイベントheadをショット・バウンド・ネット接触の3クラスにして代替する。PRは正式レビュー・CI・マージ条件を満たせば親がマージする。
- ネット接触 = ボールとネットの物理的な接触（白帯に触れて越える球、ネットに当たって落ちる球を含む）。通過だけのイベントは扱わない。
- ベンチマーク: `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2`（読み取り専用、manifest.json sha256 `0d11010fa2a10d73ba4b03ba01f107c90641868fbb8dd86978503cc761d4ac44`）。69クリップ、全クリップ 59.94005994005994 fps、ダブルス（片側2人）。各クリップの `scene/scene.json` が SceneResult v2。ボール3Dはclip_064（270 frame欠測）以外ほぼ全frameで有効。clip_012・015・026 はボール再注釈版。
- 予測イベントの共通JSON（モデル・規則とも同じ形で保存し、#2542 が照合する）:
  `{"schema":"ball_event_predictions.v1","clip_id":"video_000/clip_012","fps":59.94005994005994,"frames":N,"method":"refiner3|rule","method_version":"<commit or checkpoint sha256>","inputs":{"scene_sha256":"..."},"status":"ok|failed","reason":null,"events":[{"kind":"shot|bounce|net","frame":int,"time_s":float,"score":float,"features":{...},"flags":[...]}]}`。
  frameはクリップ動画の0始まりのframe番号（SceneResultのframe番号と同じ）。
- 目視ラベル（#2542）は、モデル・規則の出力を一切見ずに作る。ラベル作成者以外のワーカーは、ラベルを閾値の調整に使わない（調整用クリップとして明示的に割り当てられた場合を除く）。


## 入力（親が確定済み）

- 3クラスモデル: checkpoint `[local path]`（sha256 `b0d4399c23bf77bac115f46027cee545f99a41e9957c4e605f0eeb73ed16bde0`）。コードは未マージのPR #2571（branch `codex/parent-1722-issue-2567-primary`、worktree `[local path]`）。`RefinerPredictor.from_checkpoint()` で読み込み、`inference.scene.predict_scene_events()` に scene.json・clip_id・checkpoint_sha256・seed=42・fps_tolerance=0.06 を渡す。合成testのクラス別F1は shot .932 / bounce .986 / net .857（閾値0.5）。実行には #2567 のcommitを使い、そのcommitを記録する。
- 規則による検出: PR #2569（branch `codex/parent-1722-issue-2568-primary`、worktree `[local path]`）。69クリップの出力は実行commit `aac549c1181e8e34105a00d54fabe8f93f8ed700` で作成済み: `[local path]`（`predictions/video_000/clip_*.json`、`progress.json` sha256 `65e5edab63d1f443dc2efab82878a52a2b67a7968486ee2cee6a578eabe5e7cb`）。
- 代表8クリップの選定: あなたが前のターンで保存した `./outputs/tennis_scene/analyze/event-eval-selection/20261125T175807Z-e5e170/selection.json`（調整用 clip_001・009、採点用 clip_027・028・040・043・056・064）
- 目視ラベル（予測から独立に作成済み）: 
  - clip_001: `[local path]`
  - clip_009: `[local path]`
  - clip_027: `[local path]`
  - clip_028: `[local path]`
  - clip_040: `[local path]`
  - clip_043: `[local path]`
  - clip_056: `[local path]`
  - clip_064: `[local path]`
  - ラベル担当の観察: 同じ接触が視点間で最大約3 frameずれて見える例があった（カメラ間の同期ずれの可能性）。時刻誤差の解釈で考慮し、ずれの有無を可能な範囲で確かめて報告する。

## このターンでやること

1. 3クラスモデルを全69クリップに適用し、共通JSON（method=refiner3）で保存する。ピークの取り方（クラス別の閾値・最小間隔）は、まず合成で決めた既定値を使う。調整用2クリップのラベルだけを使って閾値・時刻のずれを確認してよいが、採点用6クリップは調整に使わない。
2. 規則による検出の69クリップ出力を使う（無ければ同じ入口で実行する）。
3. 照合: ラベルと予測を種別ごとに1対1で照合する（ハンガリアン法など、1つのラベルに1つの予測）。主たる許容は ±5 frame（約83 ms）とし、±2・±10 frameも併記する。`status=uncertain` のラベルと unlabelable 区間は主たる集計から除き、別に件数を出す。
4. 採点用6クリップで、方式（refiner3・rule）× 種別（shot・bounce・net）ごとに 適合率・再現率・F1・時刻誤差（中央値・平均絶対値）・対象数を出す。誤検出・見逃しの内訳（ポイント外、遮蔽、低いボレー、ネット付近、ボール3Dの誤差など）を代表例のframeと合わせて分類する。調整用2クリップの結果は別表にする。
5. 69クリップ全体の処理状況（成功・失敗理由）と、ラベルのないクリップでの両方式の一致率・件数分布を出す。
6. 一度限りの実験進行コードは常設のsrcに置かず、再現scriptと記録を knowledge に置く（knowledge-control skill）。再利用できる部品（SceneResultから入力を作る等）が必要ならsrcに置き、テストを付ける。ラベル・選定・予測JSONは記録に同梱するか、sha256付きで参照する。
7. PRを作り（日本語本文、`Closes #2542`）、CIを宣言して終了する。

## 報告

- 方式×種別の主要指標の表、誤りの主な原因、ネット接触の検出の可否、旧重みの探索結果、出力・記録の場所。


## 時間

人間は今朝この結果を待っている。照合と集計を最優先し、まず主要指標の表を最終報告に出す。knowledge記録PRとCIはその後でよい（CI待ちを宣言して終了する）。
