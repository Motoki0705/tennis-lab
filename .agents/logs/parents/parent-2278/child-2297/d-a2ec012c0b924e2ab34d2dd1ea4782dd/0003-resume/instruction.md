# #2297 正式レビュー（run parent-2278-04a226b021e649abb8085b8af2d6441e, HEAD e78be9632）のmajor 4件を修正する

PR #2298 のレビューコメントに詳細がある。4件すべて採用。同じbranchにcommit/pushし、新HEADの `Python` CI成功まで確認する。上限6時間。

1. **TensorRT詳細NVTXが速度計測に入っている**: 通常の速度計測ではexecution contextの `nvtx_verbosity` を明示的に `NONE`（またはビルド時に計測用として設定）にし、どの設定で計測したかを記録する。同じplanで条件2・3a・3bの速度（とVRAM）を training queue 経由で再計測し、metrics.json・knowledge ノード・最終報告の表を更新する。旧値との差も記録する。
2. **指定精度で主要演算が実行されたことを検証していない**: 「いずれかの層に指定dtypeがあれば受理」をやめる。条件ごとに「低精度で実行されるべき演算（例: MatMul/Gemm/Conv）」と「高精度で許容する演算（例: LayerNorm・softmax・reduction・cast）」をPROTOCOL.mdに定義し、engine inspectorの実行層で、主要演算が指定精度で実行されていることを検査する（主要演算がFP32に残ったら非ゼロで停止）。反例のinspectorデータ（変換層だけ低精度・GEMMはFP32）で拒否される回帰テストを追加し、保存済みCLIP planが検査を通ることを確認する。
3. **ncu追測で既存nsysとのkernel数照合が抜ける**: `--profiles ncu --profile-from` のとき、追測元のnsys結果を照合に使い、件数不一致を非ゼロ終了にする。レビュー指摘の反例（ncu収集を1カーネルに差し替え）で失敗する回帰テストを追加する。
4. **DINOのbox割当に最低IoUがない**: PROTOCOL.md のDINO精度指標で、対応と認めるIoU閾値を 0.5 に固定する（暫定。人間がPhase 2で許容差とあわせて見直せる旨を書く）。閾値未満のペアは参照側の欠落・候補側の追加に数え、IoU分布は対応成立ペアだけで集計する。可能ならDINO用の照合関数を今回のharnessに入れ、レビュー指摘の反例（同スコアの別boxへの置換）で欠落・追加が数えられるテストを付ける。

最終報告: 各指摘への対処（commit・テスト名・再計測結果）、更新後の5条件の速度表、新HEADとCI。
