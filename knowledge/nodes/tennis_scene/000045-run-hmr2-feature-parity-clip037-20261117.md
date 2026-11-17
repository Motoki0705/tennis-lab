---
id: run-hmr2-feature-parity-clip037-20261117
type: run
task: tennis_scene
sequence: 45
recorded_at: '2026-11-17'
title: 短いダブルスclipでHMR2特徴分離前後を比較する
issue: 1754
provider: codex
status: planned
summary_impact: none
config: {}
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-hmr2-feature-parity-clip037-20261117
  output_dir: outputs/hmr2_feature_acceptance_20261117
parents: []
relations: []
papers: []
tags: []
---

親の継続指示（#1754、worker turn 76a9f971355c49ae855fdac0d08f2ecd）により、clip_035を受入入力から外して短い入力を選び直す。既存の失敗は000044へ保持する。今回はclip_037（292frame、59.94fps、4camera、1920×1080）を最初に試す。受入成功はまだ確認していない。

mainの既存clip_037計測にも、identity 1と0が同じ290frame候補となる人物対応停止があった。source frame 0を目視すると対象コートはダブルスで、近側2人・遠側2人がいる。既定の1人/側から、既に対応済みの試合形式設定 `player_association.players_per_side=2` を明示し、main単一CLI・base・分離後の全経路を同じ設定にする。人物IDの手動統合や採用閾値の変更は行わない。

mainは別の専用worktreeへorigin/main `18a2fea05` を固定し、source無変更の公開 `src.tennis_scene.scripts.run_pipeline` を呼ぶ。clipの長さは既定frame_sampling上限内なので上限overrideは不要。元datasetは読み取りだけとし、store/exportは専用outputsへ保存する。動画4本のSHAとffprobeによる実frame数・解像度はinput-manifest.jsonへ固定する。

GPUは共有training queueのall枠で1clipずつ処理する。最初の候補が身体段まで到達すれば候補探索を終え、その保存済みbody_view_selectionと幾何/2D/三角測量を固定してbase `codex/scene-inference-runtime`（ab638fb2a）と分離後を比較する。停止した候補のnode・reason・原ログは残す。順序と事前許容差はqualification-plan.jsonを正本とする。

許容差は前回から維持し、rad/betas atol=1e-5・rtol=1e-5、位置/translation/mesh(m) atol=1e-4・rtol=1e-5、再投影(px) atol=1e-3・rtol=1e-5。ID/frame/mask/拒否コードは厳密一致を要求し、非空の有効身体配置を条件にする。各身体経路のseedは42、float32、既定batch上限、allocator cap 10GiBで固定する。

この記録はC6の身体推論境界の受入であり、速度benchmarkや全69clipの受入には広げない。標準pipelineのcheckpoint・再構成方式は変えず、現在のsummaryの判断も変えないためsummary_impactはnone。独立validatorは指定/試行/完了0/0/0。

2026-11-18の再開時、前回のqualification jobは未実行のcancelledとして照合した。CUDA異常で人間がWindowsを再起動し、再開後のbounded nvidia-smiはRTX 5060 Ti / driver 621.24 / 使用747MiBを返した。身体受入の成功には数えない。再投入では、開始前・60秒ごと・終了時のNVML health probeを10秒timeout付きで行い、再び無応答/エラーになれば時刻・処理・エラー全文・Python全thread stackを保存して当該queue workloadを終了する。CUDA例外が出た場合も再試行を繰り返さず親へ返す。これはモデル進捗のポーリングではなく、再起動後に要求されたGPU異常検知である。
