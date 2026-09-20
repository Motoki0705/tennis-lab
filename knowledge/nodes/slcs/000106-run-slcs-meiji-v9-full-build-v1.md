---
task: slcs
sequence: 106
recorded_at: 2026-09-19
date_source: experiment_date
papers: []
id: run-slcs-meiji-v9-full-build-v1
type: run
title: 'Meiji v9全56clipの生成プロセス完了（品質採用は別監査）'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  data: slcs/meiji_rgb_v9
  recipe: src/tennis_scene/configs/build_slcs_dataset.yaml
  checkpoint_warning_roles: [dino, vitpose]
metrics: {published_clips: 56, expected_source_clips: 57, excluded_clips: 1}
repro:
  commit: 767ac67c43c35f65dd294ddf69a0b6a25d519656
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'env TENNIS_RGB_GPU=0 bash /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb/scripts/datasets/build_real_rgb.sh
    --execute meiji '
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-v9-full-build-v1
  output_dir: outputs/tennis_scene/generate/meiji_rgb_v9/s42-002
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789775752855481335_3187724_slcs-real-rgb-build-meiji.log
parents: [run-slcs-meiji-v9-court-v1, run-slcs-meiji-v9-feature-reuse-v1, run-slcs-meiji-v9-observation-reuse-v4]
tags: [slcs, meiji, dataset, generation, rgb, queue]
---

## 考察 / Findings

### 要約

共有queueの1コマンド経路で対象56clipの教師生成とRGB特徴検証が終わり、jobはdoneへ移動した。
これは生成プロセスの正常終了であり、全件の品質採用成功ではない。別の全件監査で残存3件の記録不一致を検出した。

### アーキテクチャ詳細

Meijiの同期3カメラ、外注ボール観測、学習済みCourtとcrop再推論、時間連続性を使う人物選択、
PLCS/BLCSと多視点幾何補正、DINOv3特徴の既定v9 recipe。57clipのうち理由付き除外1件を除く。
旧セッションで開始したjobを引き継いだため、queueが記録した元session・commit・commandを保持する。

### メトリクスの解釈

logに56件のPublishedを確認し、RGB特徴検証も最後の `video_002/clip_023` まで完了。
生成中に発見した不整合6件は、独立CPU runで監査して退避付きで差し替えている。
本nodeはモデルの学習runではなく、学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

この長時間プロセスは公開前の教師pin/receipt guard追加前に起動しており、途中のコード変更を取り込まない。
生成終了だけで下流へ採用せず、全件監査を分離する必要性が確認された。
DINO/ViTPoseの明示的なdigest警告方針は続行を許すが、教師精度や重みbytesの完全認証を保証しない。

### 既存実験との比較

先行のCourt・人物観測・RGB特徴を再利用し、部分clipの診断から56clipの公開へ進んだ。
進行中の修復結果は各repair nodeへ分離し、このGPU runだけの改善効果として混ぜない。

### 次に有効な実験

全件監査で残った3clipをguard付きで修復し、欠落を許さない監査を通してからbroadcastとの統合・SLCS学習へ進む。
