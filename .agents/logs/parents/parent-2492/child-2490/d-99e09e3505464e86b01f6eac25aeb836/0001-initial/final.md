Sapiens2-1B／5B の評価用アダプターと TensorRT FP16 の変換手順をコミットしました。公式ソースと重みの SHA-256 を固定し、顔と左右の手首を含む COCO17 全17点の対応を照合しました。公式の水平反転を使う各バッチ2回の forward と、元画像座標への復元を実装しました。 [再現手順と資産情報](https://github.com/Motoki0705/tennis-lab/blob/decdb7c8131e7fd61f3797de20d5ff4d6f16ecbc/knowledge/runs/run-meiji-topdown-pose-comparison/sapiens2.md)

- Branch `codex/parent-2492-issue-2490-sapiens2`
- Commit `decdb7c8131e7fd61f3797de20d5ff4d6f16ecbc`
- CPU テスト43件（追加4件、5.04秒）と pre-commit の Ruff・mypy が通過しました。#2495 の疑似GTは公開済みで、手元の manifest SHA-256 も一致しました。 [#2495 公開記録](https://github.com/Motoki0705/tennis-lab/pull/2498)

**親への依頼**　Meta の規約に基づく評価利用と、そのための形式変換について、ユーザーの同意を確認してください。両重みは認証なしで取得できる公開設定です。 [Meta Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/7e5bae88456ac418ff0e58e74106c9fe192055d4/LICENSE.md)

公式重みでの推論、TensorRT 変換、全6条件の精度・速度・VRAMは未計測です。5B の FP32 は重みだけで19.074 GiBと、実機の16,311 MiBを超えます。CUDA ロードの実試行は同意確認後に行います。速度計測には #2496 の固定入力の公開も必要です。 [#2496 進捗](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}