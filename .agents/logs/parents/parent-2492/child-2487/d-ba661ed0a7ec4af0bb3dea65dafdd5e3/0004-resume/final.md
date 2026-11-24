[#2499](https://github.com/Motoki0705/tennis-lab/pull/2499) に比較表と可視化を追加し、`codex/parent-2492-issue-2487-primary` の `0d5f0830b9243268724eac88071c3b684e6d7b5f` をpushしました。CI実行中です。

- [36条件の比較表](https://github.com/Motoki0705/tennis-lab/blob/0d5f0830b9243268724eac88071c3b684e6d7b5f/knowledge/runs/run-meiji-player-model-comparison/conditions.csv) に6,844画像の採点結果14条件を反映しました。固定96画像でのDINOの処理時間P50はPT32が290.0 ms/画像、PT16が167.7 ms/画像でした。同じ条件で採点と有効な速度計測の両方が揃った行は現時点で0件です。
- 代表6 clipのcam0で、疑似GT・現行DINO＋ViTPose-H・RF-DETR-L・ECDet-Lを同一フレームで並べた静止画を生成しました。[clip_003の比較画像](https://github.com/Motoki0705/tennis-lab/blob/0d5f0830b9243268724eac88071c3b684e6d7b5f/outputs/player-model-comparison/representative-cam0/clip_003-cam0.png)
- ECPose-Lの既存TRT16 engineでは、同じ画像のscore 0.30以上の検出がバッチ1で2件、バッチ2で0件でした。再変換後の出力一致、ECPoseを含む同期可視化、依存PRが公開中の #1739 新経路での実映像検証が残っています。

対象テスト46件、pre-commit、knowledge検証が通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2499,"head":"0d5f0830b9243268724eac88071c3b684e6d7b5f","resume_on_success":true,"next_step":"#2496の残りの速度基準と補助担当のGPU結果を照合し、現行PT32/PT16の全コホート採点、Pareto図とECPoseを含む同期比較を進める。ECPose TRT16の再変換後の出力一致と#1739の下流依存を確認する。"}}