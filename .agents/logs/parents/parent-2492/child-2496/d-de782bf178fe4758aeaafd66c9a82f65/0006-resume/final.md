`codex/parent-2492-issue-2496-primary` の `13e508cb78c46663fbe58ad8be03bcf9c96816a6` をpushし、実行証拠を更新しました。CPUテスト85件、pre-commit、knowledge検査が通過しました。 [PR #2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

DINO、ViTPose-Hのflipあり・なし、HMR2特徴＋GVHMRのPT16・TRT16、計22条件を共有queueに登録しました。新規条件はparityを明示的に未判定とし、計時結果を保存します。初回のPT32の速度とparity、およびDINO＋ViTPose-H合計は暫定記録として保全しました。

再計測結果とCIは確認待ちです。再計測後に、初回との速度差と実行可否をknowledgeとPRに反映します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204420865584359985_2026-11-25T03-58-54-415640014Z_78461_person-speed-low-precision-detection.job","0-8204420865300709797_2026-11-25T03-58-54-699290202Z_78536_person-speed-low-precision-pose.job","0-8204420865009969488_2026-11-25T03-58-54-990030511Z_78605_person-speed-low-precision-body.job"]}}