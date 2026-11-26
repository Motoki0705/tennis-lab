#2497を取り込み、ViTPose-HのPT16・TRT16、flipあり・なし、batch 1/4/8の既存12条件を速度基準に採用しました。元レポートと3,600反復の入力順を照合しました。追加のqueue投入は0件です。 [速度基準の記録](https://github.com/Motoki0705/tennis-lab/blob/59faa32a28d2f54e6492abdba2429846fa0397e3/knowledge/runs/run-person-model-speed-low-precision/README.md)

主比較のflipあり・batch 1では、wall P50がPT16で22.17 ms、TRT16で12.64 msでした。NVMLの使用量のサンプル最大値はそれぞれ4,388 MiB、1,752 MiBです。新規計測のparity表記は「PT32未計測」としました。

`codex/parent-2492-issue-2490-primary` の `57fdcaacf987f397e4dbfaf42a1e6ab9383046ed` をpushしました。連番の衝突を解消し、pytest 95件、pre-commit、knowledge検査が通過しました。残るPareto比較にはRTMPoseとPMPoseの速度・VRAM結果が必要です。 [ドラフトPR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"57fdcaacf987f397e4dbfaf42a1e6ab9383046ed","resume_on_success":true,"next_step":"RTMPose・PMPose担当のPT16/TRT16速度成果物を照合し、揃い次第、固定疑似GTの精度と結合してPareto比較・改善率・採用候補をまとめる。"}}