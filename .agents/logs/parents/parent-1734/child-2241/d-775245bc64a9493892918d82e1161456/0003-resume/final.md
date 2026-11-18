#2241 の再レビュー指摘3件を修正しました。`codex/court-side-video-shared` のHEADは `e71e15a14` で、PRは **Ready for review** です。採用経路ごとの照合表と検証結果をPR本文に記録しました。[PR #2244](https://github.com/Motoki0705/tennis-lab/pull/2244)

- build・loadともに、証拠に使う全clip・全cameraに現在のコート採否規則を適用します。棄却時はclip、camera、drift理由を返します。
- build再利用時にも、全clipの校正・ボール配列のchecksumを照合します。別clipの配列が破損していれば、再採点前に停止します。
- clipの明示load・自動cache・完成markerは、共有digestと方針が一致する共有componentの出力を要求します。import由来の出力と矛盾clipを拒否することを確認しました。

meiji_4camの保存済み69 clipからCPUで **`[F,F,T,T]`、margin `0.267333`、単独一致51・警告18・矛盾0** を再現しました。有効frameは13,869で、69件のclip表と552件の仮説順位は保存記録とバイト単位で一致しました。コート方針を `fail` にすると全69 clipに棄却対象cameraがあり、既存判定の再利用も `drift_unobservable` で停止しました。[実データの内訳](https://github.com/Motoki0705/tennis-lab/blob/e71e15a1435cb60dc7d23fe8f0250dce76da05cc/knowledge/runs/run-court-side-video-shared-20261118/clips.csv)

関連CPUテストは追加9件を含む **784件成功（243.18秒）**。pre-commit、通常CI全14ジョブ、Knowledge CI全2ジョブが対象HEADで成功しました。[通常CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44296872181)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44296872167)

worker側のvalidatorは指定なしで試行・完了とも0回です。次の工程は親の正式再レビューです。一括CLIと下流sceneの全69 clip確認は #1752 に残っています。