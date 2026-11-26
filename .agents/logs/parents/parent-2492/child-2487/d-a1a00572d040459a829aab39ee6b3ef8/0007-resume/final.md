精度ジョブ6件のログと実ファイルを照合しました。保存済みの ECDet PT16 を含め、12条件すべてで6,844画像・26,276選手BBoxの採点結果を確認しました。実行不能は0件で、TensorRT engine は6重み分そろっています。[条件別の指標・生予測・SHA-256](https://github.com/Motoki0705/tennis-lab/blob/82ddda3c2bdd933cc6863931e546f21952764cfa/knowledge/runs/run-meiji-player-model-comparison/edgecrafter-accuracy.json)

| モデル | PT16 Recall@0.50 / 0.75 | TRT16 Recall@0.50 / 0.75 | 欠測込み PCK@0.20（PT16 / TRT16） |
|---|---:|---:|---:|
| ECDet-L COCO | 97.61% / 41.60% | 64.88% / 17.85% | — |
| ECDet-X COCO | 97.58% / 39.79% | 70.29% / 21.22% | — |
| ECDet-L O365 | 98.19% / 44.88% | 63.02% / 21.53% | — |
| ECDet-X O365 | 97.97% / 46.62% | 65.88% / 23.87% | — |
| ECPose-L O365 | 52.19% / 40.31% | 50.32% / 37.52% | 38.56% / 36.53% |
| ECPose-X O365 | 40.09% / 33.44% | 9.99% / 5.09% | 33.08% / 6.91% |

ECPose-X TRT16 の精度低下の原因は未特定です。ECDet v1 の PT32 は履歴として保存しています。

#2497 の `13e508cb7` に固定し、6重みの PT16・TRT16 × batch 1・2、計24条件の速度ジョブを6件登録しました。parity は `PT32未計測` です。速度とVRAMの結果は未確認です。

`codex/parent-2492-issue-2487-edgecrafter` に `82ddda3c2bdd933cc6863931e546f21952764cfa` を push しました。pytest 83件（7.58秒）、pre-commit、knowledge 検査が通過しました。精度ジョブのラッパーは、コマンド失敗や必要ファイルの欠落を条件別に記録します。[コミット](https://github.com/Motoki0705/tennis-lab/commit/82ddda3c2bdd933cc6863931e546f21952764cfa) · [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45572012051) · [#2487 の検証記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7156421785)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204406601172825111_2026-11-25T07-56-38-827174888Z_1736457_i2487-ecpose-l-o365-fixed-speed-13e508cb7.job","0-8204406600906716001_2026-11-25T07-56-39-093283998Z_1736525_i2487-ecpose-x-o365-fixed-speed-13e508cb7.job","0-8204406600618625364_2026-11-25T07-56-39-381374635Z_1736593_i2487-ecdet-l-coco-fixed-speed-13e508cb7.job","0-8204406600340455297_2026-11-25T07-56-39-659544702Z_1736660_i2487-ecdet-x-coco-fixed-speed-13e508cb7.job","0-8204406600020428136_2026-11-25T07-56-39-979571863Z_1736732_i2487-ecdet-l-o365-fixed-speed-13e508cb7.job","0-8204406599706080216_2026-11-25T07-56-40-293919783Z_1736800_i2487-ecdet-x-o365-fixed-speed-13e508cb7.job"]}}