FastHMR CameraHMR PT16の速度・VRAMを検証し、[実装PR #2501](https://github.com/Motoki0705/tennis-lab/pull/2501) と [記録PR #2534](https://github.com/Motoki0705/tennis-lab/pull/2534) を更新しました。

固定24系列でB=1/2/4を各300反復し、全900件の入力順と集計を照合しました。B=1の処理時間P50はFastHMRが **1,090.92 ms**、現行HMR2＋GVHMRが **492.28 ms** で、FastHMRは2.22倍の時間を要しました。VRAMの観測最大値はそれぞれ4,934.1 MiBと4,926.1 MiBでした。CPUスレッド数と内部のバッチ方式が異なるため、参考値として扱います。新規計測のparityは未判定です。[全条件の計測記録](https://github.com/Motoki0705/tennis-lab/blob/fb54c55101b8bd0d3a8253735b10e5c14776ceaa/knowledge/nodes/tennis_scene/000090-run-fasthmr-camerahmr-pt16-speed.md)

TRT16は、cuDNNのTF32設定とLightningのTrainer参照で変換が停止し、今回の経路では実行不能でした。保存済みGVHMR出力とFastHMR PT16から作成した代表6 clipの同期動画24本は、ハッシュと表示時刻を照合済みです。動画を見た品質判断と採否は人間の確認待ちです。配置以降と全パイプラインの速度は未計測です。[同期比較動画を開く]([local path])

`codex/parent-2492-issue-2491-primary` のHEAD `bba1666490373d573c6f0cf972a4bc946dcb6199` をpushしました。関連テスト45件とpre-commitが通過し、実装CIの起動を確認しました。記録PRのHEAD `fb54c55101b8bd0d3a8253735b10e5c14776ceaa` はCIを通過しました。[実装CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45609410979) · [記録CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45609412764)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2501,"head":"bba1666490373d573c6f0cf972a4bc946dcb6199","resume_on_success":false}}