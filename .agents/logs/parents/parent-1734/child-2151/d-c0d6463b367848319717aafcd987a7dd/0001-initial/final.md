[Draft PR #2155](https://github.com/Motoki0705/tennis-lab/pull/2155) を作成しました。branchは `codex/scene-x3d-dpt-ball`、HEADは `ab8df9b80`。**#2151は受入条件が未達です。**

#1897が未mergeのため、学習commitから最小encoderを移植しました。strict復元・scene接続・明示profileを実装し、既定i935は維持しています。

- **履歴一致：未達。** 3clip・104frameのBF16比較で5frameが超過。最大座標差16.38135px（許容1px）、score差最大0.00341302（許容0.005）。高scoreのobserved frameも含み、学習時frameworkに揃えても解消していません。
- **存在閾値：0.5380124449729919。** validation由来のFPR **16.7351%**、FNR **17.0990%**。設定へ固定済みで、独立testの値ではありません。

meijiの全4camera通常入口はcourt推定で停止し、3DはN/Aでした。別途実行した2D部品の結果です。

| clip / モデル | 有効検出frame：cam0,1,2,3 | cam0処理秒 | peak allocated GiB |
|---|---|---:|---:|
| 000 i935 | 127,233,209,141 | 5.154 | 1.367 |
| 000 X3D | 194,141,184,153 | 3.720 | 1.292 |
| 018 i935 | 127,251,202,133 | 5.125 | 1.367 |
| 018 X3D | 280,201,150,171 | 3.919 | 1.292 |

時間はload・I/O等込み、compile cache作成後の測定です。

cam1〜3のclip_000ではX3Dの通常入口が完走し、**123/356 valid・平均再投影5.317px**でした。同じ幾何へ固定した別診断では、i935が**113/356・5.041px**、X3Dが**123/356・5.317px**です。i935の通常入口の成功を示す結果ではありません。

動画は `outputs/tennis_scene/evaluate/x3d-dpt-20261116/` 配下です。

- `meiji-four-camera-final/`：overview2本＋元解像度のcamera別8本
- `meiji-fixed-geometry/clip_000-i935-vs-x3d.mp4`：条件付き3D比較

全11本の全frameデコード検査済みです。[受入記録](https://github.com/Motoki0705/tennis-lab/blob/ab8df9b809813e42e70e0be1807963f724abf936/knowledge/nodes/tennis_scene/000042-run-scene-x3d-dpt-20261116.md)に指標・再現手順・全不一致一覧を保存しました。

関連pytest、変更ファイルのpre-commit、**最終HEADのCI Python・Knowledge・Web UIは成功**。treeはclean、GPU jobは全終了です。独立validatorは指定なし、試行／完了0／0回。

履歴一致差の扱い、4cameraのcourt停止、人間による動画確認を未解決として親へ返します。