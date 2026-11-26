現行のStrongSORT++＋pose/CLIPを、#2442が公開した69 clip・1,205 frameで採点しました。全標本のraw/group IDF1は **0.8946／0.8966**、pair F1は **0.7503**、3D coverageは **3,774/4,802（78.6%）**、停止は **14/69 clip** でした。pair F1は疑義全体で **0.7265**、確信層の無作為標本で **0.8957** です。層別の人手誤り率は集計待ちです。 [層別比較表](https://github.com/Motoki0705/tennis-lab/blob/2c45445a42f1c0f131b94690ba8d3f6f0d8223da/knowledge/runs/run-i2443-meiji-4cam-tracker-comparison/full-saved-default.csv)

キューの4件は、共有GPUロックのACLで実行ユーザーの探索権限が無効になっていたため、起動前に失敗しました。再投入には、管理者による `[local path]` の探索権限の復旧と、CUDA CIと共有する4つのロックファイルの読み書き権限の確認が必要です。

採点CSVの再実行一致と69件のラベルハッシュ照合を確認し、結果と失敗記録を更新しました。HEADは `2c45445a42f1c0f131b94690ba8d3f6f0d8223da` です。 [ドラフトPR #2451](https://github.com/Motoki0705/tennis-lab/pull/2451)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}