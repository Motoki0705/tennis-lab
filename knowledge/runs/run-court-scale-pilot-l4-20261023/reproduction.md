# 再検査と再実行

`run.json` が入力/版/保存先の入口。`repro.sh` はCPU上でframing境界、metricの集計、計測wrapperの引数転送、
失敗receipt、保存済み根拠の相互整合、費用表の再計算を行う。新しいGPU・Drive出力は作らない。
`.venv/bin/python` と実行元のrepo checkoutが必要。application実行時のcommitは
`dc5c253ee4d502e7d78e904c49dbd3f2dd9e7833`。report用scriptはこのbundleに保存してある。

```
bash knowledge/runs/run-court-scale-pilot-l4-20261023/repro.sh
```

原データの厳格な再取得は、court READMEの `generate_court_scale fetch` で**新しい空のroot**へ行う。
Drive pathとrequest SHAはrun.jsonを指定し、dataset filenameのidentityとファイルbyte SHAを区別する。
688ファイル/96cache/1,102acceptedを検査するが、過去の生成速度の再現とは扱わない。
既存のmain treeデータやDrive asset/smokeを上書きしない。

GPUを使う再試走は、tennis-colab/tennis-drive skillとcourt READMEを読み、L4の一意sessionと**新しいDrive dataset path**を作る。
Git refは上のsource commit、bundleはrun.jsonのadopted assets/hashを照合する。
`job-requests.json` と `colab_*.sh` は実際に提出したargv/bodyであり、失敗したPATH不足のbodyも保持している。
再実行時にはsession・出力pathを全て一貫して新しい値へ置き換え、実際のjob requestも別記録する。
歴史的bodyを同じpathで再送してcache再利用の時間を新規生成throughputと混同しない。

1. setup bodyを実行し、returncode=0とdrive_saved_atを確認する。
2. report helper 7本（framing/timed_command/review_generation/profile_training/summarize_training/evaluate_framing/collect_final）を
   新runのoutputs内scripts/へuploadする。実行時のhashはruntime-script-hashes.jsonに記録済み。
   collect_finalのbundle版は、実行後にlocal変数のtype annotationだけを加えたことも同表へ明記した。
3. generation bodyで各24軌道・密度0、1cache保存後に明示中断し、旧rootを保持して空rootへ復旧する。
   fetch・全教師roundtrip・画像/動画review・framingを実行し、全てDrive保存してから次へ進む。
4. trainingは `colab_training_resumed.sh` を参考に**32更新/単一val**を新runで実行する。
   renderer解放、batch8合成専用、pose学習、real-video-v1とexact step32を維持する。
   evaluatorは通常の完全batch順のlogitへ追加accumulatorを当て、通常指標との一致をassertする。
5. small reportと必要なPNG/TensorBoardだけ回収する。checkpoint全量やdataset全量をローカルへ取る必要はない。
   全job terminal＋Drive保存を確認し、失敗時も当該sessionをstopして差分を確認する。

ここでのbodyは本実験の承認ではない。main用の複数valへ変える前にbest-val retentionを修正/pushし、
D2/max/U/seed/長時間条件/scene補足について親が人間の判断を取る。

学習曲線は `plot_training.py` でnative TBの実在するloss/LR tagを描画した。
通常のkg_curves実行は当runで空のsingle-point指標panelを選んだため、同じcurves.pngを明示tagの図へ置き換えた。
raw eventはGitへ入れず、Driveの学習run `logs/version_0/` とcurves.jsonのSHAから取得できる。
valのreal専用lossとpose付きtrain lossは監督範囲が違うため、対応する同一目的関数の曲線として重ねていない。
