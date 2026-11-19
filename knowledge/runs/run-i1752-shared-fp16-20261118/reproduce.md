# 共有court_sideとfp16の3clip受入

対象は meiji_4cam / video_000 の clip_000・clip_037・clip_027（356・292・235、合計883 source frame/camera、4camera）。全69clipは実行しない。設定の正本は [settings.json](settings.json)、測定コードは [acceptance.py](acceptance.py)。X3D/CSN＋DPT T32、ダブルス各side 2、共有courtとcourt_sideをload、DINOとViTPoseをfp16とする。両モデルの既定はfp32のまま。

測定時のcommitはqueueの `run.json` と terminalのcommitで照合する。統合した依存はC3 e311c078a、共有side 53cb2bd1e、precision 38b83a8b4。後のレビュー修正はこの測定に含まれない。

## 再現手順

1. native拡張は既存の [native inventory](../run-i1752-doubles-20261118/native-imports.json) と [再build手順](../run-i1752-doubles-20261118/rebuild-native.sh)で準備する。測定では同じdino-extension-a2の2つのbinaryを再利用し、投入前にSHA-256とsizeを照合した。binaryやprediction arrayはknowledgeへコピーしない。
2. 新しいOUTPUT配下のディレクトリを選び、`CUDA_VISIBLE_DEVICES='' .venv/bin/python -B knowledge/runs/run-i1752-shared-fp16-20261118/acceptance.py prepare --root <新しいrun出力先>` で対象mediaとmanifestを固定する。`input-manifest.json` には各ファイルのsizeとhash、登録clip、frame数を残す。
3. 共有training queueのall枠で同scriptの `run --root <同じ出力先> --expect-commit <完全commit>` を実行する。正確な投入コマンドと環境変数は保存した `run.json` / `repro.sh` にある。queue外でのrunは禁止し、同じ出力先でのrun再試行は拒否する。
4. `CUDA_VISIBLE_DEVICES='' .venv/bin/python -B knowledge/runs/run-i1752-shared-fp16-20261118/analysis.py --root <同じ出力先>` でclipごとの公開・identity・mask・段別時間・条件付き69clip投影を集計する。C1のraw recorderとsceneはOUTPUTに保持し、knowledgeには小さい集計と原データ参照を保存する。
5. GPU終了後に `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B knowledge/runs/run-i1752-shared-fp16-20261118/profiler_calibration.py` を実行する。同一のload済みcourt/sideのverifyをAB/BA順で測るCPU比較であり、全段のGPU時間の補正倍率には使わない。

全69clip用の [queueコマンド](full69-command.txt) は保存だけで、登録・実行はしていない。親の指示後、source・設定・共有artifactのpinと監視を改めて確認する。

## 計測の範囲

前回の `sys.setprofile(Observer.profile)` は今回使わない。C1の既存spanにheartbeatを付け、runtimeのmodel_eventでload/unloadを記録する。forward、batch、node schedulerは置換しない。C1のCUDA同期・資源サンプリング・heartbeat・初回compileの費用は今回のwallに残る。GPU計測中にCPU profiler calibrationは実行しない。

外部guardはNVMLの応答とmain threadからのheartbeatを監視する。異常では全threadのstackとGPU状態を保存し、停止する。再試行はしない。workerのSystemExitもstdoutをflushして例外全文を出し、正常な以前の `device: cuda` logによるguard誤検知を避ける。

69clipの投影は、成功したnodeのC3管理7resourceのload/unloadを固定費として残し、共有court/calibration/共有court_sideをclip数（camera別nodeはclip-camera数）、他の段をsource-frame数で換算する。export/publicationはclip数比例とframe数比例の両方を、残りの準備時間とrecorder開始前のCLI時間はvideo固定とclip比例を示す。import・外部guardの差分は固定費として加える。これは3clipの条件付き投影であり、信頼区間、全clip成功の保証、fp16単独の高速化率ではない。前回とは共有side・精度・profiler・成功clip数が同時に変わる。

CPU AFLinkは既存の追跡node内で都度loadされ、C3管理7resourceとは別にC1のモデル表へ記録される。全モデルが1回loadされるという主張にはしない。

単一入口027は `single_reference.py` で同じ設定・固定mediaを別storeへ処理した。比較は `analysis.py --root <3clip出力先> --reference`。既存の全component許容を超える補助Ball evidence差があるため、CPU比較CLIはexit1を返し、最終sceneのexact一致と厳密比較の不合格を両方保存する。GPU実行のexit0とは区別する。
