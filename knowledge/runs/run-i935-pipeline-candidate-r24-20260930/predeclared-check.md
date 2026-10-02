### 進捗ログ run 24 (2026-09-30) RAM修正・B option・2jobの投入前見積

- 実施: [事前判定基準](https://github.com/Motoki0705/tennis-lab/issues/935#issuecomment-5909424600)を283515adでcommit/push済み。seed43評価は引き続き未読。RAM guardを[4823af6d](https://github.com/Motoki0705/tennis-lab/commit/4823af6d)でpush、関連15 tests成功。8GiB起動、6GiB未満が連続sampleで30秒継続したら停止、3GiB未満は即停止。10秒bucketの最低/最高MemAvailable・process tree RSSを保存する。
- 成果物: draft [#973](https://github.com/Motoki0705/tennis-lab/pull/973)、base #972。名前付き e9_anchored_s42_covariance を実装中。bundle/検出器/共分散artifactのhashを固定し、平均・weight・presence不変の補正とschema v2保存。既定は変更しない。
- GPU事前見積（**この順で2件、resource=all、追加/自動retryなし**）:
  1. seed44のみstep0から12k更新。run23と同じ入力・選択・評価、出力だけr24へ分離。seed43実測1201秒を根拠に**20–30分、peak VRAM2–4GB、出力約0.8GB（job上限1.5GB）**。外側timeout2385秒＋kill猶予15秒で40分以内。allocator6GiB・device監視7.5GBで停止。
  2. source-video execute/load。**3–8分、peak VRAM2–6GB、出力約1.5GB（job上限3GB）**。外側timeout1185秒＋kill15秒で20分以内。allocator6GiB・device監視7.5GBで停止。CPU thread2、3camera直列。Meiji val video_000/clip_010/cam0,1,2全270frame。既存の短clipを採用し結果によるclip選択はしない。
- 両jobとも同じRAM guard、最小MemAvailable・時系列・sampled peak VRAM/process-tree RSS・所要時間をreportへ保存。RSS和は共有page重複を含み、sample間のpeakは未保証。合計job出力上限4.5GB＋小さなbundle/記録でdisk予算5GB内。旧cache/partial/checkpoint削除なし。
- Acceptance checklist 状況: guard検証済み。B option通常検証中、実動画検証は未投入。seed判定/既定切替は次run。
- 次の一手: 通常検証・source/checkpoint/hash preflight・commit/push・レビュー観点/CI確認後、2jobを順に登録してWAITING_QUEUEで終了。

【要判断】元動画とcached-evidence照合の事前許容値
- 前提情報: cacheはJPEGからの検出証拠、pipelineは元mp4のdecode。媒体差により同じframeでも入力RGBはbit一致しない。過去run7のCPU/CUDA監査もlogitの一律微小許容値には失敗した。今回のanchored source-video出力はまだ生成していない。
- 選択肢: A 保存/loadは完全一致、source/cacheは全GMMを厳しい固定許容値で照合し不一致ならBを止める / B 位置誤差の大まかな分布だけで許可する。
- 採用した暫定案と理由: A。**全270frame×3camera×全4成分**について、frame/PTSは完全一致、mean/Choleskyの絶対差<=5e-4 normalized uv（横1920pxなら約0.96px）、mixture/presence logit差<=0.01、mixture weight/presence probability差<=0.001、covariance差<=2e-5 normalized uv²、rtol=0。cachedのCholeskyだけ同じ固定artifactで補正して比較。NaN/shape/軸の不一致は失敗。各fieldの最大差/超過件数を保存する。
- 保存後、cameraごとに新しいプロセスのload-onlyで再読込し、全保存配列とartifact参照をbit一致で比較。load側のmodel/assembler呼出しは明示的に禁止して検査する。媒体差でsource/cache照合が失敗しても許容値を後から緩めず、全cameraの結果を残してB未通過とする。
- 覆す場合の影響範囲: 後続Bの採否と別grantでの追加診断のみ。今回の不一致記録・旧artifactは保持し、既定切替しない。
