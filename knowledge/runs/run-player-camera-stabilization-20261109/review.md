回転補正・レンズ写像の実装とCPU検証は成功しましたが、**ColabのL4割当拒否により比較動画の生成は未着手**です。このrunのfailedは描画試行の失敗を示します。draft PRは [#1785](https://github.com/Motoki0705/tennis-lab/pull/1785)、方式と強度候補の判断は [#1776の合意記録](https://github.com/Motoki0705/tennis-lab/issues/1776#issuecomment-6730136323)を参照してください。

回転はGaussian重みのMarkley平均、σ=0/0.06/0.08/0.12秒、±ceil(3σfps)の有限窓です。欠測runを跨がず、端は窓を切って再正規化します。位置とrollを保持し、補正後の回転で顔offsetを再計算しません。σ=0は明示的な元transformのコピーです。実装と数値library版・入力digest・有効run・各frameの左右supportを派生カメラに保存します。

最新GoProの固定レンズMISSION1系を参考に、対角159度を設計値とした等距離魚眼を選びました。メーカーの非公開較正を再現したものではありません。公開仕様と欠けている情報、直接Cycles魚眼との比較は [lens-survey.md](lens-survey.md) が正本です。survey内の1280×720の例に対し、今回の短い比較はパネル640×360、共通ピンホール2560×1440で固定しています。画角と光線は同じで、2×2比較の完成解像度は1280×720です。F2.8は参考値を記録しDOFは無効、既存の顔offsetは維持します。

C5は共通のPINHOLEのK・姿勢・時刻・pixel centerで両経路を描き、camera-zとcoverageを使った採用済みの深度合成を先に行います。その合成済みlinearized RGBに同じ写像を一度だけ掛け、最後にdisplay encodingします。深度を魚眼へ補間して再合成しません。CPUではNHTのPINHOLE/NONE契約への入力と光線/遮蔽境界/linear光の写像を確認しました。**実NHT GPU描画、超広角での背景alpha/support、C5のGPUメモリ、周辺の球・ネットのaliasは未確認**です。

CPUの主な検証は以下です。

- 関連unit/integration 126件成功、17.39秒、pytest -n 2。C3更新版から43 instance増加（回転16、レンズ13、保存/描画/公開の接続14）。setupの増分は小fixtureに収め、既存の頭部生成・投影・C3のnodeを再利用しました。
- 設定inventory/discoveryと監査CLIの既存26件成功、327.33秒。既存cold importが114.68/77.91秒で大半を占めます。
- pre-commitのruff/mypy成功。knowledge内の再現scriptはhookの対象外のためruff format/checkとCLI helpを個別確認しました。measure_cameras.pyは実データで実行済み、render_comparison.pyとverify_delivery.pyは実描画では未実行です。
- [camera-metrics.json](camera-metrics.json)は実装が保存した4列の再計測です。位置・mask・理由・元時間軸の全配列不変とσ=0のtransform完全一致を検証しました。3Hz以上パワーの減少と最大姿勢偏差を併記し、平滑性を頭部姿勢の真値精度とは扱いません。
- 初版sourceの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42551328419)はPython成功。初回の既存process終了テストの/proc読み取り競合は、該当ファイルがmainと同一であることと個別2ケース成功を確認し、失敗したshard8だけ再実行しました。テストを緩める変更はしていません。

Drive root相対 `outputs/tennis_scene/visualize/player_view/stabilization-comparison/inputs` は57ファイル、161,253,416 bytesで内容照合済みです。元C1/C2のmanifest digestと、全1010frameを補正してから元frame[0,240)へ切り出した4つのpackageを保持しています。各packageにはmesh PC2も含み、元のモデルファイルをremoteへ持ち込む必要はありません。入力manifestのgit_refとrender時のrefを揃えます。

GPU起動は共通ランナーの既定Codex gpt-6.1-sol/maxへ委任しました。[allocation-failure.json](allocation-failure.json)と[gpu-operation.json](gpu-operation.json)が実際の失敗receiptです。L4要求は2026-11-09 02:51:54 UTC、CLI0.7.4の応答は `Allocation refused (precondition failed)`。原因は特定できていません。要求セッションにowned registryがなく、remote checkout/jobは未作成でした。preview0枚、contactなし、比較動画なし、描画結果のDrive保存なし。既存の他担当2 assignmentは変更していません。

再開時は同じ#1776 worker sessionとbranch/PRを続けます。資源が利用可能になってから、残った失敗名のlocal stateを既存GPUと混同せず、所有・未割当を確認して新しい明示的なattempt名で開始してください。長時間GPU起動は引き続き共通ランナーの同じGPU担当へ委任します。

1. tennis-colab/tennis-drive/training-queue skillに従い、入力のgit_refでL4を確保します。割当前からの保守的な4時間期限と、保存・解放の余裕を記録します。依存はuv lockのCPU group、Blender5.2.1 LTS/OPTIXです。CPU/別GPUへのfallbackは行いません。
2. Driveから上記inputsを内容照合付きで取得します。resource=allのqueueを介し、render_comparison.pyの`--phase preview`を実行して元frame0/26/162×4強度×2IDを描きます。各人12枚を上限として、欠測frame・広角写像・実camera/color audit・contactを保存して確認します。
3. previewの確認後だけ、`--phase full`で同じinputsから各240frameを描きます。fullというphase名は4秒比較の全frameを意味し、1010frame全区間は意味しません。最初のsigmaの動画とatomic checkpointをDriveへ保存した時点で起動workerを終了し、親は完了・異常終了・VM終了のイベントを監視します。
4. 両IDの2×2比較、8個の個別動画、4個のID左右比較とmanifest/auditをDriveへ保存します。必要な動画とsmall auditだけ回収し、verify_delivery.pyで14動画の全decode/FPS/PTS・1920描画row・元frame対応を確認します。大きな中間PNGは既定で全量回収しません。
5. 全jobの終了・Drive receiptを確認して自分のsessionを解放し、実測・成果物と失敗/未確認をこのrunに記録します。人間が補正強度を選ぶまで全1010frameの再描画は行いません。動画生成・CI成功・PR mergeを人間の受入とは扱いません。
