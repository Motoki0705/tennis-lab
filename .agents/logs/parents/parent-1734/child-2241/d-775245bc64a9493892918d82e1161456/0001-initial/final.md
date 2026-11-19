#2241を実装し、[PR #2244](https://github.com/Motoki0705/tennis-lab/pull/2244) を **Ready for review** にしました。ブランチは `codex/court-side-video-shared`、最終HEADは `53cb2bd1e` です。既定のclip判定を維持し、`court_side.mode=video_shared` で動画共有を選べます。

**meiji_4camの全69 clipで共有判定を利用できました。** cam0を基準にcam2・cam3を180度回す `[F,F,T,T]` に確定しました。13,869有効frameでcostは0.430729、supportは0.614175、marginは0.267333で、既存閾値0.15を超えています。

| 単独判定 | clip数 | 共有モードの結果 |
|---|---:|---|
| 確定して一致 | 51 | 採用 |
| margin不足 | 18 | 警告付きで採用 |
| 確定して矛盾 | 0 | 失敗0件 |

以前停止したclip_027とclip_037も、単独margin 0.116264・0.130303を診断に残して共有判定を使えました。GPUでの不足分の補完は正常終了し、CPU集計は出力パス設定を修正して完了しました。[全69 clipの表](https://github.com/Motoki0705/tennis-lab/blob/53cb2bd1ef474a2ac59e4a2fddb73708281baec0/knowledge/runs/run-court-side-video-shared-20261118/clips.csv)・[保存済み成果物のload設定](https://github.com/Motoki0705/tennis-lab/blob/53cb2bd1ef474a2ac59e4a2fddb73708281baec0/knowledge/runs/run-court-side-video-shared-20261118/reproduce.md)

追加11件を含む関連テスト741件、pre-commit、最終HEADの[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44252472448/job/134634943835)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44252472332)が成功しました。validatorは未指定で、試行・完了ともに0回です。

#1752に残るのは一括CLIへの接続と、人物対応以降を含む全69 clipの実行です。今回確認したのはcourt sideの幾何的整合性までで、正解ラベルに対する精度は未測定です。