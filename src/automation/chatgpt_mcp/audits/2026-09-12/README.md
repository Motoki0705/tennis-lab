# 2026-09-12 MCP実行環境の事実確認・修正結果

対象はMCPの隔離実行コンテナと外部ランタイム。通常WSLのOSパッケージ・共有venv・GPUドライバーは変更していない。入力資料の観測は稼働サーバーのジョブ記録と照合し、依頼の各項目を独立して検証した。

## 確認結果

| 項目 | 確認・対応 | 残る制限 |
|---|---|---|
| R1 OS依存 | 文書のjob-16173a8f25245ce3のログで4種類の共有ライブラリ不足を確認。専用Dockerイメージにlibxcb1/libgl1/libglib2.0-0t64とFFmpegを追加。CPU/GPU両コンテナでimport、ldd、PNG往復、MPEG4生成・ffprobe・5フレーム読込が成功 | 検証したコーデックはMPEG4。全コーデックの保証ではない |
| R2 Git | Gitを同じ実行イメージに追加。資料の固定SHAでtest_server.pyは2 passed。新しいサーバーの回帰テストも成功 | リポジトリ全テストの実行ではない |
| R3 定義 | 修正前の生tools/listにもresource=[half, all]、default=all、logical two-slotが存在。修正後も保持。直接MCP経由のhalf投入とqueue上のhalfを確認。省略したcourtジョブはall | ChatGPT登録・変換後スキーマとモデル向け定義は、このCodexセッションから取得できない。差の発生層・原因は未特定。ChatGPT経由のhalf実呼び出しは未検証 |
| R4 終了理由 | Docker状態と終了コードを保持しoutcomeを追加。正常、明示exit124/143、タイムアウト、APIキャンセルを実ジョブで区別。サービス再起動後にも同じ結果 | sandbox内の監督記録は診断用。GPU停止の確認・予約解放判断には使わない。証拠のない過去記録はunknown |
| R5 入力エラー | start_commandのSHA不一致をREVISION_MISMATCH、期待/実SHA、永続化したoperation相関IDで返す。HTTP MCPのstructuredContentでも確認 | ChatGPTの実画面での表示は未確認。全種類の入力エラーを構造化したわけではない |
| R6 能力検査 | core_torch/vision/video/testの独立profileを追加し配備workflowに組込。CPUのCUDA不可視、GPUはqueue経由を維持。実court_detectionの3バッチ更新も成功 | 25ステップの小規模計算・3バッチの実モデル検証であり、長時間学習の保証ではない |

## 実行環境と証拠

最終配備ランタイムは `a22dc473a4b77df2f1f2c597ce09bef23a968d96`。
イメージIDは `sha256:ba01df4149f11199dde3ac2d57995fdd84b4761b76a98169cacd5df2e5dad802`。
CPU/GPUジョブのDocker inspectでこのIDが実際に使用されていることを確認した。レジストリへpushしたイメージではないため、レジストリのRepoDigestとは区別する。

- [稼働環境・最終CPUジョブ](environment.json): 実ロード済みランタイムSHA、トンネルready、使用イメージ、再起動後の永続性。
- [固定SHAのCPU検証](fixed-sha-cpu.json): ソース `39f3e0cb7da844ad87227081839cbed0f21cf6ed`、job-8890cab5bc892e14。能力検査と指定の2テスト。
- [CPU/GPU能力検査](capabilities.json): ソース/当時のランタイム `9d2082e0110c668510126284e98dd2bc4f7f4109`、job-fcb8dea73582d432、train-ea29bfcc0a9e41cb。依存バージョンとuv.lock SHA-256を含む。
- [終了理由とSHAエラー](outcomes.json): ソース/当時のランタイム `615fb03e21a86e7cb517758b7b96cecd4cdffb0e`。exit124はfailed、期限超過はtimed_out、API停止はcancelled。
- [修正前の生tools/list](tools-before.json) / [最終ランタイムの生tools/list](tools-after.json)。登録・変換後スキーマの代替ではない。
- [court_detection検証ノード](../../../../../knowledge/nodes/run-mcp-court-three-batches.md): ソース/当時のランタイム `615fb03e21a86e7cb517758b7b96cecd4cdffb0e`。実画像、ローカルのDINOv3事前学習重み、KP loss、batch size 1。3回の有限勾配とdecoder weight更新、loss 0.20183 → 0.20168を確認。出力先はコンテナ内 `/tmp/mcp-court-smoke`。

上記JSONはこの作業でローカルHTTP MCPから直接取得した応答。元資料に記載された添付証拠をコピーしたものではない。ジョブcommandの再実行に必要なコート検証コードはknowledgeのrun bundleに保存した。

84件のMCP単体・統合・遅延importテストが成功。ruff、コミット時mypyが成功。設定/path auditは61 runtime boundariesを検証して成功。knowledge検証は0 errors（Issue未紐付け等のwarningsは残る）。

## 配備で判明した注意点

`install-runtime`単独では既存systemdのWorkingDirectory/PYTHONPATHは更新されず、runtime-versionファイルだけが新しい状態になる。実際に旧イメージで失敗した検証runもknowledgeに残した。今回の最終配備は、クリーンな別checkoutから既存キーを再利用する`configure-secure-tunnel --start`でservice定義まで更新した。

この誤認を防ぐため、health/get_host_statusはグローバルなruntime-versionファイルではなく、インストールされたパッケージ内の不変なrevision markerを返すよう修正した。

## ChatGPT側で残る操作

ChatGPTで「tennis-lab WSL」の接続を開きRefreshし、新しい会話でツール定義にresource/half/all/default=allがあることを確認する。その会話から軽量ジョブをresource=halfで投入し、get_training_jobのresourceもhalfであることを確認する。halfは論理予約であり、VRAMの強制上限や物理GPUの半分ではない。

これは開発モード接続に対する[OpenAI公式のRefresh手順](https://developers.openai.com/plugins/deploy/connect-chatgpt#refresh-metadata)に沿う。Refreshで解消しなければ、同時刻のサーバー生定義・登録後定義・モデル向け定義を併置して次の調査を行う。現段階でキャッシュが原因とは断定しない。
