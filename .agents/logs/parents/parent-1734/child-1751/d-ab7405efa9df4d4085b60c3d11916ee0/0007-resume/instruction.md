# PR #2182 再レビューのmajor指摘への対応（親 #1734 から #1751 へ）

再レビュー（gpt-6.2-astra / xhigh）で、新たに6件のmajor指摘が出た。親の判断は各項の末尾に書いた。各指摘について、失敗条件を再現するテストを先に書いてから直してください。

## 1. モデル解放に失敗したnodeが、再起動後はcache hitで完了する

CPU実行検証。実PlanExecutorとメモリ上のstoreでunload hookを失敗させると、nodeはfailedになりましたが、同じstoreで新しいexecutorを作るとcachedになり、完了通知が呼ばれました。runner.py:144のstore.publishはruntime.nodeの終了前に実行され、実ClipStoreもscene.jsonのactive参照をその場で更新します。参照の永続化を解放成功後に移し、既存の再開テストを新しいexecutorでも検証する必要があります。

親の判断: **直す。** 参照の永続化（store.publishとscene.jsonのactive参照の更新を含む）を、解放の成功後に移す。新しいexecutorでの再開テストで確かめる。

## 2. CPUのCourt成果物がCUDA指定の実行でも再利用される

CPU実行検証。資産を固定し、標準設定のdeviceだけをcpuからcudaへ変えてstandard_definitionとExecutionPlan.decideを通すと、Courtのexecution keyは同一で、CUDA側はcachedになりました。definition.py:144のCourt設定にdeviceが含まれず、cache判定はModelSpecの照合より先に終わります。共有モデルのキーを分けても、要求したdeviceの推論は実行されません。成果物のキーにも有効なモデル設定を反映し、同じstoreで設定を変える回帰テストが必要です。

親の判断: **直す。** Courtの成果物の設定identityに、有効なdevice（と精度などのモデル設定）を入れる。同じstoreで設定を変える回帰テストを書く。cache identityが変わる範囲をPRに明記する。

## 3. 外部DINOv3の実装を切り替えても、後続clipが最初の実装を使う

CPU実行検証。実際のhubconfと同じ絶対importを使う軽量な2実装をtorch.hub.loadでA、Bの順に読むと、B単独ではBを返す処理が、連続実行ではAを返しました。Courtの共有キーは分かれても、実ローダーのdinov3.hub.backbonesはsys.modulesに残ります。またDINOv3外観encoderでは、model_resources.py:29のキーに外部実装が含まれず、生成部をfakeにした実runtimeでもBがAのencoderを借りました。実装を分離して読み込むか、同一プロセスで安全に切り替えられない組合せを明示的に拒否する必要があります。

親の判断: **直す（明示的な拒否でよい）。** 同じプロセスの中で、外部実装（DINOv3のhub実装、外観encoder）を安全に切り替えられない組合せは、明示的に拒否する。共有キーに外部実装のidentityを入れる。

## 4. runtime作成後に追加されたCourtのPythonソースを検出できない

CPU実行検証。Courtの資産一覧をメモリ上で差し替え、ModelSpec作成後・初回borrow前にソースを1件追加すると、実runtimeは生成を許可し、ローダーから追加ファイルが見えました。inference_runtime.py:249の照合は宣言済みのパスだけを走査します。追加ファイルがimport先を変えると、古いSHA一覧で別実装の出力を公開できます。既存のsource_addedテストは追加後にmodel_specs()を作り直すため、この順序を検出しません。

親の判断: **今回は直さない（対象外）。** 実行中にPythonソースを書き換える操作は、実行中にコードを編集することと同じ種類の運用であり、このPRの対象外とする。PRの「既知の制限」に1行で明記する。

## 5. 同じ重みを別パスから共有するとCourt成果物の来歴が実行順で変わる

CPU実行検証。同内容の別パスA、Bを宣言し、予測部をfakeにして実CourtKPModuleを順に通すと、Bのdiagnostics.checkpoint.checkpoint_pathにもAが記録されました。court_kp.py:156が最初に生成した共有predictorのcheckpoint_identityをそのまま各成果物へ入れるためです。B単独の実行と共有実行でpayloadとartifact IDが変わります。モデルの内容は共有し、パスや移行元rootなどの来歴は各nodeが束縛した資産から記録する必要があります。

親の判断: **直す。** モデルの内容は共有してよいが、成果物の来歴（checkpoint_pathなど）は、各nodeが束縛した資産から記録する。B単独の実行と共有実行で、payloadとartifact IDが同じになることを確かめる。

## 6. 全件cache hitでもCourt checkpointの全量ハッシュがcamera数に比例する

CPU実行検証。a5の実checkpointと3台の標準Court nodeで、全量ハッシュ関数をカウンタに置き換えて計画を構築すると、cache判定に入る前にtorch.loadが6回、全量ハッシュ要求が12回発生しました。通常のハッシュ関数に渡る合計は15.85GBです。definition.py:145とrunner.py:92が各cameraで資産を宣言し直し、checkpoint.py:171がそのたびにcheckpoint全体を前後2回読みます。変更検出を保ったまま同一資産のmetadataとdigestを計画内で共有し、モデル生成回数に加えてcheckpointの読取回数も検証する必要があります。

親の判断: **直す。** 変更の検出は保ったまま、同じ資産のmetadataとdigestを計画の中で共有する。全件がcache hitする場合の、checkpointの読取回数とハッシュ量をテストで固定する。一括処理（69clip×4camera）で効くため重要。

## 共通の条件
- 静かなフォールバックを作らない。
- GPUの再受入は不要。出力・identityの計算を変えた場合は、変わる範囲（cache identityを含む）をPRに明記する。
- CPUの関連テスト、pre-commit、CIを通し、PR本文に各指摘への対応を追記する。merge・closeは親が行う。

