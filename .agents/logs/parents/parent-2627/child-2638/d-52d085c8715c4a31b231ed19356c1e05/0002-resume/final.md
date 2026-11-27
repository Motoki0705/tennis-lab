正式レビューの3件を修正し、Readyのまま [PR #2720](https://github.com/Motoki0705/tennis-lab/pull/2720) にpushしました。PR本文の実行証拠も更新済みです。

- 第4章の関節残差式とSMPL rootの配置式に、選手ごとの骨長比から求めたスケールを反映しました。
- 第6章で、オンラインの外観・運動・残余IoU照合にposeコストを加えることと、AFLinkが実観測のフレーム番号とボックス座標を使うことを明記しました。
- 付録にSceneResult v2の6マスク・3理由コード配列と、図表の再生成・照合コマンドを収録しました。コード値は保存処理と照合済みです。

HEADは `f28881f3e2c0a57c473430e064d37471ae89508f` です。クリーンなHEADで `build.sh --check-tracked` は終了値0。PDFは26ページ・14図・17表で、警告・Overfull・文字欠落・未解決参照や引用・未追跡入力は各0件でした。根拠49行、原稿中の対応527箇所、固定資料106ファイルを照合しました。PR CIの結果は確認前です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}