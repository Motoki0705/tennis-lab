[改訂ドラフト](https://github.com/Motoki0705/tennis-lab/blob/fde68423ad15bdb8d583884680ed95cb0cf3e30f/tmp/wiki/Meiji-4cam-Benchmarks.md) を作成し、公開前検証を完了しました。冒頭の定型欄を廃止し、4版の系譜、ファイル・配列仕様、10のモデル利用経路を階層化しました。本文は242→402行、表は8→0件です。

- 旧版の全242行、62リンク、16アンカー、5コードブロックを突合し、情報欠落は0件でした。固定 revision と取得・検算コマンドの動作も維持しています。Hugging Face は匿名で HTTP 401 を確認し、権限付き取得は未確認です。[移行検査](https://github.com/Motoki0705/tennis-lab/blob/fde68423ad15bdb8d583884680ed95cb0cf3e30f/tmp/meiji-revision/migration-audit.md)
- Chromium のライト・ダーク両方で、既定表示の高さは通常幅で10,477→9,292px、390px幅で17,985→15,140pxになりました。390px幅で横スクロールが必要なブロックは9→0件で、折りたたみを全開しても収まります。[表示比較とスクリーンショット](https://github.com/Motoki0705/tennis-lab/blob/fde68423ad15bdb8d583884680ed95cb0cf3e30f/tmp/meiji-revision/render-check.md)
- Mermaid、3件の折りたたみ、既存および短縮アンカーはローカル検証を通過し、公開禁止情報の検出は0件でした。[公開前検査](https://github.com/Motoki0705/tennis-lab/blob/fde68423ad15bdb8d583884680ed95cb0cf3e30f/tmp/meiji-revision/preflight.json)

差分はローカル Wiki の対象1ファイルです。本番反映と公開後の表示確認は人間承認待ちです。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}