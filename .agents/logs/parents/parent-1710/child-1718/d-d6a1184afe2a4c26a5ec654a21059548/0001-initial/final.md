[PR #1786](https://github.com/Motoki0705/tennis-lab/pull/1786) を作成しました。

- branch: `codex/explain-html-visuals`
- HEAD: `b2dc6bd417f2adac2812abcdd26c2f7bcb8b6ef8`
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42553570091): **Python・Lint・CPU全10本成功**

画像のcaption・alt・由来と、SVG・宣言的グラフ・rendererを既存storeの復元[local path]。検証は追加unit 9件＋Node 3件、browser 2経路。依存側を含む **51件が5.24秒で成功**し、pre-commitも通過しています。browser共有setupは0.646秒、各経路は0.443秒[local path]。

初回imagegenは1枚生成し、[実画像](outputs/explain-html/visual-probe/initial.png)のimport・保存・offline表示を確認しました。**閲覧サービスからの新規画像生成は未実証**です。当面「初回生成→import、閲覧中新規生成は理由付き未完了」とするか、agent turn経由adapterの実機検証を進めるか、人間判断が必要です。当該Acceptanceは未チェックです。

#1716 の最新 `6c4acc2`へ追従済みで、契約変更要望はありません。サービス[local path]、統合レビュー・merge・merge後main CI確認は親へ引き継ぎます。