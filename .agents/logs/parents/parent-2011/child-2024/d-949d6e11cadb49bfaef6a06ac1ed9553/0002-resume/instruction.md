PR #2043 の統合レビュー（run parent-2011-ed68cc4ba6e14bd68c1ce0aefab2cce8）で major 2件の指摘がありました（レビュー出力の形式不備でrunは未完了扱いですが、指摘内容は親が確認し妥当と判断しました）。今回は2を修正してください。1は探索CLI側の形式確定後に別途依頼します。

2. PythonとTypeScriptで文書の受理・拒否条件が一致していない（citation-documents.ts）。
   - `created_at='2026-11-14T22:00:00+0900'` はPythonの serialize_document / validate_library を通るのに、UIは拒否して探索一覧全体がエラーになる。
   - `source='https:example.org/paper'` はPythonが拒否するのに、UIの httpUrl（92行目付近）はWHATWG URLで補正した後の値を検査して受理する。HTTPSのUIでは自サイト配下へ解決され、一次資料リンクを誤る。
   対応:
   - 日時・URL・外部IDの正規化など境界条件を、knowledge/README.md の契約に揃えてください。契約が曖昧ならREADMEを明確にし、PythonとTSの両方をそれに合わせてください（Python側の変更が必要なら同じPRで行ってよい）。
   - 受理すべき例と拒否すべき例を1つの共有fixture（例: `tests/fixtures/knowledge/citation_map/` 配下の境界ケース集。形式は自由）にまとめ、pytestとTSのunit testの両方から同じ例を読み込んで、両実装の判定が一致することを検証してください。今後の二重実装の乖離をこのテストで検出できるようにするのが目的です。

参考（今回は対応不要）: 1. 横断指摘として、PR #2044 が中断探索用に `status` / `stop_reason` キーと部分展開を追加しており、UIはそれを拒否する。#2044 は修正中で形式が変わる見込みのため、merge後に合わせる。

lint・unit・build・e2e・関連pytestを通してpushしてください。
