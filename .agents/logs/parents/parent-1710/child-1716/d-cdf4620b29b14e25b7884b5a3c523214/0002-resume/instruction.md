PR #1783 の統合レビュー（run parent-1710-cc89ad799a244573afc57abda462bd0b、PRへ公開済み）が request_changes でした。major 2件を同じbranchで修正し、push後のCI結果まで確認してください。

1. pr-1783-1（document.py:180付近）: 選択位置の索引が、ブラウザのDOM textContentと一致しない。`<pre>`[local path]`<textarea>`[local path]`<listing>` 開始直後のLFの除去、CRLF・CRの正規化（HTML標準の入力ストリーム前処理）、文字参照の展開を含め、UIが `textContent` から計算する位置と一致させる。期待値は手書きの独立したDOM値（実装からの写しではない）で回帰テストを追加する。UIが位置をどの単位（UTF-16 code unit か code point か）で送るかも契約に明記し、絵文字など非BMP文字を含むケースで一致を確かめる。
2. pr-1783-2（document.py:322付近）: `data-asset-id` もFigureも無い `img src`・`source src/srcset`・`a href`・`object data`・`link href` 等のmanaged asset参照からも影響先contentへ対応付ける。対応付けられない参照（どのcontent IDにも属さない等）は受理せず明示エラーにする。asset bytesだけを変えた候補で、正しい申告が受理され、未申告が拒否されるテストを追加する。

#1717・#1718 がこのbranchの上で並行して作業しているため、公開済みの型名・状態名・エラーコードは変えない（追加は可）。追加した場合は最終報告に列挙する。
修正後、HEAD・CI結果・各指摘への対応と回帰テスト（コマンドと結果）を最終報告に書く。目安は最大2時間。
