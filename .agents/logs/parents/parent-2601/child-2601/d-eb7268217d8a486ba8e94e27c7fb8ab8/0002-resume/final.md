正式レビューのmajor 2件を `0497bcfe0` で修正し、`codex/parent-2601-issue-2601-primary` にpushしました。PR本文と対応記録を更新済みで、CIは実行中です。[PR #2612](https://github.com/Motoki0705/tennis-lab/pull/2612) · [レビュー対応記録](https://github.com/Motoki0705/tennis-lab/pull/2612#issuecomment-7188136132)

- 連鎖切れになるクリップの件数と名前を保存前に表示します。得点側・状態・理由を保持し、開始前カウントを確認し直すまで終了後表示と後続への連鎖を止めます。CSV出力は該当名付きで拒否します。
- 硬式の第2セット以降は、開始前を手入力したクリップの保存済みサーブ側を後続計算の基準にします。直前からの計算と食い違う場合は、画面に両方の値を表示します。ソフトテニスでは、同じ正規化カウントに至る場合の自動サーブ側が一致することを確認しました。

pytest 254件（今回5件追加）、Node回帰10件、pre-commitが成功しました。合成動画の編集・読取専用ブラウザE2Eと、明治4カメラのv1一時コピー69件の再読込も確認しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2612,"head":"0497bcfe0ee4c7188ad541a387886e373fd5af08","resume_on_success":false}}