#1708の実装・PR作成・CI確認まで完了しました。

- PR: [#1763](https://github.com/Motoki0705/tennis-lab/pull/1763)
- branch: `codex/single-camera-gpt-annotation`
- HEAD: `842cf7cc5db7a424165e22fc17fc6e8ecbec3f2c`

camera単位のサービスとLandlock読取境界、原子的な途中保存・復旧、最終unknownを含む完成確定、探索方針の交換と独立した比較・計測を実装しました。

テスト増分は25 case。pre-commitと[最新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42514916494)は、CPU全10分割・Lint・必須Python jobすべて成功しました。ローカル回帰の既存2件の失敗は変更前mainでも再現し、PRに記載しています。

実GPT・実動画での精度検証は後続#1715です。#1709向けの保存revision・未処理範囲の取得口も用意しました。追加判断依頼はありません。統合レビュー・merge・merge後mainの確認を親へ引き継ぎます。