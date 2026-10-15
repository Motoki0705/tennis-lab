# 生田周辺テニスコート撮影候補（3DGSシーン収集用）

コート検出データセット拡張のための撮影候補地リスト（小田急線沿線中心、2026年10月調べ）。

| ファイル | 用途 |
|---|---|
| `ikuta_tennis_courts.pdf` | A4 1枚の印刷用（概略地図＋電話番号＋GoogleマップQR＋メモ欄） |
| `ikuta_tennis_courts.html` | 同内容のHTML版（QR部分がGoogleマップへのリンク） |
| `ikuta_tennis_courts_mymaps.csv` | Googleマイマップ取り込み用 |
| `ikuta_court_route.pdf` / `.html` | 駅近コート（最寄駅から徒歩15分以内の11か所）を生田駅発着で巡る最短距離ルート（A4 1枚、2日分割） |

## Googleマイマップへのプロット

1. <https://www.google.com/mymaps> →「新しい地図を作成」
2. レイヤの「インポート」で `ikuta_tennis_courts_mymaps.csv` を選択
3. 位置の列に「住所」、マーカーのタイトルに「名称」を指定

## 注意

- 地図の座標は Wikipedia・NAVITIME・MapFan 等の掲載値（一部は「位置推定」）。正確な位置は住所・QRで確認する。
- 巡回ルートは「徒歩＋鉄道」ネットワーク上の最短路を使い、巡回順を Held–Karp 法で厳密に求めた。徒歩距離は公式の徒歩分×80m、記載のない区間は直線距離×1.3 で近似している。
- 電話番号・面数はWeb上の施設情報に基づく。「面数要確認」は未確認。
