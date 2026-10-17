# native frameに対応するレビュー画像

`render_page(layout, RenderRequest)` は一ページの `RenderedImage(image, metadata)` を返す。
requestはclip manifest、prediction/revisionの参照と種類、描画方式、提示理由、frameごとの
native cropと任意の点/bboxを持つ。時間文脈・追従位置・全体表示の選択はtaskの責務とし、
baseは指定されたframe/cropを描く。範囲外を黙ってpadding・clampしない。

- 一ページ16 tiles、4列、tile各辺1024 px、ページ全体800万pixelを上限とする。
  native frameは3200万pixelまで。上限超過は小さく丸めず拒否する。
- sourceをhash検証し、必要な最初〜最後のframeをPTS照合して逐次decodeする。
  選択frameだけRGBへ変換し、crop/resizeしたtileはページに貼り付けて解放する。
  clip全体のRGBや複数ページを保持しない。描画・保存は親の資源受付後に呼び出す。
- 各tileのframe index、source frame、clip/source PTS、time base、crop端、出力位置と寸法、
  実際のx/y倍率をreceiptに残す。aspect fitの丸め後の倍率は有理数で保存する。
- `native_to_image` / `image_to_native` は `coordinates="pixel_center"` と `"edge"` を
  明示させる。点はPILのhalf-pixel resize、bboxはpixel端の座標を使う。ヘッダや余白の
  クリックは元座標へ変換せず拒否する。

マーカーなしでも参照するannotation版を固定する。描画変更・crop変更・版変更は別requestで、
`annotations/rendered/<request hash>/` に不変保存する。キャッシュ再利用時も参照とgeometryを照合する。

`RenderedImage` をworkerのtelemetry catalogへ渡すと、実際のtool画像bytesとの対応を追える。
`presented_tiles()` は指定したmanifest・annotation版と、応答IDに結び付いた実画像入力を照合する。
画像生成、画像提示、taskによる確認、教師採用は別状態で、このAPIは注釈状態を変更しない。
cropの外に対象がないという判断はできない。確認範囲・不在の根拠・教師maskはtaskの方針で検証する。
