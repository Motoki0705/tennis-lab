# Player Dataset Review

保存済みのplayer frame storeを読み取り専用で確認するWeb画面です。原画角と選手ごとの拡大を並べ、
注釈bboxの出自、遮蔽・切れ、位置未解決、既定の学習選別の理由を確認できます。

## データの位置づけ

| 系統 | 用途 / 入力と教師 | 出自 / 単位 | splitと現行の扱い |
|---|---|---|---|
| `data/player_detection/chat-player-v1` | DINO fine-tuning用。RGB + player bbox・clip-local track ID・時刻 | chat-annotationから作った固定snapshot。bboxは元画像pixelの`xyxy`。`observed / inferred / unresolved`は注釈状態で、モデル確率ではない | YouTube source動画単位のtrain / val / test。現行の学習storeで、この画面の対象 |
| `outputs/chat_annotation/annotated/processed/player` | store生成に使う上流player JSON | 処理済み注釈。全画素精度の独立検証済みGTとは限らない | 上流の現行入力。既存storeへ自動同期されない。直接のレビューは[Chat Annotation](../../../tennis_scene/chat_annotation/README.md)を参照 |
| Meiji clip配下の`annotations/player_association/labels.json` | 検出器・source比較の評価用部分ラベル | 旧COCO trackerのboxをreviewしたperson単位の参照。未ラベル予測は未知でありFPとは限らない | 学習storeのsplitとは別。固定dev / unseenは保存protocolに従う。この画面へ混ぜず、既存[benchmark](../../../../tests/benchmarks/README.md)と評価readerを使う |

storeのschemaとJPEG shard読出しは[data/store.py](../data/store.py)、学習選別は
[data/detection_dataset.py](../data/detection_dataset.py)、既定の条件は
[configs/data/default.yaml](../configs/data/default.yaml)が正本です。
画面の件数は実際に選択したstoreを集計し、build日を表示します。

このstoreはplayerを列挙したframeだけを保存します。元clipの未保存frameはタイムラインと件数で明示し、
完全な負例として扱いません。`unresolved`は選手の存在記録はあるものの位置を解決できず、bboxがない状態です。
`inferred`は遮蔽や画面端の切れ等の推定・補完で、数が多いことだけで誤りや利用不可と判断しません。
`partial`と注釈注意事項を確認し、`reviewed=true`を全pixel精度の品質保証と取り違えないでください。

Git履歴では、frame storeは[82996257f](https://github.com/Motoki0705/tennis-lab/commit/82996257fba34a5b8015cc5e7b0bb26d9b7566a3)で導入されました。
Meiji部分ラベルの評価は[312f11ec8](https://github.com/Motoki0705/tennis-lab/commit/312f11ec8c7242bef3f3fbeae14ccc0f69d70129)で追加され、
[6e324f8c9](https://github.com/Motoki0705/tennis-lab/commit/6e324f8c9ceb8e2fd0ea6979f4bd9fa1643ec759)でdataset-ownedのclip配下へ移動しました。
以前の`tests/benchmarks/fixtures/player_association`配置は過去の配置であり、現行の不足データとして復活させません。
過去の推論archiveや可視化成果物は実験結果で、現行の学習dataset familyとは区別します。

## 起動

worktreeからも既定でgit common rootの既存データを参照します。

```bash
/home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.tasks.player_detection.scripts.review_dataset --port 8895
```

ブラウザで`http://127.0.0.1:8895`を開きます。
別のデータrootを読む場合は`--project-root /abs/project --data-root /abs/data`を指定できます。
dataset生成、注釈保存、学習、推論、自動処理再開の操作はありません。
対応するローカルstoreがない場合やschema不整合は理由を表示し、代替データを生成しません。

## 確認方法

1. dataset、split、source動画、clipを選びます。検索はsource名 / ID / clip IDを対象にします。
2. 状態filterで、位置未解決、推定・補完、遮蔽、切れ、学習選別で除外されたframeを絞ります。
   sliderと前後移動はそのfilterに一致する保存frameを進み、元のclip frame番号とsource frame番号を表示します。
3. 原画角ではobservedを緑の実線、inferredを橙の破線で表示します。bboxを消して画像を確認できます。
   mouse wheelで拡大、dragで移動、全画角でリセットできます。選手の拡大は原画像からの切り出しです。
   unresolvedには座標を描かず、座標なしのカードを表示します。
4. 右側の既定選別は、review済み・位置未解決なし・画像内box最短辺の条件を反映します。
   frameが除外される場合、位置が分かる他の選手のboxも使用候補として表示しません。
   per-split stride / epoch sampling前の候補であり、品質合格や学習への最終採用ではありません。
5. 注釈注意事項を開き、補間・amodal境界の不確実性を確認します。欠落frameは負例に置き換えません。

再生は確認用の5 fpsです。filterで飛ばしたframeやstoreの欠落を含むため、元動画の速度を再現しません。
URLのhashにdataset / clip / frame / flagを保存するので、同じ箇所を再表示できます。

## 検証

`tests/unit/tasks/player_detection/test_dataset_review.py`は、未解決bboxをJSON nullで保持すること、
未保存frameの拒否、amodal / image-clipped boxの区別、既定学習選別との一致、読取専用HTTP、
データの内容ハッシュ不変を検証します。CLIは`test_review_cli.py`で入力path検証とinventory登録を確認します。
unit-test fixtureは検証専用で、レビューの証拠画像には使いません。
