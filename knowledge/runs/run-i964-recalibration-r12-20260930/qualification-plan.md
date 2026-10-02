# clip_000 full-pipeline qualification（次run、別GPU許可）

対象は `video_000/clip_000` のcam0/1/2、1,010 source frame、59.94006fps。
このrunはCPU事前点検まで。**[再較正protocol](protocol.md)の最終YAMLがcommitされる前には実行しない。**
採用保留なら較正の判断を先に解決する。未見予約clipには触れない。

## 固定する入力

- `pipeline_preflight.py` の出力を出発点とする。資産rootは元repoのdata/ckpt/third_party、
  実装rootは本worktree。全16種（3camera展開後28node）はexecute、全機能enabled、
  storeは新しいrun専用directory。ball・side・人物対応のimportは0件。
- person条件はprotocolどおり。pose batch4、HMR2 batch4を明示し、batch以外の既定は変えない。
  ball checkpoint/設定も解決済みYAMLに固定する。#935/#936の未統合branchを混ぜない。
  現在の球は既定ft-e13であり、refiner導入後の最終#930 qualificationではない。
- AFLinkはcheckpoint root下の通常ファイル。配置・hashは
  [aflink-placement.json](aflink-placement.json)、利用条件の不確実性は
  [NOTICE](../../../src/tasks/person_tracking/strongsort_NOTICE.md)を正とする。重みを公開物へ含めない。
- 最終association YAMLを指定してCPU preflightを再実行する。動画3本・全checkpoint/設定・
  実装・AFLinkのhashをfreezeコメントへ投稿する。CPU検査の成功をGPU完走とは数えない。
  source frame数・fps・camera集合を出力契約へ固定し、frame間引きやmax_framesで短縮しない。

## 実行方法と資源

新しいentryは標準 `TennisSceneOrchestrator.run` を全executeで呼び、
scene export後に新しいprocess/ClipStoreで全nodeをload-only再開する。
旧 `component_pipeline.py` はballをimportするため、この合否判定にはそのまま使わない。
次runで実行receipt/監視付きの入口を整備・通常検証してからqueueへ登録する。

提案枠: resource=all、GPU45–90分（wall上限90分）、peak見積り10–12GB、
新規disk1–2GB、CPU監査/動画10–20分。モデルは段階ごとにunload。
host RAM available>=6GiB、CPU最大4thread、pytest -n4。
全GPU VRAMを定期取得し、11.5GB到達で本jobだけを止める。allocatorを10GiB未満へ制限する。
次directiveがより小さい上限なら、その枠内で実行計画を再固定する。
この費用は**申請予定の推定**であり、run 12の不足特徴用2時間枠を転用しない。

## 合格条件（全て必要）

1. **3camera完了**: cam0/1/2のcourt、person detection/tracking/selection/pose、
   ballが全source frameを処理。courtは仕様どおりframe0の1推論であると明示する。
   raw/source rowとtime axisが一致し、全cameraの有効なcourt calibrationを持つ。
   side、人物対応、三角測量、GVHMR、body placement、scene assemblyまで理由付き停止なし。
   body viewは仕様の1camera/人物を使い、3cameraすべてでGVHMRを走らせたとは称さない。
2. **artifactの検証/再読込**: 全nodeのstatus=executed（新store）、schema/version・
   依存鎖・checksum・型・shape・有限値の検証をpass。person_detections v2 / tracks v5 /
   selected v2 / identities v3を確認する。別processのload-onlyは全node loadedで、
   モデルforwardを0回にする。scene.json→immutable export→SceneResultが同一で、
   全配列・metadataが初回と一致する。再loadによるstore変更なし。
3. **観測/選択契約**: GSI syntheticとobservedが交差0、元rowが実検出へ一意に戻る。
   cap6は選別groupにだけ適用。選択断片の実観測を領域で再切断しない。
   sceneはシングルスの2人物を持ち、各人で有効な3D root/COCO17/body出力が1frame以上ある。
   ballも2view以上に由来する有効3Dを1frame以上持つ。欠測frameは理由コード付きであり、
   100% coverageや絶対的な3D精度は合格条件としない。
4. **3camera動画レビュー**: 全約16.85秒を同期した3列で書き出し、
   raw ID、group、camera間person ID、実box、GSIを区別する。全frameをdecodeして
   frame数/fps/長さ/動画SHA-256を検証する。全長を確認し、0/25/50/75/100%の5時点、
   全handoff/ID変化の前後.5秒を追加確認し、時刻付き所見を保存する。
   目視で別人へのID交換、二人を一人にした区間、syntheticの実観測化を認めれば不合格。
   疑義を未確認のまま合格にしない。新しいdevラベル採点/調整には使わない。
5. **runtime/VRAM**: build/モデルload/推論/保存/CPU再読込/動画を分けたwall time、
   各node秒数、process peak RSS、torch peak allocated/reserved、
   GPU全体の監視peakとsampling intervalを保存する。
   割当wall/VRAM/disk枠を超えず完了する。監視欠損・OOM・timeoutは不合格。

## 失敗時と引き渡し

既存artifactや注釈へのfallback、side/IDの補完、失敗cameraの除外で合格にしない。
失敗理由・最後の完成artifact・資源peakを記録し、技術修正なら別runで証拠を分ける。
尺度/しきい値をdev動画を見ながら変更しない。合格時はreceipt、freeze commit、全hash、
3camera動画、レビュー所見、制限をIssueとPRの新しいレビュー観点コメントへ添付する。
合格後も予約未見の一回評価は別の凍結/許可手順で行う。
