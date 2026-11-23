# 親からの指示: PR #2433 の正式レビュー（round 1）の major 3件の修正

正式レビュー（run parent-2278-86ffba12c0cc4a768c776b42713f27e5、HEAD 71b132ed4）で、マージを止める指摘が3件出た。親は3件とも採用する。それぞれ修正と回帰テストを入れて、push する。

## 1. 既定 FP32 の identity が変わり、main の既存成果物を再利用できなくなる
- レビュアーの実行検証:
  - 固定 main と PR head のコミット済みソースを、既存の `compatible_source_digests` に通した。FP32 のコードハッシュは `6022ecaa…` から `e24029ab…` に変わった。
  - PR の差分で、既存の互換マニフェストと 17 ファイルのハッシュがずれ、`source_identity.py:26` の互換変換が適用されなくなる。
- 影響:
  - この値は、標準 component の execution key と publication identity の両方に入る。
  - 同じ入力・設定でもキャッシュが外れる。
  - 既存の公開済み annotation は、`overwrite=false` で古い設定として拒否される。
- 求めること:
  - 互換処理を、source identity の README の規約どおりに直す。FP32 の既存 identity を変えないことは、#2362 から続く要件。
  - 回帰テストを追加する。固定した main（この PR の merge base 603823383）由来の FP32 の identity を、現 head の標準経路で再利用できることを確かめる。
  - 現行の互換テストは、マニフェスト自身から入力を組み立てるので、このずれを検出できない。テストの期待値は、マニフェストから作らず、main 側の実ソースか、固定した digest で持つ。
  - 関連 PR #2379 の時点で同じずれがあったかも確かめ、最終報告に書く。

## 2. 先読みの途中終了で span が欠けても、記録が完了扱いになる
- レビュアーの実行検証:
  - 容量 1 の先読みキューで 1 件取得し、供給側が次の `put` で待っている状態で、利用側を閉じた。
  - clip 単位で例外を捕捉すると、供給スレッドと未終了の `decode_queue_wait` が残った。
  - それでも記録は `status=complete`、`measurement_errors=[]` になり、採番済みの span が 1 件欠けた。
- 同じ経路を通る場面:
  - 実際のボール推論で、`ball_detection.py:315` の予測・結果組み立てが途中で失敗したとき。
- 波及:
  - 残った span は、`inference_recording.py:195` の条件で、以後の退避も止める。
- 求めること:
  - 利用側が途中で終わったときは、供給側を止め、span を確定（中断として記録）してから、記録を書き出す。
  - span を確定できない場合は、完了扱いにせず、`measurement_errors` などで明示して失敗させる。黙って記録を欠かさない。
  - キューが満杯のときの途中終了の失敗テストで、この契約を確かめる。

## 3. 記録の有無で公開 identity が変わらないことを、テストで保証できていない
- レビュアーの指摘:
  - `test_inference_recording.py:137` は、記録あり・なしで、component の参照・execution key・保存バイト列を比べているが、dataset の公開までは通らない。
  - 代表 clip の検証では、両方の試行で記録が有効だった。切り替えたのは CUDA 計時の同期だけだった。
- 求めること:
  - 既存の小規模な公開 integration テストを使う。
  - 同じ入力・設定で、記録の有無だけを切り替える。
  - 公開マーカーの `publication_identity_sha256` と、公開 scene のバイト列が一致することを比べる。

## 共通
- 指摘ごとにコミットする。
- 関連テストと pre-commit を通す。
- PR 本文に、指摘ごとの修正内容とテスト結果を追記する。
- GPU 実行と E2E の再実行は不要。ただし修正 1 で、knowledge の記録（identity の値など）の記述が変わる場合は直す。
- push 後は CI 待ちを宣言して終える（`resume_on_success: true`、next_step は「最終報告」）。
- `gh run rerun` はしない。
