# #2644 event_detection のキャッシュ識別をソース識別の互換規則に揃える

担当Issue #2644 の本文が受入条件の正本。親 #1722。

## 背景

- #2605（PR #2610、merge commit 264ac76a）で入れた `event_detection` は、`settings()` に `source_commit()`（checkout の完全な commit）を含めている。この値が実行キーと公開識別の両方に入るため、無関係なマージだけで再計算・`load` 拒否になる。
- 他の component は `src/tennis_scene/pipeline/source_identity.py` の `source_digests` と `compatible_source_digests`、および互換表 JSON で commit をまたいだ再利用を保っている。まずその仕組みと、`execution_plan.py`・`orchestrator.py`・`definition.py` でのソース識別の使い方を読んでから設計する。

## 制約

- 静かなフォールバックを作らない。識別が一致しない `load` は明示的に拒否する。
- `metadata.ball_events.provenance.code_commit` は計算時の commit の記録で、キャッシュ再利用で書き換えない。スキーマの検証規則は変えない。
- 公開済みの `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3` は変更・再生成しない。注釈フォルダ `data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events` はユーザーの成果物で、読むだけにする。
- `source_identity.py` と互換表は、別の親 #1739 の未マージ PR #2470（draft）も触っている。この PR の変更は event_detection に必要な範囲に留め、無関係な整理をしない。
- テストは、commit だけを変えたときの再利用（`execute` が cached、`load` が通る）と、event_detection のソース・設定を変えたときの再計算・拒否を、3つの method それぞれで検査する。git の commit を実際に作らずに検査できる設計にする（`source_commit` の差し替えなど）。

## 作業の区切り

実装・テストを書いて commit・push し、PR（`Closes #2644`）を作る。PR 本文に実行証拠（変更後の HEAD での関連テスト、配布版の SHA256SUMS 照合、v3 読込・検証の結果）を書き、CI を `PARENT_RESULT` の wait で宣言して終了する。#2600 の `post_train prepare` の入力照合は、PR #2613 がマージ済みなら main で、未マージならそのHEADを一時的に取り込んだ別の作業ツリーで確認し、結果だけを PR 本文に書く（#2613 の変更をこの PR に含めない）。
