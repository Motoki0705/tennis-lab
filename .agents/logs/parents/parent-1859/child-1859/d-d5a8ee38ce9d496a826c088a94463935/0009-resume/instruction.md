# C3完了前に、積み重なった5本のPRをmainへ入れる準備（親から）

## 人間の判断

2026-11-18、人間が「C3を完成させる前に5本のPRをマージさせたい」と判断した。親が不安要素とその対策を示し、人間が「はい、それでは進めてください」と承認した。

- 対象PRとmergeの順番（依存の順）: #1892 → #1896 → #1897 → #1950 → #1955
- 対象外: #1906（draft、#1897の上）、#1903（#1899のdraft、#1892の上）、#1948（memory、別途閉じる予定）。これらのbranchには触れない。

## 厳守する制約

- **GPUを使わない。** 人間がGPUの使用を禁止している。テスト・検証・計測はすべてCPUで行う。全コマンドで `CUDA_VISIBLE_DEVICES=""` を設定する。training queueにjobを入れない。GPUでしか確認できない項目は、未確認として報告する。
- **C3の実行commitを残す。** C3の実行commit `7d7a2e4fa` には、親がannotated tag `ball-arch-c3-exec-7d7a2e4fa` を付けてpushした。
  - 5本のbranchはrebase・force-pushしない。mainや下のbranchの取り込みは、すべてmerge commitで行う。
  - `7d7a2e4fa` が `codex/ball-arch-desktop-20261113` の祖先のままであることを確認する。
- **実行用のworktreeとE:の出力には触れない。** `ball-arch-desktop-runtime-20261113` のworktreeと、E:上のattempt root（`desktop-recovery-*`）を変更・削除しない。
- PRのbase branchは変えない。mainへの付け替えは、親がmergeの直前に1本ずつ行う。
- mergeはしない。review要求とmergeは親が行う。

## 依頼

1. **#1892 のコンフリクトを解消する**（`codex/ball-arch-port-20261111`）。
   - `origin/main` をmerge commitで取り込む。
   - main の `fcc317f86`（scene で X3D/CSN DPT T32 を明示選択する。`x3d_csn.py`・`cnn_encoder.py`・`detectors.py`・`config.py`・README・`inference/x3d_dpt.py`・`pipeline_x3d_dpt.yaml`）と、#1892 が入れる X3D/CSN の実装を、1つの実装にまとめる。重複した実装を残さず、静かなfallbackも作らない。
   - **同等性の検証（CPU）**: 次の3つについて、統合前と統合後で出力が完全に一致すること（最大差0）と、checkpointのstrict loadを確認する。
     - main の scene X3D DPT T32 推論の経路（実checkpointと実入力）
     - #1892 が対象とする #1675 の保存checkpoint（DPT 10k、dim512・6層query）
     - #1859 の C1〜C3 / T1・T2 の構成のforward（小さな入力でよい）

     一致しない場合は、理由を報告して止める。
   - main の `x3d_csn` テストと #1892 のテストを統合し、両方の意図を残す。
2. **上のbranchへ順に取り込む。** 更新した下のbranchを、merge commitで1つ上に取り込む。
   - 順番: `codex/ball-arch-variants-20261111` ← #1892、`codex/ball-arch-execution-20261111` ← #1896、`codex/ball-msda-checkpointing-20261113` ← #1897、`codex/ball-arch-desktop-20261113` ← #1950
   - コンフリクトがあれば、各PRの意図を保って解消する。
3. **mainと合わせた状態でテストする（CPU）。** 最上段の #1955 の head に `origin/main` を合わせた状態で、次を実行する。
   - ball_detection の unit・integration
   - tennis_scene のうち ball・X3D に関係するテスト
   - `src/utils/models/components/ops/msda` のテスト
   - ruff・mypy（pre-commit）

   失敗があれば直す。CUDAが必要で実行できないテストは、未実行として一覧にする。
4. **PR本文を直す。** 5本それぞれの本文に、次を日本語で短く追記する。
   - C3がまだ完了しておらず、統合結果は途中であること
   - C3の再開は、tag `ball-arch-c3-exec-7d7a2e4fa` の固定sourceで行うこと
   - mainとの統合で何を変えたか

   #1955 と knowledge（`run-ball-arch-efficiency-s42-20261111` と関連node）には、「C3未完了・途中結果」と明記する。
5. 全branchをpushし、各PRのCI（Python・Knowledge）の成功を確認する。

## 終了条件

- 5本ともmainとコンフリクトがない。
- 各headのCIが成功している。
- 同等性の検証が通っている。
- `7d7a2e4fa` が #1955 branchの祖先である。

ここまで確認したら終了する。作業時間の上限は3時間。

## 最終報告

次を書く。
- 各PRの新しいhead
- 統合した X3D/CSN 実装の要点
- 同等性検証の方法と結果
- 実行したテスト・未実行のテストとその理由
- CIのURL
- 親がreviewで見てほしい点
