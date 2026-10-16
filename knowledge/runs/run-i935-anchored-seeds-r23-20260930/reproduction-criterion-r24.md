### 進捗ログ run 24 (2026-09-30) seed43評価を開く前の再現性判定基準

- 実施: run23 queueの失敗状態と資源reportのみを回収。seed43の評価出力・metrics・checkpoint選択結果は未読。この宣言をissueへ投稿しcommit/pushしてから扱う。
- 成果物: この事前判定基準。前回job 1790762149588430733_2215834 はRAM単発低下でfailed。resource reportはMemAvailable=6,239,236,096 bytes、GPU使用=1,629,487,104 bytesを記録。seed43学習1159.42秒＋評価41.80秒は終了し、seed44は中断。
- Acceptance checklist 状況: A設計/C/Dは前回完了。seed再現確認とBは未完了。Bの既定をこのrunで変更しない。
- 次の一手: 30秒継続/3GiB即時停止のRAM guardとテスト、seed44だけ新規12k学習、明示pipeline optionと3camera元動画execute/loadの2jobを順に投入する。

【要判断】seed再現性の事前判定規則（run24）
- 前提情報: seed42は既報。seed43評価は未読、seed44は未完。e9 detector・absolute_12k(seed42)・Meiji video_000 val・partition/gap seed42・checkpoint選択規則は固定。video_001を一切使用しない。
- 選択肢: A 各追加seedで位置分位点3項目とNLL2項目の全条件を満たす / B seed平均だけで判定する。
- 採用した暫定案と理由: A（orchestrator推奨）。**seed43とseed44それぞれ**、Meiji val全camera・両halfを合算したobserved教師の通常入力で、既存評価の点推定誤差median/p90/p95がe9 detector top-1よりすべて厳密に小さいこと。さらに同じMeiji val observed教師で、通常入力の位置NLLと人工evidence_gap frameだけの位置NLLがabsolute_12kよりそれぞれ厳密に小さいこと。全10比較が合格して「再現」とする。同値は不合格、未完/欠損/非有限は未確認として既定切替不可。checkpointは選択halfのobserved/gap等重みNLL最小、同点で早いepochを維持し、結果を見て選び直さない。
- NLLの判定対象は**倍率適用前**（head設計の再現性を同条件で比較）。共分散倍率は前回artifactの正確値1.8125148752（略記1.8125）をseed42/43/44へ固定適用し、NLL/HDR50/90/95 coverage/面積を別列で報告する。新seedで再fitせず、mean/mixture/presenceは変更しない。raw合格だけで較正後のNLL改善を主張しない。
- seed42/43/44の各値とmin/max/spreadをsource（Meiji/TrackNet/chat）・camera・selection/calibration half・通常/gap別に報告し、局所退行を隠さない。Meiji全体のゲートと層別診断を区別する。seed42でfitした固定倍率の評価はLOCO OOFではなく、独立test性能とも呼ばない。HDR MC2048/seed1729/levels50,90,95は前回どおり。
- 覆す場合の影響範囲: この判定と後続Bの採否のみ。既存checkpoint、partial seed44、cache、旧bankを保持。B切替は上記判定と元動画3camera execute/loadが両方通った後、次runでのみ行う。

