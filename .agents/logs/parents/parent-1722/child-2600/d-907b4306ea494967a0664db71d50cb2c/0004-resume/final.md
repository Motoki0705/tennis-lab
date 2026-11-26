#2600 のGPUジョブ10件が完了し、9本の融合重みと110件のtest予測を照合しました。macro-F1は次のとおりです。追加学習3方式は3 seedの平均です。

| 方式 | v3 全イベント | v3 ポイント内 | 独立6本 全イベント |
|---|---:|---:|---:|
| 球のみ | 0.671 | 0.726 | 0.505 |
| Shared | 0.665 | 0.722 | 0.566 |
| Per Player | 0.605 | 0.655 | 0.498 |
| Set Attention | 0.676 | 0.749 | 0.503 |

Set Attentionはv3の打球F1を0.766から0.826に上げました。独立6本では球のみが僅差で上回り、推論時間はSet Attentionが約25%増えました。追加検証候補はSet Attentionとし、置換と全層fine-tuningは見送りました。球欠測フレームの採点対象は0件で、同条件の精度はN/Aです。

- [実装PR #2613](https://github.com/Motoki0705/tennis-lab/pull/2613) のHEADは `c4f8ed1e142ba1411e8978d4e318eccdd3b21e74`。CIが指摘した入力検証の配置を修正し、関連16件とpre-commitが通過しました。GitHub CIは進行中です。
- [knowledge PR #2621](https://github.com/Motoki0705/tennis-lab/pull/2621) に全指標、学習曲線、代表例、再現コマンドを記録しました。保存予測から再生成した集計・図は掲載ファイルとバイト一致し、knowledge CIも通過しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2613,"head":"c4f8ed1e142ba1411e8978d4e318eccdd3b21e74","resume_on_success":false}}