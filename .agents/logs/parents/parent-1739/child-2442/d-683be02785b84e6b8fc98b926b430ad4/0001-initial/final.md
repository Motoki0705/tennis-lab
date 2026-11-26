#2442 の抽出・注釈基盤を実装し、Terra の試行結果とともに提出しました。 [ドラフト PR #2448](https://github.com/Motoki0705/tennis-lab/pull/2448)

ブランチは `codex/parent-1739-issue-2442-primary`、HEAD は `7c553b73e0b2154b9e8c42074a566063711e7e45` です。固定入力12,184ファイルの SHA-256 を照合し、別の保存先から69 clipを再抽出して結果がバイト一致しました。pytest 233件、pre-commit、knowledgeの整合検査が通過しています。

| 層 | 候補 frame | 全体実行案 | 人の確認案 |
|---|---:|---:|---:|
| 選別外れ・コート近傍 | 39,152 | 399 | 30 |
| 1視点・3D欠測 | 4,247 | 71 | 28 |
| 未割当・再投影不一致 | 2,130 | 187 | 30 |
| ID switch・断片 | 12,281 | 350 | 30 |
| 停止 clip | 9,006 | 66 | 22 |
| 確信層の無作為標本 | 3,709 | 144 | 30 |
| 重複を除いた合計 | 46,101 | 1,205 | 170 |

**試行の一致率**は、Meiji 3cam の人手確定4 clip・48 frameで、box単位が **320/322（99.38%）**、人物ID単位が **279/281（99.29%）** でした。4cam は clip_020・032・025 の16 frame・486 boxを、旧IDを伏せて注釈し、固定datasetに保存しました。clip_032で旧IDを提示した試行との一致は178/184 boxで、採用分は画像と検出番号だけで作成しました。 [試行記録](https://github.com/Motoki0705/tennis-lab/blob/7c553b73e0b2154b9e8c42074a566063711e7e45/knowledge/runs/run-i2442-meiji-pseudo-association-pilot/README.md)

**時間と費用**は、3camで2.14秒・0.00372 USD相当[local path]、4camで9.23秒・0.00481 USD相当[local path]。1,205 frameへの外挿は **5.79〜9.14 USD相当**、CLI稼働の合計は **約3.09時間**、4並列を維持できた場合は約46分です。実請求額と残りquotaは取得できていません。 [OpenAI 標準レート](https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing)

親への確認事項は、**全体実行1,205 frameの範囲と予算、層別の抜き取り170 frame**です。4camの層別誤り率は人の確認後に記録します。CIはこのHEADを親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2448,"head":"7c553b73e0b2154b9e8c42074a566063711e7e45","resume_on_success":false}}