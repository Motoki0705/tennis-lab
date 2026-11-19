planningを完了し、[設計・テスト・評価計画を #2191 に記録しました](https://github.com/Motoki0705/tennis-lab/issues/2191#issuecomment-6938028620)。

- **初期値**：保存済みH候補1694件を確認。参照作成clipを除外しても1577件あり、GPU推論なしで着手できます。
- **共通H**：複数frameの白線を測り、線の配置・終端・交点から対応を検証します。ネットも静止するため、時間集約だけでは識別できません。対応が曖昧なら明示的に失敗させます。
- **drift案**：3px超を警告、信頼できる局所Hとの変位が平均10px／最大25px超、または検査不能ならcamera/clipを失敗にする案です。数値は未確定です。
- **成果物・入口**：共通準備APIで動画成果物を固定し、camera別digestをcourt cacheとdataset完成markerの判定へ反映します。#1752の実CLI接続は同issue側との分担が必要です。
- **評価**：参照15frame/cameraを除外し、全69clipからfit用828画像・評価用276画像を選ぶ案。元録画frame番号でも重複0を確認しました。人工データ検証、既定clipの完全一致replay、全276組のoverlayも計画に含めています。

既存テストは**52 passed／14.74秒**。共有Hの精度・全camera成立・人間確認は未検証です。

branchは `codex/court-video-shared`、HEADは `be095af11`。追跡ファイル変更・commit・PR・push・CIなし。validatorは未指定、試行／完了0回です。planning指示に従い、実装開始、drift条件、評価subset、#1752との分担を親の判断事項として返します。