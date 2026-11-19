[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)のmajor 4件を修正しました。Ready、`MERGEABLE / CLEAN`です。

- 追跡特徴は完全なモデルidentity一致時だけ再利用し、不一致・不明なら再計算。
- 差し替えcomponentのModelSpecを成果物・完成markerへ反映。
- DINOの外部import元を検証し、宣言外の解決を拒否。
- 計測hookが例外を抑制しても、初期化・解放の失敗を伝播。

branch: `codex/scene-inference-runtime`  
HEAD: `caa0a01a3464670d1f8b683ad808c1fca81543bd`

CPU関連710件、pre-commit、対象HEADの[CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44323317602)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44323317926)は全成功。A〜Eの確認表とcacheへの影響をPRへ記録しました。

旧追跡v5・選別v2は再生成が必要です。GPU再実行は指示どおり行っていません。merge・closeは親へ返します。