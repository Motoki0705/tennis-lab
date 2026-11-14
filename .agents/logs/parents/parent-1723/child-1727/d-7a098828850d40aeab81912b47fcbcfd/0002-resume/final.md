[PR #1759](https://github.com/Motoki0705/tennis-lab/pull/1759)をReady・衝突なしにしました。

- branch: `codex/player-view-blender-video`
- HEAD: `711bef13f7c21fd028ef2354883e9dff4acd9121`
- mainを統合し、両方のruntime登録を保持。関連15テスト・pre-commit成功。
- [HEAD CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42547179992/attempts/2)・[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42547179968)成功。既存e2eの競合1件は同じHEADで再実行しました。

本文を`Refs #1727`に変更し、人間の判断と#1776での再受入待ちを追記済みです。人間受入は未了のままです。再描画・Colab使用はしていません。