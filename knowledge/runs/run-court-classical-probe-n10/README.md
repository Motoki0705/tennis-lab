# 古典コートモデル補助プローブ

`gchlebus/tennis-court-detection` の固定 commit
`762d077541a77abf4923f5f8f689a1410927d35e` を、Ubuntu 22.04 と
CPU 版 OpenCV 4.5.4 で実行した補助的な robustness probe である。
主比較の manifest とは別の実写10枚・B00合成test 10枚なので、主比較表へ
数値を混ぜない。

`REPORT.md` と `sweep_results.json` が実測の権威である。`figures/` は代表的な
成功、誤fit、縮退fit、外挿例を保存する。`tcd_opencv4.patch` は OpenCV 4
互換化と検証用overlay出力だけを追加し、検出閾値は変更しない。

再実行には Git、Docker、FFmpeg、および tennis-lab のデータが必要である。

```bash
bash knowledge/runs/run-court-classical-probe-n10/repro.sh
```

スクリプトは `mktemp` で新しい作業ディレクトリを作る。上流ソースやビルド
生成物をこのリポジトリへコピーせず、終了時に成果物の場所を表示する。
