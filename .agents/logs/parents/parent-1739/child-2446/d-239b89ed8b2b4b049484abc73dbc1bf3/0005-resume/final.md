meiji_4camの全69本を読み戻し、新経路では全クリップに各側2人のGVHMR身体・有効な身体配置・ボール3Dがあることを確認しました。旧経路で同じ完了基準を満たしたのは52/69本です。

- 4人の身体配置が同時に有効な割合は旧64.63%、新90.74%。新経路で最も低い `clip_016` は198/1,214フレームでした。
- ボールの5配列は全69本で旧経路と一致しました。有効なのは45,958/46,228フレームで、欠測270フレームはすべて `clip_064` です。
- Meiji 3camの未見3本では、IDF1は旧92.07%・新80.42%、選別recallは旧85.30%・新67.25%でした。4camの未使用の対応視点への再投影残差も旧7.89 px・新24.86 pxでした。

[公開用比較ページ](https://github.com/Motoki0705/tennis-lab/blob/4d2773a05923916e84123f78a38da3665d826af4/outputs/pose-comparison-bodies/index.html)は、初回停止14本を含む16本・H.264動画32ファイル、全体63.27 MBです。段階2で有効な身体の輪郭と人数を表示し、Chromiumで段階切替後も同じクリップと再生位置が保たれることを確認しました。

- 実装HEAD `4d2773a05923916e84123f78a38da3665d826af4`。回帰・契約テスト14件が成功し、CI実行中です。 [Draft PR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470)
- 記録HEAD `34354185f48cea9ca3f9a3d2c8a422b84ebac4b5`。新旧69本の完了表、定量比較、動画一覧を収録し、Knowledge CIは成功しました。 [Draft PR #2493](https://github.com/Motoki0705/tennis-lab/pull/2493)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2470,"head":"4d2773a05923916e84123f78a38da3665d826af4","resume_on_success":false}}