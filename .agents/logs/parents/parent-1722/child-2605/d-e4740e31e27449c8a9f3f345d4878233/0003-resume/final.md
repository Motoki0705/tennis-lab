`multiview-v3` をHEAD `59a62db3cd7593607c9bde0bfcc3e1c3f2c67d2e` で生成しました。全69クリップで元のv2配列がビット一致し、16,302ファイルをhard linkで共有しています。再実行でもファイルのSHA-256、913イベント、判定不能1,885フレーム、47/11/11の分割を照合できました。[manifest.json](https://github.com/Motoki0705/tennis-lab/blob/59a62db3cd7593607c9bde0bfcc3e1c3f2c67d2e/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3/manifest.json)

回帰テストは1,811件が通過しました。共有venvのPythonに `os.pidfd_open` がなく停止した2件を、同じHEAD・依存パッケージの作業用Python 3.11.17で再実行中です。その結果を受け取ってからPRを作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T053752Z_66d354fe0c1e_scene-events-v3-annotation-pidfd"]}}