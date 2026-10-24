# 凍結D2候補のL4試行：GPU起動前の失敗

4条件×4scene×12枚を変更せず、指定されたpush済みcommitで1回だけ試行した。
最初の12カメラを台帳に加えた直後、相対一時pathがNHT requestの絶対path契約に違反した。
NHT呼び出し0、完成画像0、QA0。再描画せず、全jobのDrive保存確認後にColabを停止した。
結果と再実行前の判断事項は [comparison.md](comparison.md) を参照する。

実行版・session・出力rootは [run.json](run.json)、argvは [job-requests.json](job-requests.json)、
応答は evidence/ のreceiptに記録した。実際に提出したjobはsetupとrender-probeだけである。
同JSONのcollect-reviewは準備した未提出のrequestで、collect_probe.pyもuploadだけで実行していない。
session名の大文字を含む最初のdry-runは割当前に拒否され、小文字の新規名で開始した。

## 実行時の追加計測

colab_setup.shは既存の標準ライブラリ計測scriptを使い、uv、render-only NHT、入力取得、
manifest照合を別々に計測した。稼働中のログをpersist対象にコピーしていない。
colab_render.shは凍結scriptをPython標準cProfileから1回起動した。
profileは例外時にも保存された。GPU前の失敗なので、描画・QA・画像保存のthroughputは得られていない。
親processのprofileはNHT子processの内部計測ではなく、今回その子process自体が起動していない。

## 回収

新規の実験出力rootだけをDrive CLIでdownload/verifyした。3.37MB・8ファイルで、モデルや画像配列は含まない。
運用記録のrender-probe subtreeも別rootへ検証付きで回収した。正確なコマンドはdownload receiptにある。

- tmp/court-d2-probe-l4/failure-recovered：状態・setup時間・失敗process時間・profile
- tmp/court-d2-probe-l4/render-operation：Driveに保存されたstatus・stdout・source差分

Gitには小さいJSONと失敗ログを保存し、profile本体は元のDrive出力とローカル回収先に保持した。
対応は [recovered-files.json](evidence/recovered-files.json)。元の選択JSONは凍結版との完全一致を確認した。
stopが回収したtracked patchは空、未追跡sourceは準備用collectorだけでローカルと一致した。

## CPUだけの再現

```bash
PYTHONPATH="$PWD" bash knowledge/runs/run-court-d2-probe-l4-20261024/repro.sh \
  tmp/court-d2-probe-l4/failure-recovered \
  tmp/court-d2-probe-l4/render-operation
```

このscriptはColab開始・描画・学習を行わない。失敗集計とsetupだけ更新した費用JSONを再計算してbyte照合し、
相対pathの拒否、outputだけ絶対pathにした場合のrequest pathの拒否、一時root全体の絶対path化による
request構築成功をCPUで確認する。GPU成功を検証したものではない。
費用計算の準備段階で実施した旧入力による300項目の数値一致も再現する。

## 未実行の準備コード

analyze_probe.py・update_costs.py・make_examples.pyは、192枚が成功した場合の解析用に準備し、静的検査だけを行った。
update_costs.pyは旧入力fixtureの検査だけで、新しいQA/JPEGからの費用計算は実行していない。
原画・overlay・動画・代表図、採択率は存在しない。選択192行のgeometryは前回CPU planに由来し、描画結果ではない。
これらを未生成の解析結果や本番データ版として扱わない。
