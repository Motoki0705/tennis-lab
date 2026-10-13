# 本学習前に消失した準備用VM

`i983-l4-ablation-06` はL4と科学的commit `c4b12f923b243da6b502149aa6f3912c706045e7` を確認し、環境と入力を準備した。Bの本学習を開始した証跡はないため、完了した学習やcheckpoint再開として扱わない。

- setupは2026-10-12T08:47:58Zにreturncode 0・Drive保存で終了した。
- 最初の入力準備はNotebook接続エラー後もremoteがpendingのままだった。09:14:52Zにcancelled・returncode 130・Drive保存を確認した。cancel markerを残し、所有を確認したローカルtransportだけを終了した。要求と取消receiptは `input-attempts/` に残す。
- `prepare-inputs-attempt02` は09:23:05Z〜09:30:41Zに実行し、5 datasetとB pretrained weightの検証・Drive保存をreturncode 0で完了した。
- 09:40:49Zに本番環境で取り直したDrive認証情報を反映し、接続を確認した。資格情報自体はこのbundleに含めない。
- 後続の学習提出はSSH 255で失敗し、再接続時に公式registryからsessionが消えていることを確認した。割当数0とB学習出力のDrive上の未作成も確認した。VM消失の原因は確定していない。

後継session07では環境準備・入力検証・本学習を1つのNotebook kernel jobで連続実行する。科学的条件は維持し、このVMの準備完了を後継VMの入力検証の代わりにしない。
