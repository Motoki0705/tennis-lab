# 2026-11-10 追加実験の検証設計

固定されたmodel/training実装は旧planの162 source hashesと一致させる。実験制御はこのrun内だけに置く。

- 条件表: 9モデルの順序、既存7件はoffset6000の6k、新規12/24層はoffset0/6000の2段、合計11段/66kを検査する。
- 実queue登録: CPUの一時queueで同じ投入コマンドを再送し、9件だけが存在し同じIDを返すこと、CWD/resource all/順序を検査する。
- 失敗停止: 制御が例外を記録して共通queue stopを先に作り、失敗後に後続学習を開始しない。前件の完了証拠が無いジョブも拒否する。
- 境界継承: coreのstrict model/optimizer継承に加え、実full checkpointで全named AdamW状態を比較する。段境界のLR照合、sample位置、同構成継続のPython/NumPy/Torch/CUDA RNG復元を記録する。DPT→新queryのhead/RNG初期化は旧レシピを維持する。
- 実L4受入: 第一モデル601更新以上、全loss/grad/LR finite、CNN frozenの600/601境界、連続sample/global更新を検査する。500更新checkpoint・ログのDrive receipt→marker→全payload→checkpoint内容を独立readbackで検証する。
- 永続化: 外部serviceの祖先が有限worker内でないこと、単一GPU all予約、残り8queued、成果物/ソースbundle/復旧手順のDrive保存を確認する。
