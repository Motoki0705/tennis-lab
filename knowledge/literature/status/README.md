# Daily status

`YYYY-MM-DD.json` は、当日の初期化状態とGitHub Actionsが集計した受理件数・collector quota・topic quotaを保持します。

毎時collectorは外部探索を始める前にこのファイルを読み、quota到達済みなら即座に `NO_CHANGE` で終了します。日次curatorは `initialized_at` を初期化の冪等性markerとして使用し、同日の2回目以降の実行では3本のqueue branchをresetしません。

このファイルは制御・集計用であり、論文知識そのものではありません。
