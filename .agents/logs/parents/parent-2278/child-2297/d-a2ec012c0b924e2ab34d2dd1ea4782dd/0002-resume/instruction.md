# #2297 続き: ncu 指標の追測

人間が 2026-11-20 に Windows 側 NVIDIA Control Panel で「すべてのユーザーに GPU パフォーマンスカウンターへのアクセスを許可」を設定した。

1. training queue 経由の小さな ncu 実行で `ERR_NVGPUCTRPERM` が解消したか確認する（上限30分）。まだ出る場合は、エラー出力を添えて「`wsl --shutdown` による再起動が要るか」を最終報告に書いて止める（OS操作はしない）。
2. 解消していれば、`--profile-from` で既存planを再利用し、CLIP-ReID 5条件すべての DRAM Read/Write Bytes・Memory Stall（warp stall理由）・Tensor Core利用率を PROTOCOL.md の範囲で計測し、metrics.json・validation.json・knowledge run ノードを更新する（上限3時間）。プロトコルを変える必要が出たら、変更点と理由を PROTOCOL.md と最終報告に書く。
3. 同じPR #2298 にcommit/pushし、対象HEADの `Python` CI成功を確認して、ドラフトを解除する。

最終報告: Acceptance 6項目の達成/未達と証拠、5条件の追加指標の表（DRAM R/W bytes・主要stall理由・Tensor Core利用率）と律速の解釈、PR HEAD。
