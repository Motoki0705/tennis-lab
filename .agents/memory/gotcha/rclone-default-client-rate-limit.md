---
title: "rclone設定に client_id がないと、全rclone利用者の共用OAuth clientの上限に当たり、Driveの転送が 403 rateLimitExceeded で止まる"
applies_to: all
source: "#1585 のColab本実行（2026-10-31）"
created: 2026-10-31
last_verified: 2026-10-31
evidence: "Drive用rclone設定の `[gdrive]` に `client_id` がない。失敗ログの403は `consumer 'project_number:202264815644'`（rclone組み込みのclient）、`quota_limit: defaultPerMinutePerProject`"
---

repoのDrive用rclone設定は `client_id` を持たず、rcloneに組み込まれた共用のOAuth clientを使う。この共用projectの毎分の上限は世界中のrclone利用者で共有されるので、こちらの要求量と関係なく混雑時に `googleapi: Error 403: Quota exceeded for quota metric 'Queries' ... rateLimitExceeded` が続く。2026-10-31には、ColabでのpoolとDatasetの取得（約2,400ファイル・2.5GB）が1時間のtimeoutや一覧取得の失敗で3回止まった。同じ日に、Colab workerの周期保存の `moveto` が直前に作った一時ファイルを見つけられずにjobを止めた件（#1611で修正済み）も、同じ混雑が背景にあった可能性がある。

**使い方:** Driveの転送が `project_number:202264815644` の403で止まったら、コードの不具合ではなく共用clientの混雑とみなす。`drive.py download --overwrite` の再実行は取得済みファイルを飛ばして続きから進む。長時間のColab実行で繰り返すなら、repo専用のOAuth client IDをrclone設定に入れることをユーザーに提案する（Google Cloud Consoleでの作成と再認証が必要で、エージェントだけではできない）。
