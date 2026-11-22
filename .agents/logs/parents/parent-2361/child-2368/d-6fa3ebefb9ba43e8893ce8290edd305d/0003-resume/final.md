[PR #2371](https://github.com/Motoki0705/tennis-lab/pull/2371) に、採用されたレビュー指摘2件を3言語で反映しました。

- **VHDXの復元元**。D: 上で縮小する場合の保守的な見積りは追加480.22 GiBで、実測の空き366.23 GiBを上回ります。E: を作業先とし、別の物理媒体にSHA-256照合済みのバックアップを確保して、D: に戻した後の検証まで保持する手順にしました。E: が使えない場合の復元手順も追加しました。
- **再起動をまたぐ停止**。対象unitの有効化・稼働状態を記録し、永続drop-inで起動を抑えます。Ubuntuを起動するたびに停止を確認し、検証後に元の状態へ戻します。[systemdの仕様](https://www.freedesktop.org/software/systemd/man/255/systemd.unit.html)

関連pytest 136件が通過しました。構文と模擬環境での動作を確認し、実機の確認は読み取りまでです。`codex/parent-2361-issue-2368-primary` に push 済みで、新しいHEADのCI結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2371,"head":"ab74b3a7e13e177815514ee74b2fd888eb9797d4","resume_on_success":false}}