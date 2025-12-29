# test_generate_dataset.py (WASB)

データセット生成スクリプトのテストです。

**ファイル**: `tests/e2e/wasb/test_generate_dataset.py`

## 概要

WASBのデータセット生成は複雑な依存関係があるため、多くのテストがスキップされています。

---

## test_wasb_download_videos_status

ビデオダウンロードスクリプトのステータスモードテスト。

### 状態

**スキップ**: meta.json のセットアップが必要

### 対象スクリプト

`src.wasb.scripts.generate_dataset.download_videos`

### 目的

ダウンロードを実行せずにステータス確認モードが動作することを確認。

---

## test_wasb_generate_dataset_batch_mode

バッチモードでのデータセット生成テスト。

### 状態

**スキップ**: ビデオ処理が複雑

### スキップ理由

バッチ処理には以下が必要：
- 実際のビデオファイル
- モデル推論（ボール検出、コート検出等）
- 長い処理時間

---

## test_wasb_clip_sampling_generate_samples

クリップサンプリングのサンプル生成テスト。

### 状態

**スキップ**: 既存データセット構造が必要

### スキップ理由

クリップサンプリングには以下が必要：
- 処理済みクリップを含むデータセット構造
- 複雑なセットアップ

---

## 代替テスト方法

WASBのデータセット生成をテストする場合は、以下を推奨：

1. `create_minimal_wasb_dataset` フィクスチャを使用して合成データを生成
2. データローダーの動作を `test_data_validation.py` で検証
3. 学習スクリプトのドライランモードを使用
