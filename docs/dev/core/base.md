# Base モジュール概要 (`src/base`)

`src/base` には、推論器など他モジュールから再利用される**基盤インターフェース**が定義されています。
現在は主に、推論用の抽象基底クラス `BasePredictor` を提供します。

## 1. ディレクトリ構成

- `src/base/__init__.py`
  - パブリック API として `BasePredictor` をエクスポートします。
- `src/base/api/predictor.py`
  - 推論器の抽象基底クラス `BasePredictor` を定義します。

```text
src/base
├── __init__.py          # Base モジュールのエントリポイント
└── api
    └── predictor.py     # 推論器の抽象基底クラス
```

## 2. `BasePredictor` の役割

`BasePredictor` は、学習済みチェックポイントからモデルを読み込み、バッチ推論を行う**推論器インターフェース**です。

- 典型的な利用例:
  - `BLCS`, `PLCS` などのモジュール側で、このクラスを継承した具体的な Predictor を実装し、
    Web API やスクリプトから共通インターフェースで利用できるようにします。

### 2.1. 必須メソッド

すべての Predictor 実装は、以下 2 メソッドを実装する必要があります。

- `@classmethod load_from_checkpoint(cls, checkpoint_path, device="cpu", **kwargs) -> Self`
  - チェックポイントファイルからモデルをロードし、Predictor インスタンスを生成します。
  - `checkpoint_path`: チェックポイントファイルへのパス
  - `device`: 推論に使用するデバイス（`"cpu"` や `"cuda"`）
  - 追加のハイパーパラメータなどは `**kwargs` で受け取り、各実装で解釈します。
- `predict(self, *args, **kwargs) -> dict[str, Any]`
  - バッチ推論を実行します。
  - 入出力のフォーマットは各実装に委ねられますが、戻り値は**辞書形式**で統一されます。

## 3. 依存関係と設計ポリシー

- `BasePredictor` は PyTorch (`torch.nn.Module`, `torch.device`) に依存します。
- 具体的なモデルアーキテクチャや前処理/後処理の実装は、`blcs`, `plcs` 側のサブクラスで行います。
- 上位レイヤー（アプリケーション、API、CLI）は、
  - 具体クラスではなく `BasePredictor` 型に依存することで、
  - 実装入れ替え（例: 別モデルへの差し替え）を容易にします。

## 4. 新しい Predictor を追加する際のガイド

新しい推論器を追加する場合は、次のステップに従うことを推奨します。

1. **サブクラスを定義する**
   - 例: `class MyPredictor(BasePredictor): ...`
2. **`load_from_checkpoint` を実装する**
   - チェックポイントの読み込み、モデル構築、`device` への配置をここで行います。
   - ファイル存在チェックを行い、見つからない場合は `FileNotFoundError` を投げます。
3. **`predict` を実装する**
   - 前処理（入力の Tensor 化、正規化など）
   - モデルの forward 実行
   - 後処理（スコアの整形、辞書への詰め替え）
4. **公開インターフェースとして整理する**
   - 必要に応じて、`__all__` やエクスポート位置を整え、他モジュールから `from src.base import BasePredictor` の形で利用できるようにします。

このレイヤーは、学習済みモデルの推論 API を統一するための基盤として設計されており、
`src/blcs`, `src/plcs` などの上位モジュールから再利用されます。
