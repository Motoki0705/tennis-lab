"""Unsplit checklist shapes recorded in issue #1813 before #1796 was split."""

WITH_DESCRIPTION = (
    "未分割（子issue未作成）。起票段階では実装担当や仮の子番号を作らない。\n\n"
    "分割候補は、記録契約、既存入口との接続、Git公開とREADME表示。"
)

WITH_CANDIDATES = (
    "未分割（子issue未作成）。\n\n"
    "分割候補は、①結果・CI・runtime・公開の分離と復旧、"
    "②再利用判定とresolve／gateへの接続、③レビュー用モデル選択経路、"
    "④比較評価・#1796連携・実機受入。"
    "共有ファイルの競合と既存成果を見て担当親が具体化する。"
)
