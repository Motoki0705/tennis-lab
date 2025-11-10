# チケット: SceneModel ビルド拡張（dinov3 ViT-S 統合/ヘッド切替）

## 背景/目的
`third_party/dinov3` の ViT-S 重みを backone に流用し、`cfg.head.type` による既存 head と bbox head の切替に対応する。

## 対象
- `src/models/scene_model/build.py`

## タスク
- [ ] `torch.hub.load("third_party/dinov3", 'dinov3_vits16', source='local', weights=...)` で事前学習重みをロード。
- [ ] 位置埋め込みの補間（img_size 不一致時）を実装し、安全に注入。
- [ ] `cfg.model.backbone.freeze`/`unfreeze_after` などの凍結ポリシーを実装。
- [ ] `cfg.head.type` に応じて bbox head or 既存 3D head を組み立て。
- [ ] Denoiser 用の hook/adapter が利用できるよう I/F を整備。
- [ ] ロード・切替の最小テスト（モック）を作成。

## 受け入れ基準（DoD）
- [ ] dinov3 の重み注入で例外が発生しない（モック/存在チェック）。
- [ ] `cfg.head.type` の切替でモデルが構築できる。
