"""Tennis-domain self-supervised fine-tuning of DINOv3 with LoRA.

This task fine-tunes the vendored DINOv3 ViT backbone on web-derived tennis
imagery using DINOv3's own self-distillation strategy (DINO + iBOT + KoLeo
losses with an EMA teacher). Training is kept lightweight with LoRA adapters so
the pretrained backbone's general capability is preserved while it adapts to the
tennis domain for downstream ``court_detection`` / ``ball_detection`` use.

Sub-packages:
    - ``generate_dataset``: web-derived image collection pipeline.
    - ``data``: multi-crop SSL dataset / datamodule.
    - ``models``: DINOv3 backbone loading + LoRA wrapping + SSL network.
    - ``training``: Lightning module and training runner.
    - ``scripts``: Hydra entrypoints for collection and training.
"""
