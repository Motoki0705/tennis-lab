"""Ball detection package with pretrain + pseudo-label self-training workflow."""

from src.ball_detection.pseudo.orchestrator import PseudoLabelOrchestrator

__all__ = ["PseudoLabelOrchestrator"]
