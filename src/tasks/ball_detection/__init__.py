"""Ball detection package with pretrain + pseudo-label self-training workflow."""

from src.tasks.ball_detection.pseudo.orchestrator import PseudoLabelOrchestrator

__all__ = ["PseudoLabelOrchestrator"]
