"""Adapters from source-specific motion representations to COCO-17."""

from src.tasks.plcs.motion.sources.accad import AccadCoco17Adapter
from src.tasks.plcs.motion.sources.gvhmr import GvhmrCoco17Adapter

__all__ = ["AccadCoco17Adapter", "GvhmrCoco17Adapter"]
