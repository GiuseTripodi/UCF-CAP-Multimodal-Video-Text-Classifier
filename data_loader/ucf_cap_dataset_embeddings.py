"""Legacy compatibility shim for the active UCF-CAP dataset loader.

The project now uses :mod:`data_loader.ucf_cap_loader` as the single source of
truth. This module stays only so older imports keep working.
"""

from data_loader.ucf_cap_loader import UCF101Dataset, collate_video_batch

__all__ = ["UCF101Dataset", "collate_video_batch"]
