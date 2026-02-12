"""
DEPRECATED: EmotionalIntelligence — заменена на VADEmotionalEngine (core/emotions_vad.py)
и EmotionAnalyzer (Rust ядро / bridge.py).

Этот файл оставлен для обратной совместимости.
"""

from core.emotions_vad import VADEmotionalEngine as EmotionalIntelligence

__all__ = ["EmotionalIntelligence"]
