"""
Кристина 6.0 — Python Fallback модули

Полноценные Python-реализации всех Rust-модулей из kristina_core.
Используются автоматически, если Rust ядро не скомпилировано.

API идентичен Rust-версиям (PyO3 интерфейс).
"""

from _fallback.memory_engine import MemoryEngine
from _fallback.embedding_cache import EmbeddingCache
from _fallback.emotion_analyzer import EmotionAnalyzer
from _fallback.tool_parser import ToolCallParser
from _fallback.context_compressor import ContextCompressor
from _fallback.thread_tracker import ThreadTracker

__all__ = [
    "MemoryEngine",
    "EmbeddingCache",
    "EmotionAnalyzer",
    "ToolCallParser",
    "ContextCompressor",
    "ThreadTracker",
]
