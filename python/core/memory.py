"""
Кристина 6.0 — Система памяти

Основная реализация: MemoryAdapter из bridge.py (обёртка над MemoryEngine).
Этот модуль оставлен для общих утилит памяти.

ИЗМЕНЕНИЯ v6.0:
- MemorySystem удалён (был мёртвый код, никогда не инстанцировался)
- Основная реализация — MemoryAdapter в bridge.py
"""

from utils.logging import get_logger

logger = get_logger("memory")
