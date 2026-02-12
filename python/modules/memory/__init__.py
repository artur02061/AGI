"""
Кристина 6.0 — Модуль памяти

Компоненты:
- ThreadTracker: отслеживание нитей разговора
- MemorySummarizer: иерархическая суммаризация (daily/weekly/monthly)
- KnowledgeGraph: семантическая память через тройки (subject→predicate→object)
"""

from modules.memory.thread_tracker import ThreadTracker

# Ленивый импорт тяжёлых компонентов
def get_summarizer():
    from modules.memory.summarizer import MemorySummarizer
    return MemorySummarizer()

def get_knowledge_graph():
    from modules.memory.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()
