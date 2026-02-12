"""
Кристина 6.0 — Система памяти

ИЗМЕНЕНИЯ v6.0:
- Интеграция с MemorySummarizer (иерархическая суммаризация)
- Интеграция с KnowledgeGraph (семантическая память)
- Поиск по всем уровням: raw → daily → weekly → monthly → knowledge graph
- get_relevant_context ищет ВСЕ эпизоды (не только последние 50)
- Auto-summarize при save()
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.logging import get_logger
import config

logger = get_logger("memory")


class MemorySystem:
    """Управление памятью с иерархической суммаризацией"""

    def __init__(self):
        self.working = []
        self.episodic = []
        self.semantic = {}

        # Файлы
        self.episodic_file = config.EPISODIC_MEMORY_FILE
        self.semantic_file = config.SEMANTIC_MEMORY_FILE

        # v6.0: Суммаризатор и Knowledge Graph (ленивая инициализация)
        self._summarizer = None
        self._knowledge_graph = None

        self.load()

        logger.info(f"Память: {len(self.episodic)} эпизодов")

    # ═══════════════════════════════════════════════════════════
    #          ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ ТЯЖЁЛЫХ КОМПОНЕНТОВ
    # ═══════════════════════════════════════════════════════════

    @property
    def summarizer(self):
        """Ленивая инициализация суммаризатора"""
        if self._summarizer is None:
            try:
                from modules.memory.summarizer import MemorySummarizer
                self._summarizer = MemorySummarizer()
            except Exception as e:
                logger.warning(f"⚠️ Суммаризатор недоступен: {e}")
        return self._summarizer

    @property
    def knowledge_graph(self):
        """Ленивая инициализация Knowledge Graph"""
        if self._knowledge_graph is None:
            try:
                from modules.memory.knowledge_graph import KnowledgeGraph
                self._knowledge_graph = KnowledgeGraph()
            except Exception as e:
                logger.warning(f"⚠️ Knowledge Graph недоступен: {e}")
        return self._knowledge_graph

    # ═══════════════════════════════════════════════════════════
    #                    РАБОЧАЯ ПАМЯТЬ
    # ═══════════════════════════════════════════════════════════

    def add_to_working(self, role: str, content: str):
        """Добавляет в рабочую память"""
        self.working.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        if len(self.working) > config.WORKING_MEMORY_SIZE:
            self.working.pop(0)

    # ═══════════════════════════════════════════════════════════
    #                  ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ
    # ═══════════════════════════════════════════════════════════

    def add_episode(
        self,
        user_input: str,
        response: str,
        emotion: str,
        importance: int = 1
    ):
        """Добавляет эпизод"""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "emotion": emotion,
            "importance": importance
        }

        self.episodic.append(episode)

        # v6.0: Извлекаем факты в Knowledge Graph (асинхронно, fire-and-forget)
        if self.knowledge_graph and importance >= config.config.knowledge_graph_min_importance:
            try:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(
                        self.knowledge_graph.extract_and_add(user_input, response, importance)
                    )
                except RuntimeError:
                    # No running loop — sync fallback
                    triples = self.knowledge_graph._regex_extract(user_input)
                    for s, p, o in triples:
                        self.knowledge_graph.add_triple(s, p, o, importance, "regex")
            except Exception as e:
                logger.debug(f"KG extraction skipped: {e}")

        # Ротация
        if len(self.episodic) > config.MAX_EPISODIC_MEMORY:
            self.episodic.sort(key=lambda x: x['importance'])
            self.episodic = self.episodic[100:]

    # ═══════════════════════════════════════════════════════════
    #              ПОИСК КОНТЕКСТА (МНОГОУРОВНЕВЫЙ)
    # ═══════════════════════════════════════════════════════════

    def get_relevant_context(self, query: str, max_items: int = 3) -> str:
        """
        v6.0: Многоуровневый поиск контекста.

        1. Raw episodes (все, не только последние 50!)
        2. Knowledge Graph факты
        3. Summary саммари (daily/weekly/monthly)
        """
        parts = []

        # ── 1. Raw episodes (keyword search по ВСЕМ) ──
        episode_results = self._search_episodes(query, max_items)
        if episode_results:
            for score, episode in episode_results:
                date = episode['timestamp'][:10]
                user_preview = episode['user_input'][:80]
                parts.append(f"[{date}] Пользователь: {user_preview}")

        # ── 2. Knowledge Graph ──
        if self.knowledge_graph:
            kg_context = self.knowledge_graph.get_context_for_query(query, max_facts=3)
            if kg_context:
                parts.append(kg_context)

        # ── 3. Summary search (если raw не дал достаточно) ──
        if len(parts) < max_items and self.summarizer:
            summary_results = self.summarizer.search_summaries(query, max_results=2)
            for r in summary_results:
                level = r["level"]
                key = r["key"]
                summary = r["summary"][:150]
                parts.append(f"[{level} {key}] {summary}")

        if not parts:
            return "Нет релевантного контекста."

        return "\n".join(parts)

    def _search_episodes(self, query: str, max_items: int = 3) -> List:
        """Keyword search по ВСЕМ эпизодам (не только последним 50)"""
        if not self.episodic:
            return []

        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        if not query_words:
            return []

        scored = []
        for episode in self.episodic:
            text = episode['user_input'] + " " + episode['response']
            text_words = set(w.lower() for w in text.split() if len(w) > 2)

            intersection = len(query_words & text_words)
            if intersection > 0:
                score = intersection * episode.get('importance', 1)
                scored.append((score, episode))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:max_items]

    # ═══════════════════════════════════════════════════════════
    #                  ПЕРСИСТЕНТНОСТЬ
    # ═══════════════════════════════════════════════════════════

    def save(self):
        """Сохраняет память + запускает auto-summarize"""
        try:
            # Эпизодическая
            with open(self.episodic_file, 'w', encoding='utf-8') as f:
                json.dump(self.episodic, f, ensure_ascii=False, indent=2)

            # Семантическая
            with open(self.semantic_file, 'w', encoding='utf-8') as f:
                json.dump(self.semantic, f, ensure_ascii=False, indent=2)

            # v6.0: Knowledge Graph
            if self.knowledge_graph:
                self.knowledge_graph.save()

            # v6.0: Auto-summarize
            if self.summarizer and config.config.memory_summarize_enabled:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._auto_summarize())
                except RuntimeError:
                    # Нет event loop — пропускаем
                    pass

            logger.info("💾 Память сохранена")

        except Exception as e:
            logger.error(f"Ошибка сохранения памяти: {e}")

    async def _auto_summarize(self):
        """Автоматическая суммаризация старых эпизодов"""
        if self.summarizer:
            try:
                self.episodic = await self.summarizer.auto_summarize(self.episodic)
            except Exception as e:
                logger.error(f"Ошибка auto-summarize: {e}")

    def load(self):
        """Загружает память"""
        try:
            if self.episodic_file.exists():
                with open(self.episodic_file, 'r', encoding='utf-8') as f:
                    self.episodic = json.load(f)

            if self.semantic_file.exists():
                with open(self.semantic_file, 'r', encoding='utf-8') as f:
                    self.semantic = json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки памяти: {e}")

    def clear_working(self):
        """Очищает рабочую память"""
        self.working.clear()
        logger.info("Рабочая память очищена")

    def get_stats(self) -> Dict[str, int]:
        """Статистика памяти"""
        stats = {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic_keys": len(self.semantic),
        }

        if self.summarizer:
            stats["summaries"] = self.summarizer.get_stats()

        if self.knowledge_graph:
            stats["knowledge_graph"] = self.knowledge_graph.get_stats()

        return stats
