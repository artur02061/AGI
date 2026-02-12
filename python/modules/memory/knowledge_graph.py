"""
Кристина 6.0 — Knowledge Graph (Граф Знаний)

Семантическая память на тройках (subject, predicate, object):
  "пользователь" → "любит" → "Python"
  "проект X" → "написан_на" → "Rust"
  "кот пользователя" → "зовут" → "Мурзик"

Хранение: NetworkX DiGraph + JSON персистентность.
Извлечение фактов: LLM (gemma3:4b) или regex-паттерны.

Поиск: по сущности, по предикату, по окрестности (BFS).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

# ollama — используется через AsyncClient внутри _llm_extract

from utils.logging import get_logger
import config

logger = get_logger("knowledge_graph")

# NetworkX — опционально (для продвинутого graph traversal)
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("⚠️ networkx не установлен, используем встроенный граф")


class KnowledgeGraph:
    """Граф знаний — семантическая память через тройки"""

    def __init__(self):
        self._graph_file = config.config.knowledge_graph_dir / "graph.json"
        self._model = config.config.knowledge_graph_extractor_model
        self._max_nodes = config.config.knowledge_graph_max_nodes
        self._max_edges = config.config.knowledge_graph_max_edges

        # Граф
        if HAS_NETWORKX:
            self._graph = nx.DiGraph()
        else:
            self._graph = None

        # Простое хранилище (всегда доступно, даже без networkx)
        self._triples: List[Dict] = []  # [{s, p, o, ts, importance}, ...]
        self._entity_index: Dict[str, Set[int]] = defaultdict(set)  # entity → triple indices
        self._triple_keys: Set[tuple] = set()  # O(1) duplicate check: (s, p, o)

        self._load()

        logger.info(
            f"🧠 Knowledge Graph: {len(self._triples)} фактов, "
            f"networkx={'да' if HAS_NETWORKX else 'нет'}"
        )

    # ═══════════════════════════════════════════════════════════
    #                    ДОБАВЛЕНИЕ ФАКТОВ
    # ═══════════════════════════════════════════════════════════

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        importance: int = 1,
        source: str = "auto",
    ):
        """Добавляет тройку в граф"""
        subject = subject.strip().lower()
        predicate = predicate.strip().lower()
        obj = obj.strip().lower()

        if not subject or not predicate or not obj:
            return

        # O(1) проверка дубликата через set
        triple_key = (subject, predicate, obj)
        if triple_key in self._triple_keys:
            # Обновляем важность и timestamp у существующей тройки
            for t in self._triples:
                if t["s"] == subject and t["p"] == predicate and t["o"] == obj:
                    t["importance"] = max(t["importance"], importance)
                    t["ts"] = datetime.now(timezone.utc).isoformat()
                    break
            return

        # Лимит
        if len(self._triples) >= self._max_edges:
            self._evict_least_important()

        idx = len(self._triples)
        triple = {
            "s": subject,
            "p": predicate,
            "o": obj,
            "ts": datetime.now(timezone.utc).isoformat(),
            "importance": importance,
            "source": source,
        }
        self._triples.append(triple)
        self._triple_keys.add(triple_key)

        # Индексы
        self._entity_index[subject].add(idx)
        self._entity_index[obj].add(idx)

        # NetworkX
        if self._graph is not None:
            self._graph.add_edge(subject, obj, predicate=predicate, importance=importance)

        logger.debug(f"➕ Факт: {subject} → {predicate} → {obj}")

    async def extract_and_add(self, user_input: str, response: str, importance: int = 1):
        """
        Извлекает факты из диалога через LLM и добавляет в граф.
        Вызывается при каждом save_to_memory.
        """
        if not config.config.knowledge_graph_enabled:
            return

        if importance < config.config.knowledge_graph_min_importance:
            return

        # Сначала пробуем regex-извлечение (быстро, бесплатно)
        regex_triples = self._regex_extract(user_input)
        for s, p, o in regex_triples:
            self.add_triple(s, p, o, importance, source="regex")

        # Если есть что-то содержательное — LLM извлечение
        if len(user_input) > 30 and importance >= 2:
            llm_triples = await self._llm_extract(user_input, response)
            for s, p, o in llm_triples:
                self.add_triple(s, p, o, importance, source="llm")

    # ═══════════════════════════════════════════════════════════
    #                    ПОИСК
    # ═══════════════════════════════════════════════════════════

    def query_entity(self, entity: str, max_results: int = 10) -> List[Dict]:
        """Все факты, связанные с сущностью"""
        entity = entity.strip().lower()
        results = []

        indices = self._entity_index.get(entity, set())
        for idx in indices:
            if idx < len(self._triples):
                results.append(self._triples[idx])

        # Также ищем частичное совпадение
        for key, idx_set in self._entity_index.items():
            if entity in key and key != entity:
                for idx in idx_set:
                    if idx < len(self._triples) and self._triples[idx] not in results:
                        results.append(self._triples[idx])

        results.sort(key=lambda x: x["importance"], reverse=True)
        return results[:max_results]

    def query_relation(self, subject: str = None, predicate: str = None, obj: str = None) -> List[Dict]:
        """Поиск по шаблону (s, p, o) — None = любое значение"""
        results = []
        for t in self._triples:
            if subject and t["s"] != subject.lower():
                continue
            if predicate and t["p"] != predicate.lower():
                continue
            if obj and t["o"] != obj.lower():
                continue
            results.append(t)
        return results

    def get_context_for_query(self, query: str, max_facts: int = 5) -> str:
        """
        Извлекает релевантные факты для запроса.
        Возвращает текст для включения в промпт.
        """
        query_words = set(query.lower().split())

        scored_triples = []
        for t in self._triples:
            text = f"{t['s']} {t['p']} {t['o']}"
            text_words = set(text.split())
            score = len(query_words & text_words) * t["importance"]
            if score > 0:
                scored_triples.append((score, t))

        scored_triples.sort(key=lambda x: x[0], reverse=True)

        if not scored_triples:
            return ""

        facts = []
        for _, t in scored_triples[:max_facts]:
            facts.append(f"• {t['s']} {t['p']} {t['o']}")

        return "Известные факты:\n" + "\n".join(facts)

    def get_neighbors(self, entity: str, depth: int = 1) -> Dict[str, List[Dict]]:
        """BFS по графу — находит соседей на глубину depth"""
        entity = entity.strip().lower()

        if self._graph is not None and entity in self._graph:
            # NetworkX BFS
            neighbors = {}
            for d in range(1, depth + 1):
                level_nodes = set()
                try:
                    for node in nx.single_source_shortest_path_length(self._graph, entity, cutoff=d):
                        level_nodes.add(node)
                except nx.NetworkXError:
                    pass
                neighbors[f"depth_{d}"] = self.query_entity(entity)
            return neighbors

        # Fallback: простой поиск
        return {"depth_1": self.query_entity(entity)}

    # ═══════════════════════════════════════════════════════════
    #                    ИЗВЛЕЧЕНИЕ ФАКТОВ
    # ═══════════════════════════════════════════════════════════

    def _regex_extract(self, text: str) -> List[Tuple[str, str, str]]:
        """Извлечение фактов regex-паттернами (быстро)"""
        triples = []

        patterns = [
            # "меня зовут Имя"
            (r"меня зовут\s+(\w+)", "пользователь", "зовут", 1),
            # "я работаю в Компания" / "я работаю программистом"
            (r"я работаю\s+(?:в\s+)?(.+?)(?:\.|,|$)", "пользователь", "работает", 1),
            # "мой кот/пёс/... зовут Имя"
            (r"мо[йяё]\s+(\w+)\s+зовут\s+(\w+)", None, "зовут", None),
            # "я люблю X" / "я обожаю X"
            (r"я (?:люблю|обожаю)\s+(.+?)(?:\.|,|!|$)", "пользователь", "любит", 1),
            # "я живу в Город"
            (r"я живу\s+в\s+(.+?)(?:\.|,|$)", "пользователь", "живёт_в", 1),
            # "мне N лет"
            (r"мне\s+(\d+)\s+(?:лет|год|года)", "пользователь", "возраст", 1),
        ]

        for pattern_info in patterns:
            if len(pattern_info) == 4:
                pattern, subj, pred, obj_group = pattern_info
                match = re.search(pattern, text.lower())
                if match:
                    if subj is None:
                        # Специальный паттерн с 2 группами
                        if match.lastindex and match.lastindex >= 2:
                            triples.append((match.group(1), pred, match.group(2)))
                    else:
                        obj = match.group(obj_group).strip()
                        if obj:
                            triples.append((subj, pred, obj))

        return triples

    async def _llm_extract(self, user_input: str, response: str) -> List[Tuple[str, str, str]]:
        """Извлечение фактов через LLM (async)"""
        prompt = f"""Извлеки факты из диалога в формате JSON массива троек.
Каждая тройка: {{"s": "субъект", "p": "предикат", "o": "объект"}}

Примеры:
  "Я работаю программистом в Google" → [{{"s": "пользователь", "p": "работает", "o": "программист"}}, {{"s": "пользователь", "p": "работает_в", "o": "google"}}]
  "Мой проект написан на Rust" → [{{"s": "проект пользователя", "p": "написан_на", "o": "rust"}}]

Если фактов нет — верни пустой массив [].

Пользователь: {user_input[:200]}
Ответ: {response[:200]}

JSON:"""

        try:
            from ollama import AsyncClient
            client = AsyncClient()
            resp = await client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 300},
            )

            text = resp["message"]["content"].strip()

            # Парсим JSON
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                triples = []
                for item in data:
                    if isinstance(item, dict) and "s" in item and "p" in item and "o" in item:
                        triples.append((str(item["s"]), str(item["p"]), str(item["o"])))
                return triples

        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")

        return []

    # ═══════════════════════════════════════════════════════════
    #                    ПЕРСИСТЕНТНОСТЬ
    # ═══════════════════════════════════════════════════════════

    def save(self):
        try:
            with open(self._graph_file, "w", encoding="utf-8") as f:
                json.dump(self._triples, f, ensure_ascii=False, indent=2)
            logger.debug(f"💾 Knowledge Graph: {len(self._triples)} фактов")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения графа: {e}")

    def _load(self):
        if not self._graph_file.exists():
            return
        try:
            with open(self._graph_file, "r", encoding="utf-8") as f:
                self._triples = json.load(f)

            # Перестраиваем индексы и set ключей
            self._entity_index.clear()
            self._triple_keys.clear()
            for i, t in enumerate(self._triples):
                self._entity_index[t["s"]].add(i)
                self._entity_index[t["o"]].add(i)
                self._triple_keys.add((t["s"], t["p"], t["o"]))

            # NetworkX
            if self._graph is not None:
                self._graph.clear()
                for t in self._triples:
                    self._graph.add_edge(
                        t["s"], t["o"],
                        predicate=t["p"],
                        importance=t.get("importance", 1),
                    )

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки графа: {e}")
            self._triples = []

    def _evict_least_important(self):
        """Удаляет 10% наименее важных троек"""
        count = max(1, len(self._triples) // 10)
        sorted_triples = sorted(
            enumerate(self._triples),
            key=lambda x: x[1].get("importance", 0),
        )
        indices = sorted([i for i, _ in sorted_triples[:count]], reverse=True)
        for idx in indices:
            t = self._triples.pop(idx)
            self._triple_keys.discard((t["s"], t["p"], t["o"]))

        # Перестраиваем индексы
        self._entity_index.clear()
        for i, t in enumerate(self._triples):
            self._entity_index[t["s"]].add(i)
            self._entity_index[t["o"]].add(i)

    def get_stats(self) -> Dict:
        entities = set()
        for t in self._triples:
            entities.add(t["s"])
            entities.add(t["o"])

        predicates = set(t["p"] for t in self._triples)

        return {
            "triples": len(self._triples),
            "entities": len(entities),
            "predicates": len(predicates),
            "has_networkx": HAS_NETWORKX,
        }
