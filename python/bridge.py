"""
Кристина 6.0 — Мост между Rust и Python (ИСПРАВЛЕНО)

ИСПРАВЛЕНИЯ:
- ✅ MemoryAdapter — единый API для Rust и Python MemoryEngine
- ✅ get_stats() возвращает dict (не кортеж)
- ✅ get_relevant_context() возвращает строку (не кортежи)
- ✅ .working доступен как list of dicts
- ✅ Интеграция с v6.0 (суммаризатор, KG) через адаптер
- ✅ EmbeddingCacheAdapter — нормализованный API
- ✅ ContextCompressorAdapter
- ✅ ThreadTrackerAdapter — переработан
- ✅ cosine_similarity с numpy ускорением
"""

import logging

logger = logging.getLogger("kristina.bridge")

RUST_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
#                    ПОПЫТКА ЗАГРУЗКИ RUST
# ═══════════════════════════════════════════════════════════════

try:
    import kristina_core

    _RustMemoryEngine = kristina_core.MemoryEngine
    _RustEmbeddingCache = kristina_core.EmbeddingCache
    EmotionAnalyzer = kristina_core.EmotionAnalyzer
    ToolCallParser = kristina_core.ToolCallParser
    _RustContextCompressor = kristina_core.ContextCompressor
    _RustThreadTracker = kristina_core.ThreadTracker
    cosine_similarity = kristina_core.cosine_similarity
    batch_cosine_similarity = kristina_core.batch_cosine_similarity

    RUST_AVAILABLE = True
    logger.info("🦀 Rust ядро загружено")

except ImportError as e:
    logger.warning(f"⚠️ Rust ядро не найдено ({e}) → Python fallback")

    from _fallback.memory_engine import MemoryEngine as _RustMemoryEngine
    from _fallback.embedding_cache import EmbeddingCache as _RustEmbeddingCache
    from _fallback.emotion_analyzer import EmotionAnalyzer
    from _fallback.tool_parser import ToolCallParser
    from _fallback.context_compressor import ContextCompressor as _RustContextCompressor
    from _fallback.thread_tracker import ThreadTracker as _RustThreadTracker

    try:
        import numpy as np

        def cosine_similarity(a, b):
            a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            dot = np.dot(a, b)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(dot / (na * nb)) if na * nb > 1e-8 else 0.0

        def batch_cosine_similarity(query, documents, top_k=5):
            q = np.asarray(query, dtype=np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-8:
                return []
            results = []
            for i, doc in enumerate(documents):
                d = np.asarray(doc, dtype=np.float32)
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-8 or len(d) != len(q):
                    continue
                sim = float(np.dot(q, d) / (q_norm * d_norm))
                results.append((i, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    except ImportError:
        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            return dot / (na * nb) if na * nb > 1e-8 else 0.0

        def batch_cosine_similarity(query, documents, top_k=5):
            sims = [(i, cosine_similarity(query, d)) for i, d in enumerate(documents)]
            sims.sort(key=lambda x: x[1], reverse=True)
            return sims[:top_k]


# ═══════════════════════════════════════════════════════════════
#     АДАПТЕР: MemoryEngine → единый API
# ═══════════════════════════════════════════════════════════════

class MemoryAdapter:
    """
    Единый адаптер для MemoryEngine (Rust и Python fallback).

    Нормализует API:
    - get_stats() → всегда dict
    - get_relevant_context() → всегда str
    - .working → всегда list of dicts
    - Интегрирует v6.0: KnowledgeGraph, MemorySummarizer
    """

    def __init__(self, memory_dir: str, working_size: int = 15, max_episodic: int = 2000):
        self._engine = _RustMemoryEngine(memory_dir, working_size, max_episodic)
        self._summarizer = None
        self._knowledge_graph = None

    @property
    def working(self):
        if RUST_AVAILABLE:
            raw = self._engine.get_working_memory()
            return [{"role": r, "content": c, "timestamp": t} for r, c, t in raw]
        if hasattr(self._engine, 'working'):
            return self._engine.working
        return []

    def add_to_working(self, role: str, content: str):
        self._engine.add_to_working(role, content)

    def clear_working(self):
        self._engine.clear_working()

    def add_episode(self, user_input: str, response: str, emotion: str, importance: int = 1):
        self._engine.add_episode(user_input, response, emotion, importance)
        if self.knowledge_graph and importance >= 2:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self.knowledge_graph.extract_and_add(user_input, response, importance)
                    )
                else:
                    for s, p, o in self.knowledge_graph._regex_extract(user_input):
                        self.knowledge_graph.add_triple(s, p, o, importance, "regex")
            except Exception:
                pass

    def get_relevant_context(self, query: str, max_items: int = 3) -> str:
        parts = []

        if RUST_AVAILABLE:
            for ts, preview, _score in self._engine.get_relevant_context(query, max_items):
                parts.append(f"[{ts[:10]}] Пользователь: {preview}")
        else:
            result = self._engine.get_relevant_context(query, max_items)
            if isinstance(result, str) and result != "Нет релевантного контекста.":
                parts.append(result)
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, tuple) and len(item) >= 2:
                        parts.append(f"[{item[0][:10]}] Пользователь: {item[1]}")

        if self.knowledge_graph:
            try:
                kg = self.knowledge_graph.get_context_for_query(query, max_facts=3)
                if kg:
                    parts.append(kg)
            except Exception:
                pass

        if len(parts) < max_items and self.summarizer:
            try:
                for r in self.summarizer.search_summaries(query, max_results=2):
                    parts.append(f"[{r['level']} {r['key']}] {r['summary'][:150]}")
            except Exception:
                pass

        return "\n".join(parts) if parts else "Нет релевантного контекста."

    def get_stats(self) -> dict:
        if RUST_AVAILABLE:
            w, e, s = self._engine.get_stats()
            stats = {"working": w, "episodic": e, "semantic_keys": s}
        else:
            raw = self._engine.get_stats()
            if isinstance(raw, dict):
                stats = raw
            elif isinstance(raw, tuple):
                stats = {"working": raw[0], "episodic": raw[1], "semantic_keys": raw[2] if len(raw) > 2 else 0}
            else:
                stats = {"working": 0, "episodic": 0, "semantic_keys": 0}

        if self.summarizer:
            try:
                stats["summaries"] = self.summarizer.get_stats()
            except Exception:
                pass
        if self.knowledge_graph:
            try:
                stats["knowledge_graph"] = self.knowledge_graph.get_stats()
            except Exception:
                pass
        return stats

    def save(self):
        self._engine.save()
        if self.knowledge_graph:
            try:
                self.knowledge_graph.save()
            except Exception:
                pass

    def load(self):
        pass

    def add_semantic(self, key: str, value: str):
        if hasattr(self._engine, 'add_semantic'):
            self._engine.add_semantic(key, value)

    def get_semantic(self, key: str):
        if hasattr(self._engine, 'get_semantic'):
            return self._engine.get_semantic(key)
        return None

    @property
    def summarizer(self):
        if self._summarizer is None:
            try:
                from modules.memory.summarizer import MemorySummarizer
                self._summarizer = MemorySummarizer()
            except Exception:
                pass
        return self._summarizer

    @property
    def knowledge_graph(self):
        if self._knowledge_graph is None:
            try:
                import config as cfg
                if cfg.config.knowledge_graph_enabled:
                    from modules.memory.knowledge_graph import KnowledgeGraph
                    self._knowledge_graph = KnowledgeGraph()
            except Exception:
                pass
        return self._knowledge_graph


# ═══════════════════════════════════════════════════════════════
#     АДАПТЕР: EmbeddingCache
# ═══════════════════════════════════════════════════════════════

class EmbeddingCacheAdapter:
    def __init__(self, cache_dir: str, max_size: int = 10000):
        self._cache = _RustEmbeddingCache(cache_dir, max_size)

    def get(self, text): return self._cache.get(text)
    def put(self, text, embedding): self._cache.put(text, embedding)
    def contains(self, text): return self._cache.contains(text)
    def len(self): return self._cache.len()
    def save(self): self._cache.save()
    def clear(self): self._cache.clear()

    def get_stats(self):
        if RUST_AVAILABLE:
            size, hits, misses = self._cache.get_stats()
            return {"size": size, "hits": hits, "misses": misses}
        return self._cache.get_stats() if hasattr(self._cache, 'get_stats') else {}


# ═══════════════════════════════════════════════════════════════
#     АДАПТЕР: ContextCompressor
# ═══════════════════════════════════════════════════════════════

class ContextCompressorAdapter:
    def __init__(self, compression_ratio: float = 0.3):
        self._impl = _RustContextCompressor(compression_ratio)

    def compress_conversation(self, messages): return self._impl.compress_conversation(messages)
    def extract_key_points(self, text): return self._impl.extract_key_points(text)
    def summarize_episodes(self, episodes, max_length=500): return self._impl.summarize_episodes(episodes, max_length)
    def estimate_tokens(self, text): return self._impl.estimate_tokens(text)
    def truncate_to_tokens(self, text, max_tokens): return self._impl.truncate_to_tokens(text, max_tokens)


# ═══════════════════════════════════════════════════════════════
#     АДАПТЕР: ThreadTracker
# ═══════════════════════════════════════════════════════════════

class ThreadTrackerAdapter:
    def __init__(self, timeout_secs=600):
        self._tracker = _RustThreadTracker(timeout_secs)
        self._v4_current_thread = None

    @property
    def current_thread(self):
        topic = None
        if hasattr(self._tracker, 'get_current_topic'):
            topic = self._tracker.get_current_topic()
        if topic:
            if self._v4_current_thread is None or self._v4_current_thread.get("topic") != topic:
                self._v4_current_thread = {"topic": topic, "entities": [], "messages": []}
            return self._v4_current_thread
        if hasattr(self._tracker, 'current_thread') and self._tracker.current_thread:
            return self._tracker.current_thread
        return self._v4_current_thread

    def start_thread(self, topic, entities=None):
        entities = entities or []
        self._tracker.start_thread(topic, entities)
        self._v4_current_thread = {"topic": topic, "entities": entities, "messages": []}

    def add_to_thread(self, user_input, response):
        self.add_message(user_input, response)

    def add_message(self, user_input, response):
        if hasattr(self._tracker, 'add_message'):
            self._tracker.add_message(user_input, response)
        if self._v4_current_thread is not None:
            self._v4_current_thread["messages"].append({"user": user_input, "assistant": response})

    def update(self, user_input, response):
        self._tracker.update(user_input, response)
        topic = self._tracker.get_current_topic() if hasattr(self._tracker, 'get_current_topic') else None
        if topic and self._v4_current_thread is None:
            self._v4_current_thread = {"topic": topic, "entities": [], "messages": []}
        if self._v4_current_thread:
            self._v4_current_thread.setdefault("messages", []).append({"user": user_input, "assistant": response})

    def is_related(self, text): return self._tracker.is_related(text)

    def is_related_to_thread(self, text):
        for attr in ('is_related', 'is_related_to_thread'):
            if hasattr(self._tracker, attr):
                return getattr(self._tracker, attr)(text)
        return False

    def get_context(self, max_messages=10):
        return (self._tracker.get_context() or "") if hasattr(self._tracker, 'get_context') else ""

    def get_current_topic(self):
        if hasattr(self._tracker, 'get_current_topic'):
            return self._tracker.get_current_topic()
        return self._v4_current_thread.get("topic") if self._v4_current_thread else None

    def has_active_thread(self):
        if hasattr(self._tracker, 'has_active_thread'):
            return self._tracker.has_active_thread()
        return self._v4_current_thread is not None

    def get_past_threads(self, limit=5):
        return self._tracker.get_past_threads(limit) if hasattr(self._tracker, 'get_past_threads') else []

    def end_thread(self):
        if hasattr(self._tracker, 'end_thread'):
            self._tracker.end_thread()
        self._v4_current_thread = None

    def get_stats(self):
        if hasattr(self._tracker, 'get_stats'):
            return self._tracker.get_stats()
        return {"current_thread": self._v4_current_thread is not None}

    def save(self): pass
    def load(self): pass


# ═══════════════════════════════════════════════════════════════
#                     ЭКСПОРТ
# ═══════════════════════════════════════════════════════════════

MemoryEngine = MemoryAdapter
EmbeddingCache = EmbeddingCacheAdapter
ContextCompressor = ContextCompressorAdapter
ThreadTracker = ThreadTrackerAdapter

backend = "🦀 Rust" if RUST_AVAILABLE else "🐍 Python"
logger.info(f"✅ Bridge загружен ({backend})")
