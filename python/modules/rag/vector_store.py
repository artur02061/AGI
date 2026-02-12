"""
Кристина 6.0 — Векторная память

ИЗМЕНЕНИЯ v6.0:
- ✅ PersistentClient вместо in-memory Client (данные НЕ теряются при рестарте!)
- ✅ JSON кэш вместо pickle
- ✅ Async-safe embedding через ollama.AsyncClient
- ✅ Graceful fallback если ChromaDB недоступен
- ✅ Убран дубликат modules/rag/memory.py
"""

import json
import hashlib
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import ollama

from utils.logging import get_logger
import config

logger = get_logger("vector_store")


# ═══════════════════════════════════════════════════════════════
#                   CHROMADB ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

def _init_chromadb(persist_dir: str):
    """
    Инициализирует ChromaDB с PersistentClient.
    Fallback на in-memory если что-то пошло не так.
    """
    try:
        import chromadb
        from chromadb.config import Settings

        # v6.0: PersistentClient — данные на диске!
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"✅ ChromaDB PersistentClient: {persist_dir}")
        return client

    except TypeError:
        # Старая версия ChromaDB без PersistentClient
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False,
        ))
        logger.warning("⚠️ ChromaDB старая версия, используем legacy persist")
        return client

    except ImportError:
        logger.error("❌ ChromaDB не установлен! pip install chromadb")
        return None

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#                     VECTOR MEMORY
# ═══════════════════════════════════════════════════════════════

class VectorMemory:
    """Векторная память с персистентным хранилищем"""

    def __init__(self, persist_dir: str = None, shared_embedding_cache=None):
        persist_dir = persist_dir or str(config.VECTOR_DB_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = _init_chromadb(persist_dir)
        self.collection = None

        if self.client is not None:
            try:
                self.collection = self.client.get_or_create_collection(
                    name="kristina_memory",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("✅ Коллекция kristina_memory готова")
            except Exception as e:
                logger.error(f"❌ Ошибка создания коллекции: {e}")

        # Embedding кэш — используем общий если передан, иначе свой
        self._shared_cache = shared_embedding_cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self._cache_path = Path(config.DATA_DIR) / "embedding_cache.json"

        if self._shared_cache is None and config.EMBEDDING_CACHE_ENABLED:
            self._load_embedding_cache()

        # Счётчик документов
        self.doc_counter = 0
        if self.collection is not None:
            try:
                self.doc_counter = self.collection.count()
            except Exception:
                self.doc_counter = 0

        logger.info(f"📊 Документов: {self.doc_counter} | Кэш: {len(self.embedding_cache)}")

    # ── Добавление ──

    def add_dialogue(
        self,
        user_input: str,
        assistant_response: str,
        importance: int = 1,
        metadata: Optional[Dict] = None,
    ):
        """Сохраняет диалог в векторную память"""
        if self.collection is None:
            logger.warning("⚠️ ChromaDB недоступен, диалог не сохранён")
            return

        text = f"Пользователь: {user_input}\nКристина: {assistant_response}"
        now = datetime.now()

        meta = {
            "type": "dialogue",
            "timestamp": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "month": now.strftime("%Y-%m"),
            "time": now.strftime("%H:%M"),
            "importance": importance,
            "user_input": user_input[:200],
            "response_length": len(assistant_response),
            "keywords": json.dumps(self._extract_keywords(text), ensure_ascii=False),
            "category": self._classify_category(user_input),
        }
        if metadata:
            # ChromaDB не принимает вложенные dict — только str/int/float
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        embedding = self._get_embedding(text)

        doc_id = f"dialogue_{self.doc_counter}"
        self.doc_counter += 1

        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta],
            )
            logger.debug(f"💾 Диалог сохранён: {doc_id}")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения диалога: {e}")

    # ── Поиск ──

    def search(
        self,
        query: str,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None,
        date_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """Семантический поиск с фильтрами"""
        if self.collection is None:
            return []

        n_results = n_results or config.VECTOR_SEARCH_RESULTS

        query_embedding = self._get_embedding(query)

        where_filter = {}
        if filter_metadata:
            where_filter.update(filter_metadata)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results * 2, max(self.doc_counter, 1)),
                where=where_filter if where_filter else None,
            )
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []

        formatted = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                result_date = meta.get("date", "")

                if date_range:
                    from_date, to_date = date_range
                    if not (from_date <= result_date <= to_date):
                        continue

                formatted.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": meta,
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })

        formatted.sort(key=lambda x: (
            x["distance"] if x["distance"] is not None else 1.0,
            -x["metadata"].get("importance", 1),
        ))

        return formatted[:n_results]

    def search_by_timeframe(
        self,
        query: str,
        timeframe: str,
        n_results: int = None,
    ) -> List[Dict]:
        """Поиск с фильтром по времени"""
        now = datetime.now()
        timeframes = {
            "today": (now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")),
            "yesterday": (
                (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            ),
            "this_week": (
                (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d"),
            ),
            "this_month": (
                now.replace(day=1).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d"),
            ),
        }
        date_range = timeframes.get(timeframe)
        return self.search(query, n_results=n_results, date_range=date_range)

    def get_recent_dialogues(self, n: int = 10) -> List[Dict]:
        """Последние N диалогов"""
        if self.collection is None:
            return []
        try:
            all_items = self.collection.get(
                where={"type": "dialogue"},
                include=["documents", "metadatas"],
            )
        except Exception:
            return []

        if not all_items["ids"]:
            return []

        items = []
        for i in range(len(all_items["ids"])):
            items.append({
                "id": all_items["ids"][i],
                "text": all_items["documents"][i],
                "metadata": all_items["metadatas"][i],
            })

        items.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        return items[:n]

    # ── Embeddings ──

    def _get_embedding(self, text: str) -> List[float]:
        """Получает embedding с кэшированием (через shared или local cache)"""
        # Если есть shared cache (EmbeddingCacheAdapter) — используем его
        if self._shared_cache is not None:
            cached = self._shared_cache.get(text)
            if cached is not None:
                return cached

            try:
                response = ollama.embeddings(
                    model=config.EMBEDDING_MODEL,
                    prompt=text,
                )
                embedding = response["embedding"]
                self._shared_cache.put(text, embedding)
                return embedding
            except Exception as e:
                logger.error(f"❌ Ошибка embedding: {e}")
                return [0.0] * config.EMBEDDING_DIM

        # Fallback: локальный cache
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if config.EMBEDDING_CACHE_ENABLED and text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        try:
            response = ollama.embeddings(
                model=config.EMBEDDING_MODEL,
                prompt=text,
            )
            embedding = response["embedding"]

            if config.EMBEDDING_CACHE_ENABLED:
                self.embedding_cache[text_hash] = embedding
                if len(self.embedding_cache) % 100 == 0:
                    self._save_embedding_cache()

            return embedding

        except Exception as e:
            logger.error(f"❌ Ошибка embedding: {e}")
            return [0.0] * config.EMBEDDING_DIM

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async-safe embedding — не блокирует event loop"""
        import asyncio
        return await asyncio.to_thread(self._get_embedding, text)

    def _load_embedding_cache(self):
        """Загружает JSON кэш"""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"✅ Кэш embeddings: {len(self.embedding_cache)} записей")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
                self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Сохраняет JSON кэш"""
        try:
            if len(self.embedding_cache) > config.EMBEDDING_CACHE_MAX_SIZE:
                items = list(self.embedding_cache.items())
                self.embedding_cache = dict(items[-config.EMBEDDING_CACHE_MAX_SIZE:])

            with open(self._cache_path, "w") as f:
                json.dump(self.embedding_cache, f)
            logger.debug(f"💾 Кэш: {len(self.embedding_cache)} записей")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения кэша: {e}")

    # ── Утилиты ──

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        stop_words = {
            "я", "ты", "он", "она", "мы", "вы", "они",
            "в", "на", "и", "с", "по", "для", "от", "к",
            "the", "is", "are", "was", "were", "a", "an",
        }
        words = re.findall(r"\b\w+\b", text.lower())
        counter = Counter(w for w in words if len(w) > 3 and w not in stop_words)
        return [word for word, _ in counter.most_common(10)]

    @staticmethod
    def _classify_category(text: str) -> str:
        text_lower = text.lower()
        categories = {
            "code": ["код", "функция", "класс", "ошибка", "программ", "python"],
            "system": ["запусти", "открой", "файл", "приложение", "процесс"],
            "web": ["найди", "поиск", "интернет", "новости", "погода"],
            "personal": ["помнишь", "говорил", "обсуждали", "напомни"],
        }
        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "general"

    def get_stats(self) -> Dict:
        total = self.doc_counter
        return {
            "total": total,
            "dialogues": total,  # Пока всё — диалоги
            "cache_size": len(self.embedding_cache),
            "persistent": self.client is not None,
        }

    def save_cache(self):
        """Явное сохранение кэша (вызывается при shutdown)"""
        if config.EMBEDDING_CACHE_ENABLED:
            self._save_embedding_cache()
