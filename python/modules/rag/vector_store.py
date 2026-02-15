"""
Кристина 6.0 — Векторная память

ИЗМЕНЕНИЯ v6.0:
- ✅ PersistentClient вместо in-memory Client (данные НЕ теряются при рестарте!)
- ✅ JSON кэш вместо pickle
- ✅ Async-safe embedding через ollama.AsyncClient
- ✅ Graceful fallback если ChromaDB недоступен
- ✅ Убран дубликат modules/rag/memory.py
"""

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import ollama

try:
    import orjson

    def _json_load(f):
        return orjson.loads(f.read())

    def _json_dump(obj, f):
        f.write(orjson.dumps(obj))
except ImportError:
    import json

    def _json_load(f):
        return json.load(f)

    def _json_dump(obj, f):
        json.dump(obj, f)

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
        self._persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = _init_chromadb(persist_dir)
        self.collection = None

        if self.client is not None:
            try:
                self.collection = self.client.get_or_create_collection(
                    name="kristina_memory",
                )
                logger.info("✅ Коллекция kristina_memory готова")
            except (KeyError, Exception) as e:
                logger.warning(f"⚠️ Ошибка коллекции ({e}), пересоздаю...")
                try:
                    import shutil
                    self.client = None
                    shutil.rmtree(persist_dir, ignore_errors=True)
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    self.client = _init_chromadb(persist_dir)
                    if self.client:
                        self.collection = self.client.get_or_create_collection(
                            name="kristina_memory",
                        )
                        logger.info("✅ ChromaDB пересоздана с нуля")
                except Exception as e2:
                    logger.error(f"❌ Не удалось пересоздать ChromaDB: {e2}")

        # Один async-клиент для всех embedding-запросов (предотвращает утечку транспортов)
        self._async_client: Optional[ollama.AsyncClient] = None

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

        doc_id = f"dialogue_{now.strftime('%Y%m%d_%H%M%S')}_{self.doc_counter}"
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

        # v7.4: Reranking с temporal decay + keyword overlap + importance
        formatted = self._rerank(formatted, query)

        return formatted[:n_results]

    async def search_async(
        self,
        query: str,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None,
        date_range: Optional[tuple] = None,
    ) -> List[Dict]:
        """Async семантический поиск — не блокирует event loop"""
        if self.collection is None:
            return []

        n_results = n_results or config.VECTOR_SEARCH_RESULTS

        query_embedding = await self._get_embedding_async(query)

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

        # v7.4: Reranking с temporal decay + keyword overlap + importance
        formatted = self._rerank(formatted, query)

        return formatted[:n_results]

    async def add_dialogue_async(
        self,
        user_input: str,
        assistant_response: str,
        importance: int = 1,
        metadata: Optional[Dict] = None,
    ):
        """Async сохранение диалога — не блокирует event loop"""
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
            "keywords": self._extract_keywords(text),
            "category": self._classify_category(user_input),
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        # keywords needs to be a string for ChromaDB
        if isinstance(meta["keywords"], list):
            meta["keywords"] = json.dumps(meta["keywords"], ensure_ascii=False)

        embedding = await self._get_embedding_async(text)

        doc_id = f"dialogue_{now.strftime('%Y%m%d_%H%M%S')}_{self.doc_counter}"
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

    # ── Reranking (v7.4) ──

    def _rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """
        v7.4: Трёхфакторное переранжирование результатов RAG.

        Финальный score = w1*semantic + w2*temporal + w3*keyword + w4*importance

        1. Semantic score — cosine distance из ChromaDB (инвертированная)
        2. Temporal decay — свежие воспоминания получают бонус (полураспад 7 дней)
        3. Keyword overlap — совпадение ключевых слов запроса и документа
        4. Importance — важность из метаданных
        """
        if not results:
            return results

        query_keywords = set(self._extract_keywords(query))
        now = datetime.now()

        for item in results:
            meta = item["metadata"]
            distance = item.get("distance")

            # 1. Semantic: distance → similarity (cosine distance: 0=идентичны, 2=противоположны)
            semantic = 1.0 - (distance / 2.0) if distance is not None else 0.5

            # 2. Temporal decay: exp(-lambda * age_days), half-life = 7 дней
            temporal = 0.5
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    doc_time = datetime.fromisoformat(ts)
                    age_days = max((now - doc_time).total_seconds() / 86400, 0)
                    half_life = 7.0
                    temporal = math.exp(-0.693 * age_days / half_life)  # ln(2) ≈ 0.693
                except (ValueError, TypeError):
                    pass

            # 3. Keyword overlap: Jaccard-like
            doc_keywords = set()
            kw_raw = meta.get("keywords", "")
            if isinstance(kw_raw, str):
                try:
                    doc_keywords = set(json.loads(kw_raw))
                except (ValueError, TypeError):
                    doc_keywords = set(kw_raw.split())
            elif isinstance(kw_raw, list):
                doc_keywords = set(kw_raw)

            if query_keywords and doc_keywords:
                overlap = len(query_keywords & doc_keywords)
                union = len(query_keywords | doc_keywords)
                keyword_score = overlap / union if union > 0 else 0.0
            else:
                keyword_score = 0.0

            # 4. Importance
            importance = meta.get("importance", 1)
            importance_score = min(importance / 3.0, 1.0)

            # Взвешенная сумма
            final_score = (
                0.50 * semantic +
                0.20 * temporal +
                0.15 * keyword_score +
                0.15 * importance_score
            )

            item["_rerank_score"] = final_score

        # Сортируем по финальному score (убывание)
        results.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
        return results

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

    def _get_async_client(self) -> ollama.AsyncClient:
        """Возвращает переиспользуемый AsyncClient (предотвращает утечку транспортов)"""
        if self._async_client is None:
            self._async_client = ollama.AsyncClient()
        return self._async_client

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async embedding через ollama.AsyncClient — не блокирует event loop"""
        client = self._get_async_client()

        # Shared cache (Rust EmbeddingCacheAdapter)
        if self._shared_cache is not None:
            cached = self._shared_cache.get(text)
            if cached is not None:
                return cached

            try:
                response = await client.embeddings(
                    model=config.EMBEDDING_MODEL,
                    prompt=text,
                )
                embedding = response["embedding"]
                self._shared_cache.put(text, embedding)
                return embedding
            except Exception as e:
                logger.error(f"❌ Ошибка async embedding: {e}")
                return [0.0] * config.EMBEDDING_DIM

        # Local cache fallback
        text_hash = hashlib.md5(text.encode()).hexdigest()

        if config.EMBEDDING_CACHE_ENABLED and text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        try:
            response = await client.embeddings(
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
            logger.error(f"❌ Ошибка async embedding: {e}")
            return [0.0] * config.EMBEDDING_DIM

    def _load_embedding_cache(self):
        """Загружает кэш (orjson ~5x быстрее stdlib json для больших файлов)"""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    self.embedding_cache = _json_load(f)
                logger.info(f"✅ Кэш embeddings: {len(self.embedding_cache)} записей")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
                self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Сохраняет кэш (orjson ~5x быстрее stdlib json для больших файлов)"""
        try:
            if len(self.embedding_cache) > config.EMBEDDING_CACHE_MAX_SIZE:
                items = list(self.embedding_cache.items())
                self.embedding_cache = dict(items[-config.EMBEDDING_CACHE_MAX_SIZE:])

            with open(self._cache_path, "wb") as f:
                _json_dump(self.embedding_cache, f)
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

    async def close(self):
        """Закрывает async-клиент Ollama (предотвращает ResourceWarning)"""
        if self._async_client is not None:
            try:
                if hasattr(self._async_client, '_client') and self._async_client._client:
                    await self._async_client._client.aclose()
            except Exception:
                pass
            self._async_client = None

    def save_cache(self):
        """Явное сохранение кэша (вызывается при shutdown)"""
        if config.EMBEDDING_CACHE_ENABLED:
            self._save_embedding_cache()
