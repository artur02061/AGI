"""
Python Fallback: EmbeddingCache

Зеркалит API Rust kristina_core.EmbeddingCache:
- get(text) → Optional[List[float]]
- put(text, embedding)
- contains(text) → bool
- len() → int
- save()
- clear()
- get_stats() → (size, hits, misses)
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading


class EmbeddingCache:
    """Lock-free кэш эмбеддингов — Python fallback"""

    def __init__(self, cache_dir: str, max_size: int = 10000):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / "embedding_cache.json"
        self._max_size = max_size

        self._cache: Dict[str, List[float]] = {}
        self._access_count: Dict[str, int] = {}
        self._hits = 0
        self._misses = 0

        self._lock = threading.RLock()
        self._load_from_disk()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        h = self._text_hash(text)
        with self._lock:
            if h in self._cache:
                self._access_count[h] = self._access_count.get(h, 0) + 1
                self._hits += 1
                return self._cache[h]
            self._misses += 1
            return None

    def put(self, text: str, embedding: List[float]):
        h = self._text_hash(text)
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            self._cache[h] = embedding
            self._access_count[h] = 1

    def contains(self, text: str) -> bool:
        h = self._text_hash(text)
        with self._lock:
            return h in self._cache

    def len(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Tuple[int, int, int]:
        with self._lock:
            return (len(self._cache), self._hits, self._misses)

    def save(self):
        with self._lock:
            try:
                with open(self._cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f)
            except Exception as e:
                print(f"⚠️ Ошибка сохранения кэша: {e}")

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._hits = 0
            self._misses = 0

    # ── Внутренние ──

    def _load_from_disk(self):
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            self._access_count = {k: 0 for k in self._cache}
        except Exception as e:
            print(f"⚠️ Ошибка загрузки кэша: {e}")
            self._cache = {}

    def _evict_lru(self):
        """Удаляет 10% с наименьшим access_count"""
        evict_count = max(1, self._max_size // 10)
        sorted_keys = sorted(
            self._access_count.keys(),
            key=lambda k: self._access_count.get(k, 0),
        )
        for key in sorted_keys[:evict_count]:
            self._cache.pop(key, None)
            self._access_count.pop(key, None)
