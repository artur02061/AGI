"""
Python Fallback: MemoryEngine

Зеркалит API Rust kristina_core.MemoryEngine:
- add_to_working(role, content)
- add_episode(user_input, response, emotion, importance)
- get_relevant_context(query, max_items=3) → List[(timestamp, preview, score)]
- get_working_memory() → List[(role, content, timestamp)]
- clear_working()
- save()
- get_stats() → (working, episodic, semantic)
- add_semantic(key, value)
- get_semantic(key) → Optional[str]
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import threading


class MemoryEngine:
    """Управление памятью — Python fallback для Rust MemoryEngine"""

    def __init__(self, memory_dir: str, working_size: int = 10, max_episodic: int = 1000):
        self._dir = Path(memory_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._episodic_path = self._dir / "episodic.json"
        self._semantic_path = self._dir / "semantic.json"

        self._working_size = working_size
        self._max_episodic = max_episodic

        self._working: List[Dict] = []
        self._episodic: List[Dict] = []
        self._semantic: Dict[str, str] = {}

        # Индекс ключевых слов: hash(word) → set(episode_indices)
        self._keyword_index: Dict[int, List[int]] = defaultdict(list)

        self._lock = threading.RLock()

        self._load_from_disk()

    # ── Рабочая память ──

    def add_to_working(self, role: str, content: str):
        with self._lock:
            self._working.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            while len(self._working) > self._working_size:
                self._working.pop(0)

    def get_working_memory(self) -> List[Tuple[str, str, str]]:
        with self._lock:
            return [(m["role"], m["content"], m["timestamp"]) for m in self._working]

    def clear_working(self):
        with self._lock:
            self._working.clear()

    # ── Эпизодическая память ──

    def add_episode(self, user_input: str, response: str, emotion: str, importance: int = 1):
        with self._lock:
            keywords = self._extract_keywords(user_input)
            episode = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_input": user_input,
                "response": response,
                "emotion": emotion,
                "importance": importance,
                "keywords": keywords,
            }

            idx = len(self._episodic)
            self._episodic.append(episode)

            # Обновляем индекс
            combined = f"{user_input} {response}"
            self._index_text(idx, combined)

            # Ротация по размеру
            if len(self._episodic) > self._max_episodic:
                self._evict_episodes()

    def get_relevant_context(self, query: str, max_items: int = 3) -> List[Tuple[str, str, int]]:
        """Поиск через keyword index (как в Rust)"""
        with self._lock:
            scores: Dict[int, int] = defaultdict(int)

            for word in query.split():
                word_lower = word.lower()
                if len(word_lower) <= 2:
                    continue
                h = self._word_hash(word_lower)
                if h in self._keyword_index:
                    for idx in self._keyword_index[h]:
                        scores[idx] += 1

            if not scores:
                return []

            # Комбинируем keyword score с importance
            results = []
            for idx, keyword_score in scores.items():
                if idx >= len(self._episodic):
                    continue
                ep = self._episodic[idx]
                final_score = keyword_score * ep.get("importance", 1)
                preview = ep["user_input"][:80]
                results.append((ep["timestamp"], preview, final_score))

            results.sort(key=lambda x: x[2], reverse=True)
            return results[:max_items]

    # ── Семантическая память ──

    def add_semantic(self, key: str, value: str):
        with self._lock:
            self._semantic[key] = value

    def get_semantic(self, key: str) -> Optional[str]:
        with self._lock:
            return self._semantic.get(key)

    # ── Персистентность ──

    def save(self):
        with self._lock:
            try:
                with open(self._episodic_path, "w", encoding="utf-8") as f:
                    json.dump(self._episodic, f, ensure_ascii=False, indent=2)
                with open(self._semantic_path, "w", encoding="utf-8") as f:
                    json.dump(self._semantic, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Ошибка сохранения памяти: {e}")

    def get_stats(self) -> Tuple[int, int, int]:
        with self._lock:
            return (len(self._working), len(self._episodic), len(self._semantic))

    # ── Внутренние методы ──

    def _load_from_disk(self):
        try:
            if self._episodic_path.exists():
                with open(self._episodic_path, "r", encoding="utf-8") as f:
                    self._episodic = json.load(f)
                # Перестраиваем индекс
                self._rebuild_index()
        except Exception as e:
            print(f"⚠️ Ошибка загрузки эпизодической памяти: {e}")
            self._episodic = []

        try:
            if self._semantic_path.exists():
                with open(self._semantic_path, "r", encoding="utf-8") as f:
                    self._semantic = json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки семантической памяти: {e}")
            self._semantic = {}

    def _rebuild_index(self):
        self._keyword_index.clear()
        for i, ep in enumerate(self._episodic):
            combined = f"{ep.get('user_input', '')} {ep.get('response', '')}"
            self._index_text(i, combined)

    def _index_text(self, idx: int, text: str):
        for word in text.split():
            word_lower = word.lower()
            if len(word_lower) > 2:
                h = self._word_hash(word_lower)
                self._keyword_index[h].append(idx)

    def _evict_episodes(self):
        """Удаляет 10% наименее важных эпизодов (с учётом возраста)"""
        remove_count = max(1, self._max_episodic // 10)
        now = datetime.now(timezone.utc)

        def eviction_score(item):
            idx, ep = item
            age_hours = 1.0
            try:
                ts = datetime.fromisoformat(ep['timestamp'])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = max(1.0, (now - ts).total_seconds() / 3600)
            except (KeyError, ValueError):
                pass
            return ep.get("importance", 0) / age_hours

        scored = sorted(enumerate(self._episodic), key=eviction_score)
        indices_to_remove = sorted([i for i, _ in scored[:remove_count]], reverse=True)
        for idx in indices_to_remove:
            if idx < len(self._episodic):
                self._episodic.pop(idx)
        self._rebuild_index()

    @staticmethod
    def _word_hash(word: str) -> int:
        return int(hashlib.md5(word.encode()).hexdigest(), 16)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        stop_words = {
            "я", "ты", "он", "она", "мы", "вы", "они", "в", "на", "и",
            "с", "по", "для", "от", "к", "не", "что", "это", "как", "но",
            "the", "is", "are", "a", "an", "in", "on", "for", "to", "of",
        }
        return [
            w.lower()
            for w in text.split()
            if len(w) > 3 and w.lower() not in stop_words
        ][:10]
