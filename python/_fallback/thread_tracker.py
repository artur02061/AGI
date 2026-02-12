"""
Python Fallback: ThreadTracker

Зеркалит API Rust kristina_core.ThreadTracker:
- start_thread(topic, entities)
- add_message(user_input, response)
- update(user_input, response)
- is_related(text) → bool
- get_context() → Optional[str]
- has_active_thread() → bool
- get_current_topic() → Optional[str]
- get_past_threads(limit=5) → List[(topic, duration_secs, msg_count)]
- end_thread()
"""

import re
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple


class ThreadTracker:
    """Отслеживание нитей разговора — Python fallback для Rust"""

    CONTEXT_INDICATORS = [
        "помнишь", "как мы говорили", "в той", "наша",
        "тот", "та", "то", "это", "об этом",
        "продолжим", "вернёмся к", "по поводу",
    ]

    def __init__(self, timeout_secs: int = 600):
        self.timeout_secs = timeout_secs
        self._current: Optional[Dict] = None
        self._history: List[Dict] = []
        self._lock = threading.RLock()

        self._context_re = re.compile(
            "|".join(re.escape(w) for w in self.CONTEXT_INDICATORS),
            re.IGNORECASE,
        )

    # ── v4.0 совместимость ──

    @property
    def current_thread(self):
        """Для обратной совместимости с v4.0 API"""
        with self._lock:
            return self._current

    @current_thread.setter
    def current_thread(self, value):
        with self._lock:
            self._current = value

    # ── Основные методы (Rust API) ──

    def start_thread(self, topic: str, entities: List[str] = None):
        with self._lock:
            if self._current:
                self._archive_current()
            self._current = {
                "topic": topic,
                "entities": entities or [],
                "started": datetime.now(timezone.utc),
                "messages": [],
            }

    def add_message(self, user_input: str, response: str):
        with self._lock:
            if self._current:
                self._current["messages"].append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now(timezone.utc),
                })

    def update(self, user_input: str, response: str):
        """Создаёт нить если нет, добавляет сообщение, проверяет timeout"""
        now = datetime.now(timezone.utc)
        with self._lock:
            # Проверяем timeout
            if self._current and self._current["messages"]:
                last_ts = self._current["messages"][-1]["timestamp"]
                elapsed = (now - last_ts).total_seconds()
                if elapsed > self.timeout_secs:
                    self._archive_current()

            # Создаём если нет
            if not self._current:
                self._current = {
                    "topic": user_input[:50],
                    "entities": [],
                    "started": now,
                    "messages": [],
                }

            self._current["messages"].append({
                "user": user_input,
                "assistant": response,
                "timestamp": now,
            })

    def is_related(self, text: str) -> bool:
        with self._lock:
            if not self._current:
                return False

            # Timeout
            elapsed = (datetime.now(timezone.utc) - self._current["started"]).total_seconds()
            if elapsed > self.timeout_secs:
                return False

            text_lower = text.lower()

            # Тема
            if self._current["topic"].lower() in text_lower:
                return True

            # Сущности
            for entity in self._current.get("entities", []):
                if entity.lower() in text_lower:
                    return True

            # Контекстные маркеры
            return bool(self._context_re.search(text_lower))

    # Алиас для совместимости с v4.0
    def is_related_to_thread(self, text: str) -> bool:
        return self.is_related(text)

    def get_context(self) -> Optional[str]:
        with self._lock:
            if not self._current:
                return None

            elapsed = (datetime.now(timezone.utc) - self._current["started"]).total_seconds()
            if elapsed > self.timeout_secs:
                return None

            parts = [f"Текущая тема: {self._current['topic']}"]

            entities = self._current.get("entities", [])
            if entities:
                parts.append(f"Упоминается: {', '.join(entities[:5])}")

            recent = self._current["messages"][-3:]
            if recent:
                parts.append("\nПоследние сообщения:")
                for msg in recent:
                    preview = msg["user"][:60]
                    parts.append(f"  Пользователь: {preview}")

            return "\n".join(parts)

    def has_active_thread(self) -> bool:
        with self._lock:
            if not self._current:
                return False
            elapsed = (datetime.now(timezone.utc) - self._current["started"]).total_seconds()
            return elapsed <= self.timeout_secs

    def get_current_topic(self) -> Optional[str]:
        with self._lock:
            if self._current:
                return self._current["topic"]
            return None

    def get_past_threads(self, limit: int = 5) -> List[Tuple[str, float, int]]:
        with self._lock:
            return [
                (t["topic"], t["duration_secs"], t["message_count"])
                for t in self._history[-limit:]
            ]

    def end_thread(self):
        with self._lock:
            if self._current:
                self._archive_current()

    # Алиас для v4.0 совместимости
    def add_to_thread(self, user_input: str, response: str):
        self.add_message(user_input, response)

    # ── Внутренние ──

    def _archive_current(self):
        """Без блокировки — вызывается из-под self._lock"""
        if not self._current:
            return
        duration = (datetime.now(timezone.utc) - self._current["started"]).total_seconds()
        self._history.append({
            "topic": self._current["topic"],
            "duration_secs": duration,
            "message_count": len(self._current["messages"]),
        })
        if len(self._history) > 20:
            self._history = self._history[-20:]
        self._current = None
