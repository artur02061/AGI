"""
Thread Memory — отслеживание текущей темы разговора

v7.5: Семантический timeout вместо чисто временного.
Тема меняется когда:
  1. Прошло > 30 минут (было 10 минут) — мягкий fallback
  2. Новое сообщение семантически далеко от текущей темы
Это позволяет долгим разговорам сохранять контекст.
"""

from typing import List, Dict, Optional
from datetime import datetime
import threading

from utils.logging import get_logger
import config

logger = get_logger("thread_tracker")


class ThreadMemory:
    """Отслеживает текущую нить разговора (thread-safe)"""

    def __init__(self, sentence_encoder=None):
        self.current_thread = None
        self.thread_history = []
        # v7.5: Увеличен timeout с 600s (10 мин) до 1800s (30 мин)
        self.timeout = max(config.THREAD_TIMEOUT_SECONDS, 1800)
        self._sentence_encoder = sentence_encoder

        self._lock = threading.Lock()

        logger.info(f"✅ Thread Memory: timeout={self.timeout}s, семантический режим")
    
    def start_thread(self, topic: str, entities: List[str] = None):
        """Начинает новую нить (thread-safe)"""
        
        with self._lock:  # ✅ Защита от race conditions
            # Сохраняем предыдущую
            if self.current_thread:
                self._end_thread_unsafe()  # Вызываем unsafe версию, т.к. lock уже захвачен
            
            self.current_thread = {
                "topic": topic,
                "entities": entities or [],
                "started": datetime.now(),
                "messages": []
            }
            
            logger.info(f"🧵 Новая нить: {topic}")
    
    def add_to_thread(self, user_input: str, response: str):
        """Добавляет сообщение в нить (thread-safe)"""
        
        with self._lock:  # ✅ Защита от race conditions
            if self.current_thread:
                self.current_thread["messages"].append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now()
                })
                
                logger.debug(f"➕ Добавлено в нить (всего: {len(self.current_thread['messages'])})")
    
    def get_thread_context(self) -> Optional[str]:
        """Возвращает контекст нити (thread-safe)"""
        
        with self._lock:  # ✅ Защита от race conditions
            if not self.current_thread:
                return None
            
            # Проверяем timeout
            elapsed = (datetime.now() - self.current_thread["started"]).total_seconds()
            
            if elapsed > self.timeout:
                self._end_thread_unsafe()  # Unsafe версия, т.к. lock уже захвачен
                return None
            
            # Формируем контекст
            context = f"Текущая тема: {self.current_thread['topic']}\n"
            
            if self.current_thread['entities']:
                entities_str = ', '.join(self.current_thread['entities'][:5])
                context += f"Упоминается: {entities_str}\n"
            
            # Последние 3 сообщения
            recent = self.current_thread['messages'][-3:]
            if recent:
                context += "\nПоследние сообщения:\n"
                for msg in recent:
                    preview = msg['user'][:60]
                    context += f"  Пользователь: {preview}\n"
            
            return context
    
    def is_related_to_thread(self, text: str) -> bool:
        """Проверяет, относится ли текст к текущей нити (thread-safe)"""

        with self._lock:
            if not self.current_thread:
                return False

            # Проверяем timeout
            elapsed = (datetime.now() - self.current_thread["started"]).total_seconds()
            if elapsed > self.timeout:
                self._end_thread_unsafe()
                return False

            text_lower = text.lower()

            # Проверяем тему
            if self.current_thread['topic'].lower() in text_lower:
                return True

            # Проверяем сущности
            for entity in self.current_thread['entities']:
                if entity.lower() in text_lower:
                    return True

            # Проверяем контекстные указатели
            context_indicators = [
                "помнишь", "как мы говорили", "в той же теме",
                "продолжим", "вернёмся к", "насчёт того",
                "по поводу", "как я говорил", "об этом же",
            ]

            if any(indicator in text_lower for indicator in context_indicators):
                return True

            # v7.5: Семантическая проверка (если другие методы не сработали)
            if not self._is_topic_change(text):
                # Не смена темы = связано с текущей нитью
                return True

            return False
    
    def _end_thread_unsafe(self):
        """
        Завершает текущую нить БЕЗ захвата lock
        
        Используется внутри методов, где lock уже захвачен
        """
        
        if self.current_thread:
            duration = (datetime.now() - self.current_thread["started"]).total_seconds()
            
            self.current_thread["duration"] = duration
            self.current_thread["ended"] = datetime.now()
            self.current_thread["message_count"] = len(self.current_thread["messages"])
            
            self.thread_history.append(self.current_thread)
            
            # Ограничиваем историю
            if len(self.thread_history) > 20:
                self.thread_history = self.thread_history[-20:]
            
            logger.info(
                f"🧵 Нить завершена: {self.current_thread['topic']} "
                f"({duration:.0f}с, {self.current_thread['message_count']} сообщений)"
            )
            
            self.current_thread = None
    
    def _end_thread(self):
        """Публичный метод завершения нити (thread-safe)"""
        with self._lock:
            self._end_thread_unsafe()
    
    def get_past_threads(self, limit: int = 5) -> List[Dict]:
        """Возвращает прошлые нити (thread-safe)"""
        with self._lock:  # ✅ Защита от race conditions
            return self.thread_history[-limit:]

    def _is_topic_change(self, user_input: str) -> bool:
        """
        v7.5: Семантическая проверка смены темы.
        Возвращает True если новое сообщение — это другая тема.
        """
        if not self._sentence_encoder or not self.current_thread:
            return False

        try:
            # Кодируем новое сообщение
            new_vec = self._sentence_encoder(user_input)
            if not new_vec:
                return False

            # Кодируем текущую тему
            topic_text = self.current_thread.get('topic', '')
            # Берём последнее сообщение пользователя для сравнения
            messages = self.current_thread.get('messages', [])
            if messages:
                last_user_msg = messages[-1].get('user', topic_text)
                topic_vec = self._sentence_encoder(last_user_msg)
            else:
                topic_vec = self._sentence_encoder(topic_text)

            if not topic_vec:
                return False

            # Косинусное сходство
            import math
            dot = sum(a * b for a, b in zip(new_vec, topic_vec))
            norm1 = math.sqrt(sum(a * a for a in new_vec))
            norm2 = math.sqrt(sum(b * b for b in topic_vec))
            if norm1 < 1e-10 or norm2 < 1e-10:
                return False
            similarity = dot / (norm1 * norm2)

            # Порог: если сходство < 0.3 — это другая тема
            if similarity < 0.3:
                logger.info(
                    f"🔄 Семантическая смена темы: sim={similarity:.2f} < 0.3"
                )
                return True

        except Exception as e:
            logger.debug(f"Semantic topic check failed: {e}")

        return False

    def update(self, user_input: str, response: str):
        """
        Обновляет текущую нить (thread-safe).

        v7.5: Семантическая проверка смены темы +
              увеличенный timeout (30 минут вместо 10).
        """

        with self._lock:
            now = datetime.now()

            # Проверяем таймаут существующей нити
            if self.current_thread:
                last_message = self.current_thread['messages'][-1] if self.current_thread['messages'] else None

                should_end = False
                if last_message:
                    last_timestamp = last_message['timestamp']
                    elapsed = (now - last_timestamp).total_seconds()
                    if elapsed > self.timeout:
                        should_end = True

                # v7.5: Семантическая проверка (без lock, т.к. мы уже внутри)
                if not should_end and len(self.current_thread.get('messages', [])) >= 3:
                    if self._is_topic_change(user_input):
                        should_end = True

                if should_end:
                    self._end_thread_unsafe()

            # Создаём или обновляем нить
            if not self.current_thread:
                self.current_thread = {
                    'topic': user_input[:50],
                    'started': now,
                    'messages': [],
                    'entities': []
                }
                logger.info(f"🧵 Новая нить: {self.current_thread['topic']}")

            # Добавляем сообщение
            self.current_thread['messages'].append({
                'user': user_input,
                'assistant': response,
                'timestamp': now
            })
    
