"""
Thread Memory — отслеживание текущей темы разговора - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

from typing import List, Dict, Optional
from datetime import datetime
import threading  # ✅ ИСПРАВЛЕНИЕ: Добавлена защита от race conditions

from utils.logging import get_logger
import config

logger = get_logger("thread_tracker")

class ThreadMemory:
    """Отслеживает текущую нить разговора (thread-safe)"""
    
    def __init__(self):
        self.current_thread = None
        self.thread_history = []
        self.timeout = config.THREAD_TIMEOUT_SECONDS
        
        # ✅ ИСПРАВЛЕНИЕ: Добавлена блокировка для thread-safety
        self._lock = threading.Lock()
        
        logger.info("✅ Thread Memory инициализирована (thread-safe)")
    
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
        
        with self._lock:  # ✅ Защита от race conditions
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
            
            # Проверяем контекстные указатели (только фразы, не одиночные слова)
            context_indicators = [
                "помнишь", "как мы говорили", "в той же теме",
                "продолжим", "вернёмся к", "насчёт того",
                "по поводу", "как я говорил", "об этом же",
            ]

            if any(indicator in text_lower for indicator in context_indicators):
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

    def update(self, user_input: str, response: str):
        """
        Обновляет текущую нить (thread-safe)
        
        ✅ ИСПРАВЛЕНИЕ: Полностью переписан метод с защитой от race conditions
        """
        
        with self._lock:  # ✅ Защита от race conditions
            now = datetime.now()
        
            # Проверяем таймаут существующей нити
            if self.current_thread:
                last_message = self.current_thread['messages'][-1] if self.current_thread['messages'] else None
                
                if last_message:
                    last_timestamp = last_message['timestamp']
                    elapsed = (now - last_timestamp).total_seconds()
                    
                    if elapsed > self.timeout:
                        # Закрываем старую нить
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
    
    def _close_current_thread(self):
        """
        DEPRECATED: Используйте _end_thread() или _end_thread_unsafe()
        
        Оставлен для обратной совместимости
        """
        logger.warning("⚠️ _close_current_thread() deprecated, используйте _end_thread()")
        self._end_thread()
