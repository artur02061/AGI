"""
Кристина 6.0 — Автономный агент с нативным tool calling

ИЗМЕНЕНИЯ v6.0:
- ✅ Нативный Ollama tool calling (tools= parameter)
- ✅ Убраны ACTION: / FINAL_ANSWER: / THOUGHT: маркеры
- ✅ Убран regex-парсинг tool_name("args")
- ✅ Убрана зависимость от utils/parsers.py
- ✅ Модель сама решает когда вызывать tool, когда отвечать
- ✅ Structured output — Ollama возвращает JSON tool_calls напрямую
"""

import asyncio
import hashlib
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from ollama import AsyncClient

from utils.logging import get_logger
import config

logger = get_logger("agent")


class AgentCore:
    """
    Автономный агент с нативным tool calling.

    Вместо ReAct-парсинга текста (ACTION: / FINAL_ANSWER:)
    используем Ollama tools= API — модель сама возвращает
    структурированные tool_calls или текстовый ответ.
    """

    def __init__(self, tools: Dict, memory, identity, vector_memory, thread_memory):
        self.client = AsyncClient()
        self.tools = tools  # {name: callable}
        self.memory = memory
        self.identity = identity
        self.vector_memory = vector_memory
        self.thread_memory = thread_memory
        # v6.0: Строим Ollama tool schemas из BaseTool.schema
        self._ollama_tools = self._build_ollama_tools()

        # Consciousness-модули (инициализируются через set_consciousness_modules)
        self.vad_emotions = None
        self.self_awareness = None
        self.metacognition = None

        # Статистика
        self.stats = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "total_errors": 0,
            "cache_hits": 0,
        }

        # Кэш ответов (asyncio-safe через Lock)
        self.response_cache = {} if config.RESPONSE_CACHE_ENABLED else None
        self._cache_lock = asyncio.Lock()

        logger.info(
            f"Агент v6.0 (native tool calling): "
            f"{len(self.tools)} инструментов, "
            f"{len(self._ollama_tools)} зарегистрировано"
        )

    # ═══════════════════════════════════════════════════════════
    #                    ГЛАВНЫЙ ЦИКЛ
    # ═══════════════════════════════════════════════════════════

    async def process(self, user_input: str) -> str:
        """Обрабатывает запрос пользователя"""

        self.stats["total_queries"] += 1
        self._request_tool_calls = 0
        self._request_errors = 0
        logger.info(f"Запрос: {user_input[:50]}...")

        self._current_query = user_input

        # Кэш
        if self.response_cache is not None:
            cached = await self._check_cache(user_input)
            if cached:
                self.stats["cache_hits"] += 1
                return cached

        # Thread
        self._update_thread_context(user_input)

        # === НАТИВНЫЙ TOOL CALLING LOOP ===
        final_response = await self._tool_calling_loop(user_input)

        # Сохранение
        if self.response_cache is not None:
            await self._save_to_cache(user_input, final_response)

        if self.thread_memory.current_thread:
            self.thread_memory.add_to_thread(user_input, final_response)

        self._save_to_vector_memory(user_input, final_response)

        return final_response

    async def _tool_calling_loop(self, user_input: str) -> str:
        """
        v6.0: Нативный tool calling loop.

        Вместо парсинга ACTION:/FINAL_ANSWER: из текста:
        1. Отправляем сообщение + tools= в Ollama
        2. Если модель вернула tool_calls → выполняем, отправляем результат обратно
        3. Если модель вернула текст → это финальный ответ
        4. Повторяем до max_iterations
        """

        system_prompt = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        error_count = 0
        tool_call_count = 0

        for iteration in range(config.AGENT_MAX_ITERATIONS):
            logger.debug(f"Итерация {iteration + 1}/{config.AGENT_MAX_ITERATIONS}")

            try:
                response = await self.client.chat(
                    model=config.MODEL,
                    messages=messages,
                    tools=self._ollama_tools if self._ollama_tools else None,
                    options={
                        "temperature": config.TEMPERATURE,
                        "num_ctx": config.CONTEXT_WINDOW,
                        "num_predict": config.MAX_RESPONSE_TOKENS,
                    },
                )
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                return f"Произошла ошибка: {e}"

            message = response["message"]

            # ── Модель вызвала инструменты? ──
            tool_calls = message.get("tool_calls")

            if tool_calls:
                # Добавляем ответ модели в историю
                messages.append(message)

                # Выполняем каждый tool call
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"].get("arguments", {})

                    logger.info(f"🔧 Tool call: {func_name}({func_args})")

                    result = await self._execute_tool(func_name, func_args)

                    if config.LOG_TOOL_CALLS:
                        logger.info(f"📊 Результат: {result[:100]}...")

                    # Добавляем результат как tool response
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                    })

                    tool_call_count += 1
                    self.stats["total_tool_calls"] += 1
                    self._request_tool_calls += 1

                    # Проверка ошибок
                    if "ERROR" in str(result):
                        error_count += 1
                        self.stats["total_errors"] += 1
                        self._request_errors += 1

                        if error_count >= config.AGENT_MAX_ERRORS:
                            logger.warning(f"⚠️ Лимит ошибок ({config.AGENT_MAX_ERRORS})")
                            return (
                                "Извини, возникли проблемы с выполнением. "
                                "Попробуй переформулировать запрос."
                            )

                # Продолжаем цикл — модель увидит результаты и решит что дальше
                continue

            else:
                # ── Модель вернула текстовый ответ → это финальный ответ ──
                final_text = message.get("content", "").strip()

                if final_text:
                    logger.info(
                        f"✅ Ответ за {iteration + 1} итераций, "
                        f"{tool_call_count} tool calls"
                    )
                    return final_text
                else:
                    # Пустой ответ — подталкиваем
                    messages.append(message)
                    messages.append({
                        "role": "user",
                        "content": "Пожалуйста, дай ответ на русском языке.",
                    })
                    continue

        # Лимит итераций
        logger.warning("⚠️ Достигнут лимит итераций")
        return "Хм, я немного запуталась. Можешь переформулировать проще?"

    # ═══════════════════════════════════════════════════════════
    #                  ВЫПОЛНЕНИЕ ИНСТРУМЕНТОВ
    # ═══════════════════════════════════════════════════════════

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """
        v6.0: Выполняет tool call по имени и kwargs.
        Больше не нужен parse_tool_call / shlex / regex!
        """

        if name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"ERROR: Инструмент '{name}' не найден. Доступны: {available}"

        tool_func = self.tools[name]

        try:
            # Ollama native tool calling передаёт args как dict → kwargs
            if isinstance(args, dict):
                result = await tool_func(**args)
            else:
                result = await tool_func(*args)
            return str(result)
        except Exception as e:
            logger.error(f"Tool error {name}(args={args}): {e}", exc_info=True)
            return f"ERROR: {e}"

    # ═══════════════════════════════════════════════════════════
    #                  OLLAMA TOOL SCHEMAS
    # ═══════════════════════════════════════════════════════════

    def _build_ollama_tools(self) -> List[Dict]:
        """
        v6.0: Конвертирует все инструменты в Ollama tool format.

        Результат:
        [
            {"type": "function", "function": {"name": "web_search", ...}},
            ...
        ]
        """
        ollama_tools = []

        for name, tool_func in self.tools.items():
            # Пробуем получить schema из BaseTool
            tool_obj = None

            # tool_func может быть bound method (tool.execute)
            if hasattr(tool_func, "__self__"):
                tool_obj = tool_func.__self__

            if tool_obj and hasattr(tool_obj, "schema"):
                schema = tool_obj.schema
                ollama_tool = schema.to_ollama_tool()
                ollama_tools.append(ollama_tool)
            else:
                # Fallback: генерируем минимальную schema
                logger.warning(f"⚠️ {name}: нет schema, генерируем автоматически")
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Инструмент {name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                })

        logger.info(f"🔧 Зарегистрировано {len(ollama_tools)} Ollama tools")
        return ollama_tools

    # ═══════════════════════════════════════════════════════════
    #                  СИСТЕМНЫЙ ПРОМПТ
    # ═══════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        """
        v6.0: Упрощённый промпт.
        Не нужно объяснять формат ACTION:/FINAL_ANSWER: —
        модель сама вызывает tools через native API.
        """

        # Контекст памяти
        memory_context = self._build_memory_context()

        # Thread контекст
        thread_str = ""
        if self.thread_memory.current_thread:
            thread = self.thread_memory.current_thread
            thread_str = f"🧵 Текущая тема: {thread['topic']}"
            if thread.get('entities'):
                thread_str += f"\n   Упоминается: {', '.join(thread['entities'][:3])}"

        # Принудительный русский
        russian_instruction = ""
        if config.FORCE_RUSSIAN_ONLY:
            russian_instruction = (
                "⚠️ КРИТИЧЕСКИ ВАЖНО: ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ!\n"
            )

        # Self-awareness narrative (2.5)
        awareness_str = ""
        if hasattr(self, 'self_awareness') and self.self_awareness:
            narrative = self.self_awareness.get_narrative_summary()
            if narrative:
                awareness_str = f"\n{narrative}"

        # VAD emotional style
        vad_str = ""
        if hasattr(self, 'vad_emotions') and self.vad_emotions:
            style = self.vad_emotions.get_response_style()
            vad_str = f"\nТон: {style.get('tone', 'neutral')} | Эмоция: {style.get('emotional_label', 'нейтрально')}"

        prompt = f"""{russian_instruction}Ты — Кристина, AI-ассистент.

{thread_str}

{memory_context}
{awareness_str}

ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском
2. НЕ придумывай данные — используй инструменты для поиска
3. web_search даёт только URLs — используй web_fetch для чтения
4. Для файлов используй ПОЛНЫЕ пути
5. Отвечай кратко и по делу (3-5 предложений)

Настроение: {self.identity.current_mood} | Энергия: {self.identity.energy_level}%{vad_str}
"""

        return prompt

    def _build_memory_context(self) -> str:
        """Контекст из памяти"""

        lines = []

        # Рабочая память
        if self.memory.working:
            recent = []
            for msg in self.memory.working[-5:]:
                content = msg['content'][:80]
                recent.append(f"{msg['role']}: {content}")

            if recent:
                lines.append("⏱️ ТЕКУЩАЯ СЕССИЯ:")
                lines.extend(recent)

        # Векторная память
        if hasattr(self, '_current_query'):
            results = self.vector_memory.search(
                self._current_query,
                n_results=2,
                filter_metadata={"type": "dialogue"},
            )

            old_memories = []
            for r in results:
                ts = r['metadata'].get('timestamp', '')
                if ts:
                    try:
                        age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 60
                    except (ValueError, TypeError):
                        age = config.VECTOR_MIN_AGE_MINUTES + 1  # считаем старым
                    if age > config.VECTOR_MIN_AGE_MINUTES:
                        date = r['metadata'].get('date', '')
                        text = r['text'][:60]
                        old_memories.append(f"[{date}] {text}")

            if old_memories:
                lines.append("\n📚 Из долговременной памяти:")
                lines.extend(old_memories)

        return "\n".join(lines) if lines else ""

    # ═══════════════════════════════════════════════════════════
    #                  THREAD / CACHE / STATS
    # ═══════════════════════════════════════════════════════════

    def _update_thread_context(self, user_input: str):
        if not self.thread_memory.is_related_to_thread(user_input):
            topic = self._detect_topic(user_input)
            entities = self._extract_entities(user_input)
            if topic:
                self.thread_memory.start_thread(topic, entities)

    def _detect_topic(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        starters = {
            "давай поговорим о": "обсуждение",
            "расскажи о": "информация",
            "помоги с": "помощь",
            "я работаю над": "проект",
            "найди": "поиск",
        }
        for starter, topic_type in starters.items():
            if starter in text_lower:
                after = text_lower.split(starter)[1].split('.')[0].split('?')[0].strip()
                return f"{topic_type}: {after[:50]}"
        return text[:50] if len(text.split()) > 3 else None

    def _extract_entities(self, text: str) -> List[str]:
        return list(set(re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', text)[:3]))

    def _make_cache_key(self, query: str) -> str:
        """Кэш-ключ без коллизий (hashlib вместо truncation)"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _check_cache(self, query: str) -> Optional[str]:
        async with self._cache_lock:
            cache_key = self._make_cache_key(query)
            if cache_key in self.response_cache:
                cached = self.response_cache[cache_key]
                age = (datetime.now() - cached['timestamp']).total_seconds()
                if age < config.AGENT_CACHE_TTL:
                    return cached['response']
                del self.response_cache[cache_key]
            return None

    async def _save_to_cache(self, query: str, response: str):
        async with self._cache_lock:
            cache_key = self._make_cache_key(query)
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': datetime.now(),
            }
            if len(self.response_cache) > 100:
                sorted_items = sorted(
                    self.response_cache.items(),
                    key=lambda x: x[1]['timestamp'],
                )
                for key, _ in sorted_items[:20]:
                    del self.response_cache[key]

    def _save_to_vector_memory(self, user_input: str, response: str):
        importance = min(3, 1 + self._request_tool_calls)
        metadata = {
            'tool_calls': self._request_tool_calls,
            'had_errors': self._request_errors > 0,
        }
        if self.thread_memory.current_thread:
            metadata['thread'] = self.thread_memory.current_thread['topic']

        self.vector_memory.add_dialogue(
            user_input=user_input,
            assistant_response=response,
            importance=importance,
            metadata=metadata,
        )

    def get_stats(self) -> Dict:
        return self.stats.copy()

    def reset_stats(self):
        self.stats = {
            "total_queries": 0,
            "total_tool_calls": 0,
            "total_errors": 0,
            "cache_hits": 0,
        }
