# Кристина 6.0 — Детальный Code Review

> Проведён полный аудит ~4700 строк Python: 30+ файлов, все модули, fallback-реализации, утилиты.
> Дата: 2026-02-12

---

## КРИТИЧЕСКИЕ БАГИ (Runtime Crashes)

### 1. `orchestrator.py:102-104` — TypeError при вызове DirectorAgent.execute()

**Severity: CRITICAL — Crash на fast path**

Fast path вызывает:
```python
final_response = await self.director.execute(
    {"tool": None, "args": [], "user_input": user_input},
    context=context,  # ← ОШИБКА
)
```

Но сигнатура `DirectorAgent.execute()` (`director.py:155`):
```python
async def execute(self, task: Dict[str, Any]) -> str:
```

Метод принимает **один** позиционный аргумент `task`. Ключевой аргумент `context=` **не существует** в сигнатуре. Вызов упадёт с `TypeError: execute() got an unexpected keyword argument 'context'`.

**Как должно быть:** контекст передаётся внутри dict:
```python
final_response = await self.director.execute(
    {"type": "general", "input": user_input, "context": context}
)
```

---

### 2. `orchestrator.py:133` — MetaCognition.record_outcome() вызывается с неверными аргументами

**Severity: CRITICAL — Crash при каждом успешном запросе**

Вызов в orchestrator:
```python
self.metacognition.record_outcome(user_input, final_response, elapsed)
```

Сигнатура метода (`metacognition.py:115`):
```python
def record_outcome(self, confidence: float, was_correct: bool, topic: str = ""):
```

Передаётся: `(str, str, float)` — ожидается: `(float, bool, str)`. Строка `user_input` будет интерпретирована как `confidence` (float), что вызовет ошибку при арифметических операциях в строке 127: `abs(c - (1.0 if correct else 0.0))` где `c` — строка.

---

### 3. `identity.py:82-90` — update_mood() никогда не принимает VAD-метки

**Severity: HIGH — Настроение никогда не обновляется**

`VADEmotionalEngine.mood` возвращает русские метки: `"паника"`, `"тревога"`, `"радость"`, `"спокойствие"`, и т.д.

Но `IdentityEngine.update_mood()`:
```python
valid_moods = ['happy', 'satisfied', 'neutral', 'curious', 'frustrated', 'tired']
if emotion in valid_moods:
    self.current_mood = emotion
```

Русские метки **никогда не совпадут** с английским списком. Настроение всегда остаётся `"neutral"` (значение по умолчанию). Вся VAD-система эмоций фактически не влияет на поведение бота.

---

### 4. `orchestrator.py:131-132` — record_strategy_outcome() с несовпадающими ключами

**Severity: MEDIUM — Стратегии никогда не записываются**

```python
strategy = plan.get("primary_agent", "director")  # → "director", "executor", "analyst", "reasoner"
self.metacognition.record_strategy_outcome(strategy, success=True)
```

Но `MetaCognition.strategies` (`metacognition.py:47`):
```python
self.strategies = ["direct", "tool_use", "web_search", "delegate", "creative"]
```

Имена агентов (`"director"`, `"executor"`) **не совпадают** со стратегиями (`"direct"`, `"tool_use"`). Метод проверяет `if strategy in self._strategy_history` — ключ не найдётся, запись не произойдёт. Вся UCB-система выбора стратегий бесполезна.

---

## БАГИ (Функциональные)

### 5. `app_finder.py:144` — Неверное сообщение об ошибке (copy-paste)

```python
# В методе _parse_desktop_file():
except Exception as e:
    logger.debug(f"Ошибка при поиске Epic Games: {e}")  # ← COPY-PASTE: это парсер .desktop файлов, не Epic Games
```

---

### 6. `vram_manager.py:88-120` — ensure_loaded() не проверяет модели

Метод должен проверять наличие моделей в Ollama, но:
1. Кэширует список моделей (строка 104-105)
2. **Никогда не проверяет**, есть ли модели агентов в этом кэше
3. Просто безусловно добавляет агентов в `loaded_models` (строка 115-116)

```python
for agent_name in agent_names:
    if agent_name not in self.loaded_models:
        self.loaded_models.add(agent_name)  # ← Добавляется без проверки
```

Вся логика кэширования моделей (`_model_cache`) мертва.

---

### 7. `main.py:305` — Блокирующий input() в async контексте

```python
user_input = input("Ты: ").strip()  # ← Блокирует event loop
```

`input()` — синхронная блокирующая функция. Внутри `async def main()` она блокирует весь event loop. Для CLI-демо это приемлемо, но если добавить фоновые задачи (суммаризация, мониторинг), они будут зависать.

---

### 8. `agent.py:371` — datetime.fromisoformat() без обработки ошибок

```python
age = (datetime.now() - datetime.fromisoformat(ts)).total_seconds() / 60
```

Если `ts` содержит невалидный формат (corrupted data, timezone-aware vs naive), метод упадёт с `ValueError`, и весь `_build_memory_context()` крашнется. Нет try/except вокруг.

---

### 9. `app_finder.py:26` — Относительный путь к кэшу

```python
self.app_cache_file = Path("data/app_cache.json")
```

Путь **относительный**. Если программа запущена не из `python/` директории, кэш создастся в неожиданном месте или не загрузится.

---

## МЁРТВЫЙ КОД

### 10. `core/emotions.py` — Файл-обёртка, ничем не используется

```python
from core.emotions_vad import VADEmotionalEngine as EmotionalIntelligence
__all__ = ["EmotionalIntelligence"]
```

Ни один файл в проекте не импортирует `EmotionalIntelligence`. Файл полностью мёртвый.

---

### 11. `core/context_compressor.py` — Аналогичная пустая обёртка

```python
from bridge import ContextCompressor
__all__ = ["ContextCompressor"]
```

`ContextCompressor` импортируется напрямую из `bridge.py` во всех местах. Этот файл не используется.

---

### 12. `director.py:15` — Неиспользуемый импорт lru_cache

```python
from functools import lru_cache  # ← Нигде не используется
```

---

### 13. `executor.py:7,111` — Дублирующийся импорт re

```python
# Строка 7:
import re
# Строка 111 (внутри метода):
import re  # ← Дублирующий импорт
```

---

### 14. `analyst.py:153-184` — Метод _summarize() никогда не вызывается

Метод `_summarize()` определён, но `execute()` не содержит ветки для `task_type == "summarization"`. Мёртвый код.

---

### 15. `core/brain/__init__.py` — Пустой модуль-заглушка

Директория `core/brain/` содержит только пустой `__init__.py`. Нигде не импортируется.

---

## КОСТЫЛИ И АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

### 16. `main.py:179-181` — Monkey-patching атрибутов вместо конструктора

```python
agent.vad_emotions = vad_emotions
agent.self_awareness = self_awareness
agent.metacognition = metacognition
```

Динамическое добавление атрибутов в runtime вместо передачи через `__init__()`. Это:
- Не видно в определении класса
- Не проверяется линтерами
- `hasattr()` проверки по всему коду (`agent.py:313`, `agent.py:320`)

---

### 17. `bridge.py:127-142` — Хрупкая async/sync граница в add_episode()

```python
try:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(
        self.knowledge_graph.extract_and_add(user_input, response, importance)
    )
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
except RuntimeError:
    # No running loop — sync fallback
    for s, p, o in self.knowledge_graph._regex_extract(user_input):
        self.knowledge_graph.add_triple(s, p, o, importance, "regex")
```

Костыль с callback для подавления предупреждений. Проблемы:
- `task.exception()` в callback не обрабатывает ошибку, просто возвращает её
- Fire-and-forget task может потеряться при shutdown
- Sync fallback вызывает приватный метод `_regex_extract`

---

### 18. `config.py:345-355` — Гигантский __getattr__ для обратной совместимости

50+ маппингов `UPPER_CASE → lower_case` через `__getattr__` уровня модуля. Это:
- Скрывает реальные зависимости от IDE
- Не работает с type checkers (mypy)
- Каждый `config.MODEL` проходит через цепочку `__getattr__ → _COMPAT_MAP → getattr(config, ...)` вместо прямого доступа

---

### 19. `orchestrator.py` vs `agent.py` — Дублирование логики

Оба класса (`AgentCore` и `Orchestrator`) содержат:
- Свой `process()` loop
- Свою логику сохранения в память (`_save_to_memory` / `_save_to_vector_memory`)
- Свою thread tracking логику
- Свой system prompt builder

При `multi_agent_enabled=True` используется Orchestrator, при `False` — AgentCore. Дублирование ~150 строк.

---

### 20. Threading locks в async-коде

`MemoryEngine`, `EmbeddingCache`, `ThreadTracker` (fallback) используют `threading.RLock()`, но всё приложение — asyncio-based. `threading.RLock` блокирует **весь event loop** при захвате, а не только текущую корутину. В production с параллельными корутинами это вызовет deadlocks.

---

## ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ

### 21. `memory_engine.py:210`, `embedding_cache.py:39` — MD5 для хеширования

```python
return int(hashlib.md5(word.encode()).hexdigest(), 16)
```

MD5 медленнее и тяжелее, чем нужно для hash map. Проект уже зависит от xxhash (через Rust), но Python fallback использует MD5. Для keyword index лучше подойдёт `hash()` или `xxhash`.

---

### 22. `web_tools.py:31` — rate_limiter никогда не очищается при создании WebSearchTool

```python
self.rate_limiter = []  # ← Определён, но никогда не используется
```

`WebSearchTool` определяет `self.rate_limiter`, но не использует его. Rate limiting реализован только в `WebFetchTool`.

---

### 23. `executor.py:90-101` — Опасный fallback при несовпадении аргументов

```python
except TypeError as e:
    try:
        if isinstance(args, dict):
            result = await tool(*args.values())  # ← Порядок значений не гарантирован
        else:
            result = await tool(**{f"arg{i}": v for i, v in enumerate(args)})  # ← Произвольные kwargs
```

При `TypeError` код пытается вызвать инструмент альтернативным способом:
- `*args.values()` — порядок значений dict **не гарантирован** (до Python 3.7 точно, после — insertion order, но это зависит от того, как LLM формирует JSON)
- `**{f"arg{i}": ...}` — создаёт произвольные kwargs (`arg0`, `arg1`), которые не совпадут с реальными параметрами

---

### 24. `vector_store.py` — ChromaDB PersistentClient создаётся без проверки версии

Если формат хранения ChromaDB изменится при обновлении пакета, база может не загрузиться. Нет миграции или проверки версии.

---

### 25. Энергия падает слишком медленно

`identity.py:78-79`:
```python
if self.conversation_depth % 10 == 0:
    self.energy_level = max(20, self.energy_level - config.ENERGY_DECAY_PER_MESSAGE)
```

`ENERGY_DECAY_PER_MESSAGE = 1`. Энергия падает на 1% каждые 10 сообщений. Со стартовых 100% до минимума 20% нужно **800 сообщений**. Система энергии фактически бесполезна.

---

## БЕЗОПАСНОСТЬ

### 26. SSRF-защита неполная в validate_url()

`validators.py:186-192`:
```python
try:
    ip = ipaddress.ip_address(host)
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return False, f"Доступ к приватным IP запрещён: {host}"
except ValueError:
    pass  # hostname, не IP — OK
```

Проблема: DNS rebinding не проверяется. Злоумышленник может создать домен, который резолвится в `127.0.0.1`, и SSRF-фильтр его пропустит. Однако для desktop-приложения это малорелевантно.

---

## ИТОГО

| Категория | Количество |
|-----------|-----------|
| Критические баги (crash) | 2 |
| Серьёзные баги (broken feature) | 2 |
| Функциональные баги | 5 |
| Мёртвый код | 6 |
| Костыли / архитектурные проблемы | 5 |
| Потенциальные проблемы | 5 |
| Безопасность | 1 |
| **Всего** | **26** |

### Приоритет исправлений:

1. **Немедленно:** #1 (TypeError в fast path), #2 (crash record_outcome)
2. **Высокий:** #3 (VAD → mood disconnect), #4 (strategy mismatch)
3. **Средний:** #5-#9 (функциональные баги)
4. **Низкий:** #10-#15 (мёртвый код), #16-#20 (рефакторинг)
