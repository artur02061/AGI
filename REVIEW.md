# Code Review: Kristina 6.0 — Полный аудит кода

> Проанализировано ~5500 строк Python по 30+ файлам, включая fallback-реализации, утилиты, system control.

---

## 1. БАГИ (ломают выполнение)

### 1.1. `_save_to_vector_memory` — кумулятивные stats вместо per-request
**Файл:** `core/agent.py:429-433`
```python
importance = min(3, 1 + self.stats["total_tool_calls"])
'had_errors': self.stats["total_errors"] > 0,
```
`total_tool_calls` — глобальный счётчик за сессию. После 2-го tool call importance навсегда = 3. После первой ошибки все диалоги помечены `had_errors=True`. **Нужен per-request счётчик.**

### 1.2. Ротация эпизодов без учёта возраста
**Файл:** `core/memory.py:124-126` (dead code, но аналогичная проблема в `_fallback/memory_engine.py:184-195`)
```python
self.episodic.sort(key=lambda x: x['importance'])
self.episodic = self.episodic[100:]
```
Удаляет по importance без timestamp. Свежий эпизод с importance=1 удалится раньше старого с importance=2.

Та же проблема в fallback: `_evict_episodes()` сортирует по importance, игнорируя время.

### 1.3. `asyncio.get_event_loop()` deprecated
**Файлы:** `bridge.py:129`, `core/memory.py:110`

В Python 3.12+ выбрасывает `RuntimeError` без текущего loop. `asyncio.run()` внутри работающего loop — тоже `RuntimeError`. **Нужен `asyncio.get_running_loop()`.**

### 1.4. Неправильный strptime для ISO weeks
**Файл:** `modules/memory/summarizer.py:208, 346`
```python
dt = datetime.strptime(f"{year}-W{w:02d}-1", "%Y-W%W-%w")
```
`%W` — это week 0-53 (Monday-first), не ISO week. Для ISO нужен `%G-W%V-%u`. Даёт **неправильные даты** при суммаризации.

### 1.5. Sync `requests.get()` блокирует event loop
**Файлы:** `tools/web_tools.py:181,249,300` + `modules/rag/vector_store.py:285`

Все web tools используют синхронный `requests.get()` в async контексте. `httpx` есть в requirements но не используется. Sync `ollama.embeddings()` тоже блокирует. **Парализует все параллельные async tasks.**

### 1.6. Executor вызывает tools с positional args
**Файл:** `core/agents/executor.py:80`
```python
result = await tool(*args)
```
Tools ожидают kwargs (`query=`, `filepath=`). Positional работает случайно для 1-arg tools, но `list_directory(directory, pattern)` получит аргументы в неправильном порядке.

### 1.7. Два независимых embedding cache
**Файлы:** `bridge.py:85` (EmbeddingCacheAdapter) + `modules/rag/vector_store.py:96` (VectorMemory.embedding_cache)

Два кэша хранят одни и те же embedding'и, не знают друг о друге. Удвоение памяти, нет консистентности.

### 1.8. `AppFinder` крашится на Linux
**Файл:** `modules/system_control/app_finder.py:8`
```python
import winreg
```
`winreg` доступен только на Windows. На Linux/macOS при `import app_finder` — **сразу ImportError**. Это крашит `SystemController.__init__()` и весь `initialize_system()`.

### 1.9. `os.startfile()` только Windows
**Файл:** `modules/system_control/controller.py:450`
```python
os.startfile(path)
```
`os.startfile` существует только на Windows. На Linux — `AttributeError`.

### 1.10. `_is_running()` проверяет `.exe` suffix на любой ОС
**Файл:** `modules/system_control/controller.py:61`
```python
if self._is_running(app_name_clean + '.exe'):
```
Hardcoded `.exe` — на Linux процессы не имеют расширения.

### 1.11. `get_system_status()` hardcoded root disk
**Файл:** `modules/system_control/controller.py:268`
```python
disk = psutil.disk_usage('/')
```
На Windows корневой диск — `C:\`. `'/'` может не существовать.

### 1.12. `search_file()` итерирует Windows-диски `CDEFGH` на Linux
**Файл:** `modules/system_control/controller.py:349-352`
```python
for drive in "CDEFGH":
    drive_path = f"{drive}:/"
```
На Linux эти пути не существуют, и home directories (`/home/user/Documents`) не добавляются как fallback.

### 1.13. `_fallback/memory_engine.py` — keyword index corrupts after eviction
**Файл:** `_fallback/memory_engine.py:184-195`

После `_evict_episodes()` вызывается `_rebuild_index()`, но индексы в `_keyword_index` ссылаются на позиции в `_episodic` list. После `pop(idx)` все элементы после `idx` сдвигаются — **но между `pop` вызовами индексы не обновляются**. Удаление в обратном порядке (`reverse=True`) спасает от race, но `_rebuild_index()` вызывается один раз в конце — OK, это корректно. Однако **между eviction и rebuild**, если другой thread вызовет `get_relevant_context()`, он получит **stale индексы**.

### 1.14. `ThreadMemory.is_related_to_thread()` — ложные срабатывания
**Файл:** `modules/memory/thread_tracker.py:113-118`
```python
context_indicators = ["тот", "та", "то", "это", "об этом"]
if any(indicator in text_lower for indicator in context_indicators):
    return True
```
Слова "это", "то", "та" встречаются в 90%+ русских предложений. Практически **любое** сообщение считается related к текущей нити.

### 1.15. `EmbeddingCache` — MD5 collision risk для кэш-ключей
**Файлы:** `_fallback/embedding_cache.py:39`, `_fallback/memory_engine.py:198`

MD5 hash усечён до 8 hex символов (32 бита) для keyword index. При 10000+ записях вероятность коллизии > 1% (birthday paradox). Это приведёт к **неправильному поиску** в memory.

---

## 2. ПРОБЛЕМЫ АРХИТЕКТУРЫ

### 2.1. `MemorySystem` — мёртвый код
`core/memory.py` содержит полный класс `MemorySystem` (~300 строк) с summarization, auto-save, decay. **Никогда не инстанцируется.** `main.py` использует `MemoryAdapter` из `bridge.py`.

### 2.2. `identity.analyze_interaction()` никогда не вызывается
Логика эволюции личности (`core/identity.py:92`) существует, но не подключена к pipeline. Personality не развивается.

### 2.3. MetaCognition полностью отключена
`select_strategy()`, `record_strategy_outcome()`, `estimate_confidence()`, `record_outcome()` — ни один метод не вызывается из orchestrator/agent. UCB-алгоритм для выбора стратегии написан (~200 строк) но **никогда не используется**.

### 2.4. VAD не обновляет Identity
`VADEmotionalEngine` обновляется в `process_input()` (`main.py:284`), но `identity.update_mood()` **не вызывается** с результатом. `identity.current_mood` = `"neutral"` вечно.

### 2.5. `SelfAwareness` narrative не в промптах
`get_narrative_summary()` генерирует inner narrative, но он **никогда не инжектится в system prompt**. Вся self-awareness (~300 строк) — runtime dead code.

### 2.6. Orchestrator делает 3+ LLM вызова на запрос
`analyze_request()` → `execute()` (если primary=director, это ещё один LLM call) → `synthesize_response()`. Для простого "сколько времени?" — 3 LLM вызова вместо 1.

### 2.7. `_build_task` для executor теряет plan контекст
```python
if agent_name == "executor":
    return {"tool": None, "args": [], "user_input": user_input}
```
Director анализирует запрос, определяет нужный tool, но executor получает `tool=None` и вынужден переопределять tool самостоятельно.

### 2.8. Три дублирующих ThreadTracker'а
1. `_fallback/thread_tracker.py` — fallback Rust API
2. `modules/memory/thread_tracker.py` — Python `ThreadMemory`
3. `bridge.py:ThreadTrackerAdapter` — wrapper

Первые два никогда не используются напрямую (только через adapter), но содержат **разную логику** (разные context indicators, разные timeout checks). Если Rust недоступен, adapter оборачивает fallback, который дублирует `ThreadMemory` с другими деталями.

### 2.9. `emotion_analyzer` vs `vad_emotions` — две системы эмоций
`EmotionAnalyzer` (keyword matching) и `VADEmotionalEngine` (3D model) работают параллельно. Результат `EmotionAnalyzer` **выбрасывается** (`main.py:269` — `emotion = ...` не используется дальше). VAD обновляется, но не связан с identity. Нет единой эмоциональной системы.

### 2.10. `ToolCallParser` создаётся но не используется в v6.0
**Файл:** `main.py:95`
```python
tool_parser = ToolCallParser([])
```
v6.0 использует native Ollama tool calling. `ToolCallParser` (regex-based) больше не нужен. Он создаётся, ему передаются known_tools (`main.py:168`), но **ни один модуль не вызывает** `tool_parser.parse()`.

---

## 3. ЗАГЛУШКИ И МЁРТВЫЙ КОД

| Что | Где | Проблема |
|-----|-----|----------|
| `MemoryAdapter.load()` | `bridge.py:205` | Пустой метод |
| `ThreadTrackerAdapter.save()/load()` | `bridge.py:355-356` | Пустые, thread history теряется |
| `ContextCompressor` | `main.py:97` | Создаётся, никогда не используется |
| `EmotionAnalyzer` результат | `main.py:269` | `emotion = analyze(text)` — переменная не используется |
| `response_cache` | `agent.py:51` | Наивный MD5 cache без инвалидации, кэширует tool-call результаты |
| `AnalystAgent._web_search_and_analyze()` | `analyst.py:119-176` | Мёртвый метод, ~60 строк |
| `AnalystAgent._analyze_data()` | `analyst.py:178` | Сигнатура `Dict`, но получает `str` — упадёт |
| `MemorySystem` | `core/memory.py` | ~300 строк мёртвого кода |
| `parsers.py` | `utils/parsers.py` | Deprecated в v6.0, но файл остался |
| `_log_operation()` | `controller.py:470` | Метод никогда не вызывается из других методов controller |
| `GetCurrencyRateTool` | `web_tools.py:270` | Не зарегистрирован в `main.py` |
| `SearchAppsTool` | `system_tools.py:140` | Не зарегистрирован в `main.py` |
| `kill_process()` | `controller.py:198` | Метод существует, но **нет** `KillProcessTool` |
| `open_file()` | `controller.py:417` | Метод существует, но **нет** `OpenFileTool` |

---

## 4. КОСТЫЛИ

| Что | Где | Проблема |
|-----|-----|----------|
| 60+ UPPER_CASE маппингов через `__getattr__` | `config.py:334-340` | Невозможно grep-нуть использования, нет autocomplete |
| `_COMPUTED_MAP` с lambda | `config.py:324-331` | `"SYSTEM_MONITOR_INTERVAL": lambda: 30` — hardcoded в lambda |
| `_v4_current_thread` dual state | `bridge.py:284` | ThreadTracker хранит состояние дважды |
| kwargs→positional fallback | `agent.py:224-229` | Если kwargs не подошли — `tool(*args.values())` |
| Docstring v5.0 / Code v6.0 | `main.py:2` | `"""Кристина 5.0"""` но `version="6.0"` |
| DDGS dual-library | `web_tools.py:18-31` | `ddgs` + `duckduckgo_search` с разными API |
| ChromaDB triple fallback | `vector_store.py:32-68` | PersistentClient → Client → None |
| Pickle→JSON migration | `vector_store.py:314-324` | Одноразовый код навсегда |
| `Path.home() / "Desktop"` | `file_tools.py:140,268` | Не работает на Linux, русской Windows |
| `for drive in "CDEFGH"` | `controller.py:349` | Windows-only логика в универсальном коде |
| `app_name_clean + '.exe'` | `controller.py:61` | Windows-only в коде с `config.py` для Linux |
| `except:` (bare) ×15+ | `app_finder.py` | 15+ bare `except:` проглатывают все ошибки |
| `import requests` внутри методов | `web_tools.py:177,246,299` | Import внутри каждого вызова вместо top-level |

---

## 5. ПРОБЛЕМЫ БЕЗОПАСНОСТИ

### 5.1. `file_access_mode: "unrestricted"` по умолчанию
**Файл:** `config.py:112`
LLM может создать/удалить/перезаписать **любой** файл без подтверждения пользователя.

### 5.2. Нет подтверждения для деструктивных file operations
`CreateFileTool`, `WriteFileTool`, `DeleteFileTool` — LLM вызывает их самостоятельно через native tool calling. Нет промежуточного confirm.

### 5.3. `blocked_extensions` только Windows
```python
blocked_extensions: List[str] = [".sys", ".dll", ".exe", ".msi", ...]
```
Нет `.sh`, `.so`, `.py`, нет проверки `/etc/passwd`, `/etc/shadow`, `~/.ssh/`.

### 5.4. `subprocess.Popen(app['path'], shell=False)` без sanitization
**Файл:** `controller.py:71`
`app['path']` приходит из кэша, который заполняется из registry/filesystem scan. Если кэш corrupted или подменён — запуск произвольного executable.

### 5.5. `validate_url()` блокирует localhost но не private ranges
**Файл:** `validators.py:160-161`
Блокирует `localhost`, `127.0.0.1`, но **не** `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`. LLM может сканировать внутреннюю сеть через `web_fetch`.

### 5.6. `win32com.client.Dispatch` в `AppFinder`
**Файл:** `app_finder.py:344-346`
Использование COM Automation (WScript.Shell) для чтения .lnk файлов. Если .lnk crafted maliciously — потенциальный вектор.

---

## 6. ПРОБЛЕМЫ СОВМЕСТИМОСТИ

### 6.1. Windows-only код без guards
`app_finder.py`, `controller.py` содержат Windows-specific код (`winreg`, `os.startfile`, `.exe`, drive letters, `win32com.client`) без `platform.system()` проверок. На Linux весь system_control модуль **неработоспособен**.

### 6.2. `httpx` в requirements, `requests` в коде
`requirements.txt` рекомендует `httpx` как замену `requests`, но все web tools используют `requests`. Оба пакета нужны одновременно.

### 6.3. `structlog` в main.py, `logging` в utils/logging.py
Две конфликтующие системы логирования. `main.py` настраивает `structlog`, а `utils/logging.py` настраивает стандартный `logging` с `ColoredFormatter`. Child loggers через `get_logger()` используют `logging`, не `structlog`.

### 6.4. `send2trash` — optional без fallback
**Файл:** `file_tools.py:159`
`DeleteFileTool` требует `send2trash`, которого нет в `requirements.txt`. Если не установлен — **delete не работает**.

### 6.5. `pynvml` — optional без requirements
GPU monitoring в `controller.py:292` требует `pynvml`, которого нет в основных requirements (только в комментариях как optional).

---

## 7. RACE CONDITIONS И CONCURRENCY

### 7.1. `response_cache` не thread-safe
**Файл:** `agent.py:51`
`self.response_cache = {}` — обычный dict. При concurrent доступе (asyncio + threads) — возможна порча данных.

### 7.2. `ThreadTrackerAdapter._v4_current_thread` не синхронизирован
**Файл:** `bridge.py:284`
`_v4_current_thread` dict модифицируется без lock, хотя underlying `_RustThreadTracker` имеет threading.RLock.

### 7.3. `MemoryAdapter.add_episode()` — fire-and-forget async task
**Файл:** `bridge.py:131`
```python
asyncio.create_task(self.knowledge_graph.extract_and_add(...))
```
Task создаётся но **не отслеживается**. Если исключение — оно проглатывается. В Python 3.12+ untracked tasks генерируют warnings.

---

## 8. PERFORMANCE

### 8.1. `VectorMemory.search()` — O(n) linear scan
При каждом search ChromaDB делает embedding запрос (sync, блокирующий), потом linear scan по коллекции. Без batch embedding.

### 8.2. `KnowledgeGraph._find_duplicate()` — O(n) per insert
**Файл:** `knowledge_graph.py:85-88`
Линейный проход по всем триплетам на каждое добавление. При 15000 триплетов — заметное замедление.

### 8.3. `search_file()` — полный обход файловой системы
**Файл:** `controller.py:361`
`Path(search_path).walk()` обходит всё дерево каталогов. На системном диске с 500K+ файлов — несколько минут, блокируя event loop (sync операция).

### 8.4. `AppFinder.scan_system()` при первом запуске
Синхронный обход registry + Program Files + Start Menu + Desktop + Steam + Epic. На системе с 100+ программами — 10-30 секунд блокирующего IO при первом запуске.

---

## ИТОГОВАЯ СВОДКА

| Категория | Количество |
|-----------|-----------|
| Баги (крашат / дают неправильные результаты) | 15 |
| Архитектурные проблемы | 10 |
| Заглушки / мёртвый код | 14 |
| Костыли | 13 |
| Безопасность | 6 |
| Совместимость | 5 |
| Race conditions | 3 |
| Performance | 4 |
| **ИТОГО** | **70** |

### TOP-5 критичных для исправления:

1. **Windows-only system_control** — на Linux весь модуль крашится при импорте (`winreg`)
2. **Sync вызовы в async** — `requests.get()`, `ollama.embeddings()` блокируют event loop
3. **Модули не связаны** — MetaCognition, VAD, SelfAwareness, Identity, ContextCompressor инициализируются но не влияют на поведение
4. **Кумулятивные stats** — importance и had_errors некорректны после первого запроса
5. **ISO week parsing** — месячная суммаризация даёт неправильные даты
