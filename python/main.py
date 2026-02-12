"""
Кристина 5.0 — Multi-Agent AI Assistant (Hybrid Rust+Python)

Точка входа. Все CPU-интенсивные модули работают через Rust ядро (kristina_core).
Если Rust не собран — автоматический fallback на Python.
"""

import asyncio
import sys
import signal
from pathlib import Path

# Добавляем корневую директорию
sys.path.insert(0, str(Path(__file__).parent))

import structlog
from config import config

# ── Structured logging ──
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger("kristina")

# ── Rust/Python bridge ──
from bridge import (
    RUST_AVAILABLE, MemoryEngine, EmbeddingCache, EmotionAnalyzer,
    ToolCallParser, ContextCompressor, ThreadTracker,
)


# ═══════════════════════════════════════════════════════════════
#                    GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════════════════════

class GracefulShutdown:
    """Обработчик сигналов для корректного завершения"""

    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        log.info("shutdown_signal", signal=name)
        self.should_exit = True


# ═══════════════════════════════════════════════════════════════
#                    ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

async def initialize_system():
    """Инициализирует все компоненты"""

    log.info("system_init", version="6.0", rust_core=RUST_AVAILABLE)

    # ── Health check: Ollama ──
    try:
        from ollama import AsyncClient
        client = AsyncClient()
        await client.list()
        log.info("ollama_connected")
    except Exception as e:
        log.error("ollama_unavailable", error=str(e))
        print(f"\n❌ Ollama недоступна: {e}")
        print("   Запусти Ollama: ollama serve")
        sys.exit(1)

    # ── Rust/Hybrid компоненты ──
    memory = MemoryEngine(
        str(config.memory_dir),
        working_size=config.working_memory_size,
        max_episodic=config.max_episodic_memory,
    )
    log.info("component_ready", name="MemoryEngine", backend="rust" if RUST_AVAILABLE else "python")

    embedding_cache = EmbeddingCache(
        str(config.data_dir),
        max_size=config.embedding_cache_max_size,
    )
    log.info("component_ready", name="EmbeddingCache",
             size=embedding_cache.len(),
             backend="rust" if RUST_AVAILABLE else "python")

    emotion_analyzer = EmotionAnalyzer()

    tool_parser = ToolCallParser([])  # Заполнится после регистрации tools

    context_compressor = ContextCompressor(compression_ratio=0.3)

    thread_tracker = ThreadTracker(timeout_secs=config.thread_timeout_seconds)

    # v6.0: Новые модули (consciousness + SIGMA)
    from core.emotions_vad import VADEmotionalEngine
    from core.metacognition import MetaCognition
    from core.self_awareness import SelfAwareness

    vad_emotions = VADEmotionalEngine()
    metacognition = MetaCognition()
    self_awareness = SelfAwareness()
    log.info("component_ready", name="VAD+MetaCog+SelfAwareness", source="consciousness+sigma")

    # ── Python компоненты (I/O-bound, Rust не нужен) ──
    from core.identity import IdentityEngine

    identity = IdentityEngine()

    # Системный контроллер
    from modules.system_control.controller import SystemController
    system_controller = SystemController()

    # ── Инструменты ──
    # Веб (web_tools.py — рабочие реализации, без дубликатов)
    from tools.web_tools import WebSearchTool, WebFetchTool, GetWeatherTool, GetCurrentTimeTool
    # Система (без дубликатов GetWeatherTool/GetCurrentTimeTool)
    from tools.system_tools import (
        SystemStatusTool, LaunchAppTool, ListProcessesTool,
    )
    from tools.file_tools import (
        SearchFilesTool, ReadFileTool, DeleteFileTool,
        ListDirectoryTool, CreateFileTool, WriteFileTool,
    )
    from tools.memory_tools import RecallMemoryTool, SearchMemoryTool

    tools = {}

    # Веб
    for cls in [WebSearchTool, WebFetchTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    # Система
    for cls, needs_ctrl in [
        (SystemStatusTool, True), (LaunchAppTool, True),
        (ListProcessesTool, True),
    ]:
        t = cls(system_controller)
        tools[t.schema.name] = t.execute

    # Время и погода (из web_tools — рабочие реализации)
    for cls in [GetCurrentTimeTool, GetWeatherTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    # Файлы
    for cls in [SearchFilesTool, ReadFileTool, DeleteFileTool,
                ListDirectoryTool, CreateFileTool, WriteFileTool]:
        t = cls(system_controller) if cls == SearchFilesTool else cls()
        tools[t.schema.name] = t.execute

    # Память
    from modules.rag.vector_store import VectorMemory
    vector_memory = VectorMemory()

    for cls in [RecallMemoryTool, SearchMemoryTool]:
        t = cls(memory) if cls == RecallMemoryTool else cls(vector_memory)
        tools[t.schema.name] = t.execute

    # Обновляем парсер с известными инструментами
    tool_parser.set_known_tools(list(tools.keys()))

    log.info("tools_registered", count=len(tools))

    # ── Агент / Оркестратор ──
    vram_manager = None

    if config.multi_agent_enabled:
        from core.orchestrator import Orchestrator

        agent = Orchestrator(
            tools=tools,
            memory=memory,
            identity=identity,
            vector_memory=vector_memory,
            thread_memory=thread_tracker,
        )

        vram_manager = getattr(agent, 'vram_manager', None)

        log.info("orchestrator_ready", agents=4)
    else:
        from core.agent import AgentCore
        agent = AgentCore(
            tools=tools,
            memory=memory,
            identity=identity,
            vector_memory=vector_memory,
            thread_memory=thread_tracker,
        )

    log.info("system_ready")

    return {
        "agent": agent,
        "memory": memory,
        "identity": identity,
        "emotions": emotion_analyzer,
        "vad_emotions": vad_emotions,
        "metacognition": metacognition,
        "self_awareness": self_awareness,
        "vector_memory": vector_memory,
        "embedding_cache": embedding_cache,
        "system_controller": system_controller,
        "vram_manager": vram_manager,
        "thread_tracker": thread_tracker,
        "context_compressor": context_compressor,
    }


# ═══════════════════════════════════════════════════════════════
#                    ОБРАБОТКА ЗАПРОСОВ
# ═══════════════════════════════════════════════════════════════

async def process_input(user_input: str, components: dict) -> str:
    """Обрабатывает ввод пользователя"""

    text = user_input.strip()
    text_lower = text.lower()

    # ── Спецкоманды ──
    if text_lower in ("выход", "exit", "quit", "пока"):
        print("\n💭 Кристина: Пока! Было приятно пообщаться.")
        return "EXIT"

    if text_lower in ("статус", "status"):
        mem_stats = components["memory"].get_stats()
        w = mem_stats.get("working", 0)
        e = mem_stats.get("episodic", 0)
        s = mem_stats.get("semantic_keys", 0)

        status = f"📊 Память: рабочая {w} | эпизоды {e} | факты {s}"

        # v6.0: VAD + Self-Awareness
        vad = components.get("vad_emotions")
        if vad:
            vs = vad.state
            status += f"\n🎭 Эмоции: {vs.label} (V:{vs.valence:.2f} A:{vs.arousal:.2f} D:{vs.dominance:.2f})"

        sa = components.get("self_awareness")
        if sa:
            status += f"\n🧠 {sa.get_self_description()}"

        mc = components.get("metacognition")
        if mc:
            intro = mc.introspect()
            status += f"\n🔍 Калибровка: {intro['calibration']['calibration_error']:.2f} | Неизвестных тем: {intro['known_unknowns']['count']}"

        return status

    if text_lower in ("очистить память", "clear memory"):
        components["memory"].clear_working()
        return "✅ Рабочая память очищена!"

    if text_lower in ("помощь", "help"):
        return (
            "📖 Команды: статус, очистить память, помощь, выход\n"
            "💡 Примеры: «удали файл», «запусти Chrome», «найди информацию», «сколько времени»"
        )

    # ── Эмоция ──
    emotion = components["emotions"].analyze(text)
    components["identity"].increment_conversation_depth()

    # ── Обработка через агента ──
    had_errors = False
    try:
        response = await components["agent"].process(text)
    except Exception as exc:
        log.error("process_error", error=str(exc), exc_info=True)
        response = f"Произошла ошибка: {exc}"
        had_errors = True

    # ── v6.0: Обновляем VAD эмоции, self-awareness, metacognition ──
    vad = components.get("vad_emotions")
    if vad:
        vad.update_from_dialogue(text, response, had_errors=had_errors)

    sa = components.get("self_awareness")
    if sa:
        user_happy = any(w in text.lower() for w in ["спасибо", "круто", "отлично", "молодец"])
        user_angry = any(w in text.lower() for w in ["ошибка", "не работает", "бред", "опять"])
        sa.update(
            valence=vad.state.valence if vad else 0.0,
            had_errors=had_errors,
            user_expressed_satisfaction=user_happy,
            user_expressed_frustration=user_angry,
        )

    mc = components.get("metacognition")
    if mc:
        mc.update_agency(not had_errors)

    return response


# ═══════════════════════════════════════════════════════════════
#                    ГЛАВНЫЙ ЦИКЛ
# ═══════════════════════════════════════════════════════════════

async def main():
    shutdown = GracefulShutdown()
    components = await initialize_system()

    print(f"\n{'=' * 60}")
    print(f"💬 Кристина 6.0 {'🦀 Rust' if RUST_AVAILABLE else '🐍 Python'} | Введи 'помощь'")
    print(f"{'=' * 60}\n")

    try:
        while not shutdown.should_exit:
            try:
                user_input = input("Ты: ").strip()
                if not user_input:
                    continue

                response = await process_input(user_input, components)
                if response == "EXIT":
                    break

                print(f"Кристина: {response}\n")

            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as exc:
                log.error("main_loop_error", error=str(exc))
                print(f"❌ Ошибка: {exc}\n")

    finally:
        log.info("shutdown_start")

        # Сохранение
        components["memory"].save()
        components["embedding_cache"].save()

        # v6.0: сохраняем кэш векторной памяти
        if hasattr(components.get("vector_memory"), "save_cache"):
            components["vector_memory"].save_cache()

        # v6.0: сохраняем metacognition
        if components.get("metacognition"):
            components["metacognition"].save()

        if components["vram_manager"]:
            components["vram_manager"].cleanup()

        log.info("shutdown_complete")
        print("\n💭 До встречи!")


if __name__ == "__main__":
    import platform
    
    # uvloop только для Linux/macOS
    if platform.system() != "Windows":
        try:
            import uvloop
            uvloop.install()
            log.info("uvloop_installed")
        except ImportError:
            pass
    else:
        log.info("platform", os="Windows", event_loop="standard asyncio")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n💭 Завершение...")