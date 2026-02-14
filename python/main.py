"""
Кристина 6.0 — Multi-Agent AI Assistant (Hybrid Rust+Python)

Точка входа. Все CPU-интенсивные модули работают через Rust ядро (kristina_core).
Если Rust не собран — автоматический fallback на Python.
"""

import asyncio
import sys
import signal
from pathlib import Path

# Добавляем корневую директорию
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from utils.logging import get_logger

# ── Единое логирование через utils/logging ──
log = get_logger("main")

# ── Rust/Python bridge ──
from bridge import (
    RUST_AVAILABLE, MemoryEngine, EmbeddingCache,
    ThreadTracker,
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
        log.info(f"Shutdown signal: {name}")
        self.should_exit = True


# ═══════════════════════════════════════════════════════════════
#                    ИНИЦИАЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════

async def initialize_system():
    """Инициализирует все компоненты"""

    log.info(f"system_init: version=6.0, rust_core={RUST_AVAILABLE}")

    # ── Health check: Ollama ──
    try:
        from ollama import AsyncClient
        client = AsyncClient()
        await client.list()
        # Закрываем health-check клиент (предотвращает ResourceWarning)
        if hasattr(client, '_client') and client._client:
            await client._client.aclose()
        log.info("Ollama connected")
    except Exception as e:
        log.error(f"Ollama unavailable: {e}")
        print(f"\n❌ Ollama недоступна: {e}")
        print("   Запусти Ollama: ollama serve")
        sys.exit(1)

    # ── Rust/Hybrid компоненты ──
    memory = MemoryEngine(
        str(config.memory_dir),
        working_size=config.working_memory_size,
        max_episodic=config.max_episodic_memory,
    )
    log.info(f"MemoryEngine ready (backend={'rust' if RUST_AVAILABLE else 'python'})")

    embedding_cache = EmbeddingCache(
        str(config.data_dir),
        max_size=config.embedding_cache_max_size,
    )
    log.info(f"EmbeddingCache ready (size={embedding_cache.len()}, backend={'rust' if RUST_AVAILABLE else 'python'})")

    thread_tracker = ThreadTracker(timeout_secs=config.thread_timeout_seconds)

    # v6.0: Новые модули (consciousness + SIGMA)
    from core.emotions_vad import VADEmotionalEngine
    from core.metacognition import MetaCognition
    from core.self_awareness import SelfAwareness

    vad_emotions = VADEmotionalEngine()
    metacognition = MetaCognition()
    self_awareness = SelfAwareness()
    log.info("VAD+MetaCog+SelfAwareness ready")

    # ── Python компоненты (I/O-bound, Rust не нужен) ──
    from core.identity import IdentityEngine

    identity = IdentityEngine()

    # Системный контроллер
    from modules.system_control.controller import SystemController
    system_controller = SystemController()

    # ── Инструменты v6.1 (JARVIS Edition) ──

    # Веб
    from tools.web_tools import (
        WebSearchTool, WebFetchTool, DownloadFileTool,
        GetWeatherTool, GetCurrentTimeTool, GetCurrencyRateTool,
    )
    # Система
    from tools.system_tools import (
        SystemStatusTool, LaunchAppTool, ListProcessesTool, SearchAppsTool,
        KillProcessTool, OpenFileTool,
        SystemInfoTool, DiskUsageTool, NetworkInfoTool,
        RunCommandTool, ClipboardReadTool, ClipboardWriteTool, GetEnvTool,
    )
    # Файлы
    from tools.file_tools import (
        SearchFilesTool, ReadFileTool, DeleteFileTool,
        ListDirectoryTool, CreateFileTool, WriteFileTool,
        AppendFileTool, CopyFileTool, MoveFileTool, RenameFileTool,
        FileInfoTool, CreateDirectoryTool, ArchiveTool, ExtractArchiveTool,
    )
    # Память и заметки
    from tools.memory_tools import (
        RecallMemoryTool, SearchMemoryTool,
        SaveNoteTool, ListNotesTool, DeleteNoteTool, ReadNoteTool,
    )

    tools = {}

    # --- Веб ---
    for cls in [WebSearchTool, WebFetchTool, DownloadFileTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    # --- Время, погода, валюта ---
    for cls in [GetCurrentTimeTool, GetWeatherTool, GetCurrencyRateTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    # --- Система (с контроллером) ---
    for cls in [SystemStatusTool, LaunchAppTool, ListProcessesTool,
                SearchAppsTool, KillProcessTool, OpenFileTool]:
        t = cls(system_controller)
        tools[t.schema.name] = t.execute

    # --- Система (без контроллера) ---
    for cls in [SystemInfoTool, DiskUsageTool, NetworkInfoTool,
                RunCommandTool, ClipboardReadTool, ClipboardWriteTool, GetEnvTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    # --- Файлы ---
    for cls in [SearchFilesTool, ReadFileTool, DeleteFileTool,
                ListDirectoryTool, CreateFileTool, WriteFileTool,
                AppendFileTool, CopyFileTool, MoveFileTool, RenameFileTool,
                FileInfoTool, CreateDirectoryTool, ArchiveTool, ExtractArchiveTool]:
        t = cls(system_controller) if cls == SearchFilesTool else cls()
        tools[t.schema.name] = t.execute

    # --- Память ---
    from modules.rag.vector_store import VectorMemory
    vector_memory = VectorMemory(shared_embedding_cache=embedding_cache)

    for cls in [RecallMemoryTool, SearchMemoryTool]:
        t = cls(memory) if cls == RecallMemoryTool else cls(vector_memory)
        tools[t.schema.name] = t.execute

    # --- Заметки ---
    for cls in [SaveNoteTool, ListNotesTool, DeleteNoteTool, ReadNoteTool]:
        t = cls()
        tools[t.schema.name] = t.execute

    log.info(f"Tools registered: {len(tools)} (JARVIS Edition v6.1)")

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

        log.info("Orchestrator ready (4 agents)")
    else:
        from core.agent import AgentCore
        agent = AgentCore(
            tools=tools,
            memory=memory,
            identity=identity,
            vector_memory=vector_memory,
            thread_memory=thread_tracker,
        )

    # 2.5: Передаём consciousness-модули агенту для использования в промптах
    agent.vad_emotions = vad_emotions
    agent.self_awareness = self_awareness
    agent.metacognition = metacognition

    log.info("System ready")

    return {
        "agent": agent,
        "memory": memory,
        "identity": identity,
        "vad_emotions": vad_emotions,
        "metacognition": metacognition,
        "self_awareness": self_awareness,
        "vector_memory": vector_memory,
        "embedding_cache": embedding_cache,
        "system_controller": system_controller,
        "vram_manager": vram_manager,
        "thread_tracker": thread_tracker,
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

    # ── Обновление состояния ──
    components["identity"].increment_conversation_depth()

    # ── Обработка через агента ──
    had_errors = False
    try:
        response = await components["agent"].process(text)
    except Exception as exc:
        log.error(f"Process error: {exc}", exc_info=True)
        response = f"Произошла ошибка: {exc}"
        had_errors = True

    # ── v6.0: Обновляем VAD эмоции, self-awareness, metacognition ──
    vad = components.get("vad_emotions")
    if vad:
        vad.update_from_dialogue(text, response, had_errors=had_errors)
        # 2.4: VAD → Identity mood sync
        components["identity"].update_mood(vad.mood)

    # 2.2: Эволюция личности на основе взаимодействия
    components["identity"].analyze_interaction(text, response)

    sa = components.get("self_awareness")
    if sa:
        input_lower = text.lower()
        user_happy = any(w in input_lower for w in ["спасибо", "круто", "отлично", "молодец"])
        user_angry = any(w in input_lower for w in ["ошибка", "не работает", "бред", "опять"])
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
                user_input = (await asyncio.to_thread(input, "Ты: ")).strip()
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
                log.error(f"Main loop error: {exc}")
                print(f"❌ Ошибка: {exc}\n")

    finally:
        log.info("Shutdown started")

        # v7.4: Закрываем все async-соединения (предотвращает ResourceWarning)
        agent = components.get("agent")
        if agent and hasattr(agent, 'close'):
            try:
                await agent.close()
            except Exception as e:
                log.debug(f"Agent close error: {e}")

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

        log.info("Shutdown complete")
        print("\n💭 До встречи!")


if __name__ == "__main__":
    import platform
    
    # uvloop только для Linux/macOS
    if platform.system() != "Windows":
        try:
            import uvloop
            uvloop.install()
            log.info("uvloop installed")
        except ImportError:
            pass
    else:
        log.info("Platform: Windows, event_loop=standard asyncio")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n💭 Завершение...")