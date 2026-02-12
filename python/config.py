"""
Конфигурация Кристины 6.0 — Pydantic Settings

ИЗМЕНЕНИЯ v6.0:
- Embedding: nomic-embed-text → bge-m3 (100+ языков, 1024 dim, 8192 tokens)
- Иерархическая память (daily/weekly/monthly summaries)
- Knowledge Graph настройки
- Рекомендации по моделям в комментариях
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


# ═══════════════════════════════════════════════════════════════
#                       TYPED CONFIG
# ═══════════════════════════════════════════════════════════════

class OllamaHost(BaseSettings):
    gpu: str = "http://localhost:11434"
    cpu: str = "http://localhost:11435"


class AgentModelConfig(BaseSettings):
    name: str
    device: str = "gpu"
    host: str = "http://localhost:11434"
    vram_mb: int = 0
    priority: int = 0
    timeout: int = 30


class KristinaConfig(BaseSettings):
    """Единая типизированная конфигурация"""

    model_config = {"env_prefix": "KRISTINA_", "env_nested_delimiter": "__"}

    # ── Базовые пути ──
    base_dir: Path = Path(__file__).parent
    data_dir: Optional[Path] = None
    memory_dir: Optional[Path] = None
    vector_db_dir: Optional[Path] = None
    knowledge_graph_dir: Optional[Path] = None  # v6.0

    # ── Модели ──
    # Основная: gemma3:12b — хороша для русского, помещается в 8GB VRAM
    # Альтернативы: qwen3:14b (лучше tool calling), llama4-scout:17b (лучший мультиагент)
    model: str = "gemma3:12b"

    # Роутер: лёгкая модель для классификации запросов
    # Альтернатива: qwen3:1.7b (быстрее, но хуже русский)
    router_model: str = "gemma3:4b"

    # v6.0: bge-m3 — мультиязычный, 100+ языков, 1024 dim, 8192 tokens
    # Предыдущая: nomic-embed-text (768 dim, слабый русский)
    # Альтернатива: snowflake-arctic-embed-m-v2.0 (768 dim, хороший EN)
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024  # v6.0: bge-m3 выдаёт 1024

    # ── Генерация ──
    temperature: float = 0.7
    context_window: int = 16384
    max_response_tokens: int = 1024

    # ── Агент / ReAct ──
    agent_max_iterations: int = 6
    agent_max_errors: int = 3
    agent_cache_ttl: int = 300
    force_russian_only: bool = True

    # ── Память ──
    working_memory_size: int = 15
    short_term_memory_size: int = 50
    max_episodic_memory: int = 2000
    vector_search_results: int = 5
    vector_min_age_minutes: int = 30
    thread_timeout_seconds: int = 600

    # v6.0: Иерархическая суммаризация
    memory_summarize_enabled: bool = True
    # После скольки raw-эпизодов запускать daily summary
    memory_daily_threshold: int = 30
    # Модель для суммаризации (лёгкая, быстрая)
    memory_summarizer_model: str = "gemma3:4b"
    # Максимум raw-эпизодов хранить (старые → суммари)
    memory_max_raw_episodes: int = 200
    # Сколько дней хранить daily summaries перед → weekly
    memory_daily_retention_days: int = 14
    # Сколько недель хранить weekly перед → monthly
    memory_weekly_retention_weeks: int = 8

    # v6.0: Knowledge Graph
    knowledge_graph_enabled: bool = True
    # Максимум узлов в графе
    knowledge_graph_max_nodes: int = 5000
    # Максимум рёбер
    knowledge_graph_max_edges: int = 15000
    # Модель для извлечения фактов
    knowledge_graph_extractor_model: str = "gemma3:4b"
    # Минимальная важность для сохранения факта (1-5)
    knowledge_graph_min_importance: int = 2

    # ── Web tools ──
    web_search_max_results: int = 5
    web_fetch_max_size: int = 5000
    web_request_timeout: int = 10
    web_rate_limit: int = 10

    # ── Файловая система ──
    file_access_mode: str = "safe"
    file_search_max_depth: int = 4
    file_search_max_results: int = 10
    blocked_directories: List[Path] = [
        Path("/etc"),
        Path("/boot"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
        Path("C:/Windows/System32"),
    ]
    allowed_directories: List[Path] = []
    blocked_extensions: List[str] = [
        # Windows
        ".sys", ".dll", ".exe", ".msi", ".bat",
        ".cmd", ".ps1", ".vbs", ".reg", ".scr",
        # Linux/macOS
        ".so", ".ko", ".deb", ".rpm",
    ]
    protected_processes: List[str] = [
        # Windows
        "System", "csrss.exe", "wininit.exe",
        "services.exe", "lsass.exe", "dwm.exe", "explorer.exe",
        # Linux
        "systemd", "init", "kthreadd", "sshd",
    ]

    # ── Личность ──
    default_mood: str = "neutral"
    initial_energy: int = 100
    energy_decay_per_message: int = 1

    # ── Кэширование ──
    embedding_cache_enabled: bool = True
    embedding_cache_max_size: int = 20000
    response_cache_enabled: bool = True
    response_cache_ttl: int = 300

    # ── Multi-Agent ──
    multi_agent_enabled: bool = True
    hybrid_mode: bool = True
    max_vram_gb: float = 8.0
    gpu_vram_reserved: float = 5.5
    enable_parallel_execution: bool = True
    max_parallel_agents: int = 3
    hot_loaded_agents: List[str] = ["director"]

    # ── Агенты ──
    ollama_hosts: OllamaHost = OllamaHost()

    # ── Мониторинг ──
    cpu_warning_threshold: int = 85
    ram_warning_threshold: int = 90
    gpu_warning_threshold: int = 95
    system_monitor_interval: int = 30

    # ── Логирование ──
    debug_mode: bool = False
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_tool_calls: bool = True
    log_agent_thoughts: bool = True
    log_file_operations: bool = True

    # ── Rust core ──
    use_rust_core: bool = True

    @model_validator(mode='after')
    def setup_paths(self):
        """Инициализация вычисляемых полей"""
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.memory_dir is None:
            self.memory_dir = self.data_dir / "memory"
        if self.vector_db_dir is None:
            self.vector_db_dir = self.data_dir / "vector_db"
        if self.knowledge_graph_dir is None:
            self.knowledge_graph_dir = self.data_dir / "knowledge_graph"

        # Создаём директории
        for dir_path in [self.data_dir, self.memory_dir,
                         self.vector_db_dir, self.knowledge_graph_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        return self

    @property
    def episodic_memory_file(self) -> Path:
        return self.memory_dir / "episodic.json"

    @property
    def semantic_memory_file(self) -> Path:
        return self.memory_dir / "semantic.json"

    @property
    def summaries_dir(self) -> Path:
        """v6.0: Директория для иерархических саммари"""
        d = self.memory_dir / "summaries"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def goals_file(self) -> Path:
        return self.memory_dir / "goals.json"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "kristina.log"

    @property
    def embedding_cache_file(self) -> Path:
        return self.data_dir / "embedding_cache.json"  # v6.0: JSON вместо .bin

    @property
    def agent_models(self) -> Dict[str, AgentModelConfig]:
        gpu = self.ollama_hosts.gpu
        cpu = self.ollama_hosts.cpu
        return {
            # Director: основной планировщик, GPU
            # gemma3:12b — хороший русский, помещается в 8GB
            # Альтернатива: qwen3:14b (лучше structured output)
            "director": AgentModelConfig(
                name="gemma3:12b", device="gpu", host=gpu,
                vram_mb=6000, priority=0, timeout=120
            ),
            # Executor: выполняет tool calls, CPU
            # gemma3:4b — быстрый, достаточный для tool execution
            "executor": AgentModelConfig(
                name="gemma3:4b", device="cpu", host=cpu,
                vram_mb=0, priority=1, timeout=120
            ),
            # Analyst: анализ данных, поиск, CPU
            # qwen3:4b — хорош для аналитики и структурирования
            "analyst": AgentModelConfig(
                name="qwen3:4b", device="cpu", host=cpu,
                vram_mb=0, priority=2, timeout=120
            ),
            # Reasoner: логика и рассуждения, CPU
            # deepseek-r1:7b — chain-of-thought, хорош для логики
            # Альтернатива: qwen3:8b с /think (быстрее)
            "reasoner": AgentModelConfig(
                name="deepseek-r1:7b", device="cpu", host=cpu,
                vram_mb=0, priority=3, timeout=120
            ),
        }

    @property
    def agent_timeouts(self) -> Dict[str, int]:
        return {name: cfg.timeout for name, cfg in self.agent_models.items()}


# ═══════════════════════════════════════════════════════════════
#                     ГЛОБАЛЬНЫЙ СИНГЛТОН
# ═══════════════════════════════════════════════════════════════

config = KristinaConfig()


# ═══════════════════════════════════════════════════════════════
#     ОБРАТНАЯ СОВМЕСТИМОСТЬ (через __getattr__)
#
#     Вместо 50+ глобальных переменных, которые не обновляются
#     при изменении config в runtime, используем ленивый доступ.
#     Код `import config; config.MODEL` работает как раньше.
# ═══════════════════════════════════════════════════════════════

# Маппинг СТАРОЕ_ИМЯ → атрибут config
_COMPAT_MAP = {
    "MODEL": "model",
    "ROUTER_MODEL": "router_model",
    "TEMPERATURE": "temperature",
    "CONTEXT_WINDOW": "context_window",
    "MAX_RESPONSE_TOKENS": "max_response_tokens",
    "BASE_DIR": "base_dir",
    "DATA_DIR": "data_dir",
    "MEMORY_DIR": "memory_dir",
    "VECTOR_DB_DIR": "vector_db_dir",
    "WORKING_MEMORY_SIZE": "working_memory_size",
    "MAX_EPISODIC_MEMORY": "max_episodic_memory",
    "MULTI_AGENT_ENABLED": "multi_agent_enabled",
    "HOT_LOADED_AGENTS": "hot_loaded_agents",
    "MAX_VRAM_GB": "max_vram_gb",
    "GPU_VRAM_RESERVED": "gpu_vram_reserved",
    "ENABLE_PARALLEL_EXECUTION": "enable_parallel_execution",
    "MAX_PARALLEL_AGENTS": "max_parallel_agents",
    "EMBEDDING_MODEL": "embedding_model",
    "EMBEDDING_DIM": "embedding_dim",
    "EMBEDDING_CACHE_ENABLED": "embedding_cache_enabled",
    "EMBEDDING_CACHE_FILE": "embedding_cache_file",
    "EMBEDDING_CACHE_MAX_SIZE": "embedding_cache_max_size",
    "RESPONSE_CACHE_ENABLED": "response_cache_enabled",
    "RESPONSE_CACHE_TTL": "response_cache_ttl",
    "AGENT_MAX_ITERATIONS": "agent_max_iterations",
    "AGENT_MAX_ERRORS": "agent_max_errors",
    "AGENT_CACHE_TTL": "agent_cache_ttl",
    "FORCE_RUSSIAN_ONLY": "force_russian_only",
    "VECTOR_SEARCH_RESULTS": "vector_search_results",
    "VECTOR_MIN_AGE_MINUTES": "vector_min_age_minutes",
    "THREAD_TIMEOUT_SECONDS": "thread_timeout_seconds",
    "EPISODIC_MEMORY_FILE": "episodic_memory_file",
    "SEMANTIC_MEMORY_FILE": "semantic_memory_file",
    "GOALS_FILE": "goals_file",
    "LOG_LEVEL": "log_level",
    "LOG_FILE": "log_file",
    "LOG_FORMAT": "log_format",
    "LOG_TOOL_CALLS": "log_tool_calls",
    "LOG_AGENT_THOUGHTS": "log_agent_thoughts",
    "LOG_FILE_OPERATIONS": "log_file_operations",
    "DEBUG_MODE": "debug_mode",
    "DEFAULT_MOOD": "default_mood",
    "INITIAL_ENERGY": "initial_energy",
    "ENERGY_DECAY_PER_MESSAGE": "energy_decay_per_message",
    "WEB_SEARCH_MAX_RESULTS": "web_search_max_results",
    "WEB_FETCH_MAX_SIZE": "web_fetch_max_size",
    "WEB_REQUEST_TIMEOUT": "web_request_timeout",
    "WEB_RATE_LIMIT": "web_rate_limit",
    "FILE_ACCESS_MODE": "file_access_mode",
    "FILE_SEARCH_MAX_DEPTH": "file_search_max_depth",
    "FILE_SEARCH_MAX_RESULTS": "file_search_max_results",
    "CPU_WARNING_THRESHOLD": "cpu_warning_threshold",
    "RAM_WARNING_THRESHOLD": "ram_warning_threshold",
    "GPU_WARNING_THRESHOLD": "gpu_warning_threshold",
    "BLOCKED_EXTENSIONS": "blocked_extensions",
    "BLOCKED_DIRECTORIES": "blocked_directories",
    "ALLOWED_DIRECTORIES": "allowed_directories",
    "PROTECTED_PROCESSES": "protected_processes",
    "SHORT_TERM_MEMORY_SIZE": "short_term_memory_size",
}

# Специальные computed-свойства
_COMPUTED_MAP = {
    "AGENT_MODELS": lambda: {k: v.model_dump() for k, v in config.agent_models.items()},
    "AGENT_TIMEOUTS": lambda: config.agent_timeouts,
    "OLLAMA_GPU_HOST": lambda: config.ollama_hosts.gpu,
    "OLLAMA_CPU_HOST": lambda: config.ollama_hosts.cpu,
    "IDENTITY_DIR": lambda: config.base_dir / "core" / "identity_data",
    "SYSTEM_MONITOR_INTERVAL": lambda: config.system_monitor_interval,
}


# Экспортируемые имена для IDE autocomplete и grep
__all__ = ["config"] + list(_COMPAT_MAP.keys()) + list(_COMPUTED_MAP.keys())


def __getattr__(name):
    """Ленивый доступ к UPPER_CASE переменным через config.lower_case.

    Маппинг задан в _COMPAT_MAP и _COMPUTED_MAP выше.
    Пример: config.MODEL → config.config.model
    """
    if name in _COMPAT_MAP:
        return getattr(config, _COMPAT_MAP[name])
    if name in _COMPUTED_MAP:
        return _COMPUTED_MAP[name]()
    raise AttributeError(f"module 'config' has no attribute {name!r}")
