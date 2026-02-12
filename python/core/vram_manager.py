"""
VRAM Manager v2.0 — Hybrid CPU+GPU - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

from typing import List, Dict, Set
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from utils.logging import get_logger
import config

logger = get_logger("vram_manager")

class VRAMManager:
    """
    Упрощённый VRAM Manager для Hybrid архитектуры
    
    CPU модели не используют VRAM, поэтому управление проще
    """
    
    def __init__(self):
        self.max_vram = config.MAX_VRAM_GB * 1024 * 1024 * 1024
        
        # CPU модели всегда "загружены"
        self.loaded_models: Set[str] = set()
        
        # Инициализация NVML
        self.nvml_initialized = False
        self.gpu_handle = None
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
                logger.info("✅ NVML инициализирован")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось инициализировать NVML: {e}")
        else:
            logger.warning("⚠️ pynvml не установлен, мониторинг GPU недоступен")
        
        # Отмечаем CPU агентов как загруженных
        for agent_name, agent_config in config.AGENT_MODELS.items():
            if agent_config.get("device") == "cpu":
                self.loaded_models.add(agent_name)
                logger.info(f"🖥️  Агент {agent_name} (CPU) всегда доступен")
        
        logger.info(f"✅ VRAM Manager v2.0: Hybrid CPU+GPU режим")
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Получает использование VRAM (только GPU)"""
        
        if not self.nvml_initialized or not self.gpu_handle:
            return {
                "used_gb": 0.0,
                "free_gb": 8.0,
                "total_gb": 8.0,
                "usage_percent": 0.0,
                "available": False
            }
        
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            
            return {
                "used_gb": info.used / (1024**3),
                "free_gb": info.free / (1024**3),
                "total_gb": info.total / (1024**3),
                "usage_percent": (info.used / info.total) * 100,
                "available": True
            }
        
        except Exception as e:
            logger.error(f"❌ Ошибка получения VRAM: {e}")
            return {
                "used_gb": 0.0,
                "free_gb": 8.0,
                "total_gb": 8.0,
                "usage_percent": 0.0,
                "available": False
            }
    
    async def ensure_loaded(self, agent_names: List[str]) -> bool:
        """
        Проверяет доступность моделей в Ollama.

        v6.0 ИСПРАВЛЕНИЕ: вместо no-op, реально проверяем через ollama.list()
        """

        logger.debug(f"📋 Запрос загрузки: {agent_names}")

        # Кэшируем список моделей (обновляем раз в 60с)
        import time
        now = time.time()
        if not hasattr(self, '_model_cache') or now - self._model_cache_time > 60:
            try:
                from ollama import AsyncClient
                client = AsyncClient()
                models_resp = await client.list()
                self._model_cache = {m.get('name', '').split(':')[0] for m in models_resp.get('models', [])}
                self._model_cache_time = now
            except Exception as e:
                logger.warning(f"⚠️ Не удалось проверить модели Ollama: {e}")
                # Fallback: считаем всё загруженным
                for agent_name in agent_names:
                    self.loaded_models.add(agent_name)
                return True

        missing = []
        for agent_name in agent_names:
            if agent_name not in self.loaded_models:
                # Проверяем, есть ли модель агента в Ollama
                agent_cfg = config.AGENT_MODELS.get(agent_name, {})
                model_name = agent_cfg.get("name", "").split(":")[0] if isinstance(agent_cfg, dict) else ""
                if hasattr(self, '_model_cache') and model_name and model_name not in self._model_cache:
                    missing.append(f"{agent_name} ({model_name})")
                self.loaded_models.add(agent_name)

        if missing:
            logger.warning(f"⚠️ Модели не найдены в Ollama: {', '.join(missing)}")

        logger.debug("✅ Все агенты доступны")
        return True
    
    def get_stats(self) -> Dict:
        """Статистика менеджера"""
        
        vram_info = self.get_vram_usage()
        
        # Разделяем агентов по устройствам
        gpu_agents = []
        cpu_agents = []
        
        for agent_name, agent_config in config.AGENT_MODELS.items():
            if agent_config.get("device") == "gpu":
                gpu_agents.append(agent_name)
            else:
                cpu_agents.append(agent_name)
        
        return {
            "vram": vram_info,
            "loaded_agents": list(self.loaded_models),
            "gpu_agents": gpu_agents,
            "cpu_agents": cpu_agents,
            "nvml_available": self.nvml_initialized,
            "mode": "hybrid"
        }
    
    # ✅ ИСПРАВЛЕНИЕ: Явный метод cleanup вместо ненадёжного __del__
    def cleanup(self):
        """
        Явная очистка ресурсов NVML
        
        Должен вызываться при завершении работы приложения:
        - В main.py при выходе
        - При обработке сигналов SIGTERM/SIGINT
        """
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
                self.gpu_handle = None
                logger.info("✅ NVML корректно завершён")
            except Exception as e:
                logger.error(f"❌ Ошибка при завершении NVML: {e}")
    
    def __del__(self):
        """
        Fallback cleanup (может не вызваться!)
        Лучше использовать явный cleanup()
        """
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass  # В __del__ нельзя логировать
