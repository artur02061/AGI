"""
Базовый класс для всех агентов (Hybrid CPU+GPU) - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from ollama import AsyncClient
from datetime import datetime
import asyncio

from utils.logging import get_logger
import config

class BaseAgent(ABC):
    """Базовый агент Multi-Agent системы с поддержкой CPU/GPU"""
    
    def __init__(
        self,
        name: str,
        model_config: Dict,
        capabilities: List[str],
        description: str = ""
    ):
        self.name = name
        self.model = model_config["name"]
        self.device = model_config.get("device", "gpu")
        self.host = model_config.get("host", config.OLLAMA_GPU_HOST)
        self.capabilities = capabilities
        self.description = description
        
        # Создаём клиент для нужного хоста
        self.client = AsyncClient(host=self.host)
        
        self.logger = get_logger(f"agent.{name}")
        
        # Статистика
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "timeout_errors": 0
        }
        
        # Состояние
        self.is_loaded = (self.device == "cpu")  # CPU модели всегда "загружены"
        self.last_used = None
        
        device_emoji = "🎮" if self.device == "gpu" else "🖥️"
        self.logger.info(
            f"{device_emoji} Агент {name} ({self.device.upper()}) "
            f"инициализирован (модель: {self.model})"
        )
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> str:
        """
        Выполняет задачу
        
        Args:
            task: Словарь с описанием задачи
        
        Returns:
            Результат выполнения
        """
        pass
    
    def can_handle(self, task_type: str) -> bool:
        """Может ли агент обработать задачу"""
        return task_type in self.capabilities
    
    async def _call_model(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: List[Dict] = None,
    ) -> str:
        """
        Вызов модели через Ollama с timeout и обработкой ошибок.
        
        v6.0: Поддержка tools= parameter для native tool calling.
        """
        
        start_time = datetime.now()
        
        # Таймаут для агента
        timeout = config.AGENT_TIMEOUTS.get(self.name, 30)
        
        try:
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            # v6.0: native tool calling
            if tools:
                call_kwargs["tools"] = tools

            response = await asyncio.wait_for(
                self.client.chat(**call_kwargs),
                timeout=timeout,
            )
            
            message = response["message"]
            content = message.get("content", "")
            
            # Обновляем статистику
            elapsed = (datetime.now() - start_time).total_seconds()
            self._update_stats(True, elapsed)
            
            device_emoji = "⚡" if self.device == "gpu" else "🔄"
            self.logger.debug(f"{device_emoji} Ответ получен за {elapsed:.2f}s")
            
            # v6.0: если модель вернула tool_calls, возвращаем полный message
            if message.get("tool_calls"):
                return message
            
            return content
        
        # ✅ ИСПРАВЛЕНИЕ: Явная обработка timeout
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["timeout_errors"] += 1
            self._update_stats(False, elapsed)
            
            self.logger.error(f"⏱️ Timeout ({timeout}s) при вызове {self.model}")
            raise TimeoutError(f"Модель {self.model} не ответила за {timeout}s")
        
        # ✅ ИСПРАВЛЕНИЕ: Детальная обработка других ошибок
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self._update_stats(False, elapsed)
            
            self.logger.error(f"❌ Ошибка вызова модели {self.model}: {type(e).__name__}: {e}")
            raise
    
    def _update_stats(self, success: bool, elapsed: float):
        """Обновляет статистику"""
        
        self.stats["total_calls"] += 1
        
        if success:
            self.stats["successful_calls"] += 1
        else:
            self.stats["failed_calls"] += 1
        
        self.stats["total_time"] += elapsed
        
        if self.stats["total_calls"] > 0:
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["total_calls"]
        
        self.last_used = datetime.now()
    
    async def close(self):
        """Закрывает соединение с Ollama (предотвращает ResourceWarning)"""
        try:
            if hasattr(self.client, '_client') and self.client._client:
                await self.client._client.aclose()
                self.logger.debug(f"🔌 Соединение {self.name} закрыто")
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        return {
            "name": self.name,
            "model": self.model,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            **self.stats
        }
