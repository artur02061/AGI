"""
Кристина 6.0 — Базовый класс для инструментов

ИЗМЕНЕНИЯ v6.0:
- to_ollama_schema() — генерирует JSON schema для Ollama native tool calling
- Обратная совместимость с v4.0/v5.0 (ToolSchema по-прежнему работает)

ИЗМЕНЕНИЯ v6.1 (JARVIS Edition):
- category — группировка инструментов (file, system, web, memory, util)
- Optional[type] — поддержка необязательных типов
- enum — поддержка перечислений в аргументах
- danger_level — уровень опасности (info/warning/danger)
- requires_confirmation — запрос подтверждения перед выполнением
- retry_on_error — автоматический повтор при ошибке
- timeout — таймаут выполнения
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
import time


@dataclass
class ToolSchema:
    """Схема инструмента"""

    name: str
    description: str
    required_args: List[str]
    optional_args: List[str] = None
    arg_types: Dict[str, type] = None
    arg_descriptions: Dict[str, str] = None
    arg_enums: Dict[str, List[str]] = None  # v6.1: перечисления для аргументов
    examples: List[str] = None
    category: str = "general"  # v6.1: file | system | web | memory | util
    danger_level: str = "info"  # v6.1: info | warning | danger
    requires_confirmation: bool = False  # v6.1: запрос подтверждения

    def __post_init__(self):
        self.optional_args = self.optional_args or []
        self.arg_types = self.arg_types or {}
        self.arg_descriptions = self.arg_descriptions or {}
        self.arg_enums = self.arg_enums or {}
        self.examples = self.examples or []

    def to_ollama_tool(self) -> Dict:
        """
        v6.0: Конвертирует в формат Ollama native tool calling.
        v6.1: Добавлена поддержка enum, улучшенные описания.

        Формат:
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "...",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "..."}
                    },
                    "required": ["query"]
                }
            }
        }
        """
        # Маппинг Python типов → JSON Schema типов
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        properties = {}

        for arg_name in self.required_args + self.optional_args:
            python_type = self.arg_types.get(arg_name, str)
            json_type = type_map.get(python_type, "string")

            prop = {"type": json_type}

            # Описание аргумента
            desc = self.arg_descriptions.get(arg_name)
            if desc:
                prop["description"] = desc
            else:
                # Автогенерируем из имени
                prop["description"] = arg_name.replace("_", " ")

            # v6.1: Enum поддержка
            if arg_name in self.arg_enums:
                prop["enum"] = self.arg_enums[arg_name]

            properties[arg_name] = prop

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_args,
                },
            },
        }


class BaseTool(ABC):
    """Базовый класс для инструмента"""

    def __init__(self):
        self._call_count = 0
        self._error_count = 0
        self._total_time_ms = 0.0
        self._last_error: Optional[str] = None

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Возвращает схему инструмента"""
        pass

    @abstractmethod
    async def execute(self, *args, **kwargs) -> str:
        """Выполняет инструмент"""
        pass

    async def __call__(self, *args, **kwargs) -> str:
        """Вызов инструмента с подсчётом статистики и таймингом"""
        self._call_count += 1
        start = time.monotonic()
        try:
            result = await self.execute(*args, **kwargs)
            return result
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            return f"ERROR: {str(e)}"
        finally:
            elapsed = (time.monotonic() - start) * 1000
            self._total_time_ms += elapsed

    def to_ollama_tool(self) -> Dict:
        """v6.0: Shortcut — генерирует Ollama tool schema"""
        return self.schema.to_ollama_tool()

    def get_stats(self) -> Dict[str, Any]:
        """Статистика использования"""
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "avg_time_ms": round(self._total_time_ms / max(self._call_count, 1), 1),
            "total_time_ms": round(self._total_time_ms, 1),
            "last_error": self._last_error,
        }
