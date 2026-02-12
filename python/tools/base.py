"""
Кристина 6.0 — Базовый класс для инструментов

ИЗМЕНЕНИЯ v6.0:
- to_ollama_schema() — генерирует JSON schema для Ollama native tool calling
- Обратная совместимость с v4.0/v5.0 (ToolSchema по-прежнему работает)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolSchema:
    """Схема инструмента"""

    name: str
    description: str
    required_args: List[str]
    optional_args: List[str] = None
    arg_types: Dict[str, type] = None
    arg_descriptions: Dict[str, str] = None  # v6.0: описания аргументов
    examples: List[str] = None

    def __post_init__(self):
        self.optional_args = self.optional_args or []
        self.arg_types = self.arg_types or {}
        self.arg_descriptions = self.arg_descriptions or {}
        self.examples = self.examples or []

    def to_ollama_tool(self) -> Dict:
        """
        v6.0: Конвертирует в формат Ollama native tool calling.

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
        """Вызов инструмента с подсчётом статистики"""
        self._call_count += 1
        try:
            result = await self.execute(*args, **kwargs)
            return result
        except Exception as e:
            self._error_count += 1
            return f"ERROR: {str(e)}"

    def to_ollama_tool(self) -> Dict:
        """v6.0: Shortcut — генерирует Ollama tool schema"""
        return self.schema.to_ollama_tool()

    def get_stats(self) -> Dict[str, int]:
        """Статистика использования"""
        return {
            "calls": self._call_count,
            "errors": self._error_count,
        }
