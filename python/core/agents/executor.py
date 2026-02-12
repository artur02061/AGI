"""
Executor Agent — быстрое выполнение действий
"""

from typing import Optional, Dict, Any
import re

from core.agents.base_agent import BaseAgent
import config

class ExecutorAgent(BaseAgent):
    """
    Исполнитель — быстрые системные действия
    
    Особенность: выполняет БЕЗ размышлений, мгновенно
    """
    
    def __init__(self, tools: Dict):
        # Получаем конфиг из config.py
        model_config = config.AGENT_MODELS["executor"]
        
        super().__init__(
            name="executor",
            model_config=model_config,
            capabilities=[
                "delete_file",
                "read_file",
                "list_directory",
                "launch_app",
                "system_status",
                "list_processes",
                "get_current_time",
                "get_weather",
                "search_files",
                "create_file",
                "write_file"
            ],
            description="Быстрый исполнитель системных команд"
        )
        
        self.tools = tools
    
    async def execute(self, task: Dict[str, Any]) -> str:
        """
        Выполняет задачу напрямую через инструменты
        
        Args:
            task: {
                "tool": "delete_file",
                "args": ["filename.txt"],
                "user_input": "оригинальный запрос" (опционально)
            }
        """
        
        tool_name = task.get("tool")
        args = task.get("args", [])
        user_input = task.get("user_input", "")
        
        # Если tool не указан, пытаемся определить из user_input
        if not tool_name and user_input:
            detected = self._detect_tool_from_input(user_input)
            if detected:
                tool_name = detected["tool"]
                args = detected["args"]
        
        if not tool_name:
            return "ERROR: Не указан инструмент для выполнения"
        
        if tool_name not in self.tools:
            return f"ERROR: Инструмент {tool_name} недоступен"
        
        if tool_name not in self.capabilities:
            return f"ERROR: Я не умею выполнять {tool_name}"
        
        try:
            self.logger.info(f"⚡ Выполнение: {tool_name}({args})")
            
            # Прямой вызов инструмента
            tool = self.tools[tool_name]
            result = await tool(*args)
            
            self._update_stats(True, 0.1)  # Быстрое выполнение
            
            return str(result)
        
        except Exception as e:
            self._update_stats(False, 0.1)
            self.logger.error(f"Ошибка выполнения {tool_name}: {e}")
            return f"ERROR: {str(e)}"
    
    def _detect_tool_from_input(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Определяет инструмент из текста"""
    
        import re
    
        text_lower = user_input.lower()
    
        # === ЧТЕНИЕ ФАЙЛА (по упоминанию) ===
        if any(word in text_lower for word in ['файл', 'активатор', 'на рабочем столе']):
            # Ищем название файла
            match = re.search(r'([\wа-яёА-ЯЁ]+\.\w+)', user_input, re.I)
        
            if match and any(word in text_lower for word in ['прочитай', 'открой', 'покажи', 'видишь', 'есть файл']):
                filename = match.group(1)
                return {"tool": "read_file", "args": [filename]}
    
        # === СОЗДАНИЕ БАТНИКА/СКРИПТА ===
        if any(phrase in text_lower for phrase in ['создай батник', 'создай скрипт', 'оптимизир', 'автоматизир']):
            # Это задача для Director, вернём None
            return None
    
        # === СОЗДАНИЕ ФАЙЛА ===
        if any(word in text_lower for word in ['создай файл', 'создать файл', 'новый файл']):
            match_file = re.search(r'([\wа-яёА-ЯЁ]+\.\w+)', user_input, re.I)
            match_content = re.search(r'напиши[^:]*:\s*(.+)', user_input, re.I)
        
            if match_file:
                filename = match_file.group(1)
                content = match_content.group(1) if match_content else "Пустой файл"
                return {"tool": "create_file", "args": [filename, content]}
    
        # === УДАЛЕНИЕ ФАЙЛА ===
        if any(word in text_lower for word in ['удали', 'удалить', 'удал', 'сотри']):
            # Ищем имя файла с расширением
            match = re.search(r'([\wа-яёА-ЯЁ\-]+\.\w+)', user_input, re.I)
            if match:
                return {"tool": "delete_file", "args": [match.group(1)]}
        
            # Если не нашли — ищем "этот файл" или "его"
            if any(word in text_lower for word in ['этот файл', 'его', 'этот']):
                # Нужен контекст из памяти - пока возвращаем None
                return None
    
        # === ЗАПУСК ПРИЛОЖЕНИЯ ===
        if any(word in text_lower for word in ['запусти', 'открой', 'запустить', 'открыть']):
            if 'файл' not in text_lower:
                # Ищем название приложения
                match = re.search(r'(?:запусти|открой|запустить|открыть)\s+(?:приложение\s+)?(\w+)', text_lower)
                if match:
                    return {"tool": "launch_app", "args": [match.group(1)]}
    
        # === СТАТУС СИСТЕМЫ ===
        if 'статус систем' in text_lower or 'status' in text_lower:
            return {"tool": "system_status", "args": []}
    
        # === ВРЕМЯ ===
        if any(p in text_lower for p in ['время', 'час', 'который час', 'сколько время']):
            return {"tool": "get_current_time", "args": []}
    
        # === ПРОЦЕССЫ ===
        if any(p in text_lower for p in ['процесс', 'список процесс', 'запущенные']):
            return {"tool": "list_processes", "args": []}
    
        # === ПОГОДА ===
        if 'погода' in text_lower:
            match = re.search(r'(?:в|для)\s+([\wа-яёА-ЯЁ]+)', text_lower)
            city = match.group(1) if match else "Moscow"
            return {"tool": "get_weather", "args": [city]}
    
        return None