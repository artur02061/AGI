"""
Системные инструменты (запуск приложений, процессы, мониторинг)
"""

from modules.system_control.controller import SystemController
from typing import Dict, Any
from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_process_name
import config

logger = get_logger("system_tools")

class LaunchAppTool(BaseTool):
    """Запускает приложение"""
    
    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="launch_app",
            description="Запускает приложение или игру на компьютере",
            required_args=["app_name"],
            arg_types={"app_name": str},
            examples=[
                'launch_app("steam")',
                'launch_app("chrome")',
                'launch_app("discord")'
            ]
        )
    
    async def execute(self, app_name: str) -> str:
        logger.info(f"Запуск приложения: {app_name}")
        
        result = await self.controller.launch_app(app_name)
        
        if result["success"]:
            logger.info(f"✅ {result['message']}")
        else:
            logger.warning(f"❌ {result['message']}")
        
        return result["message"]


class SystemStatusTool(BaseTool):
    """Показывает статус системы"""
    
    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="system_status",
            description="Показывает текущую нагрузку системы (CPU, RAM, GPU)",
            required_args=[],
            examples=['system_status()']
        )
    
    async def execute(self) -> str:
        logger.debug("Получение статуса системы")
        
        status = await self.controller.get_system_status()
        
        cpu = status['cpu']['usage_percent']
        ram = status['ram']['usage_percent']
        
        result = f"💻 CPU: {cpu:.1f}%\n💾 RAM: {ram:.1f}%"
        
        # GPU если доступен
        gpu = status.get('gpu', {})
        if 'usage_percent' in gpu and gpu.get('error') != "Недоступен":
            result += f"\n🎮 GPU: {gpu['usage_percent']}%"
            
            if 'temperature_c' in gpu:
                result += f" (🌡️ {gpu['temperature_c']}°C)"
        
        # Предупреждения
        warnings = []
        if cpu > config.CPU_WARNING_THRESHOLD:
            warnings.append("⚠️ CPU перегружен!")
        if ram > config.RAM_WARNING_THRESHOLD:
            warnings.append("⚠️ RAM заполнена!")
        
        if warnings:
            result += "\n\n" + "\n".join(warnings)
        
        return result


class ListProcessesTool(BaseTool):
    """Список процессов"""
    
    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_processes",
            description="Показывает список запущенных процессов",
            required_args=[],
            optional_args=["filter_keyword", "limit"],
            arg_types={"filter_keyword": str, "limit": int},
            examples=[
                'list_processes()',
                'list_processes("chrome")',
                'list_processes(limit=5)'
            ]
        )
    
    async def execute(self, filter_keyword: str = None, limit: int = 10) -> str:
        logger.debug(f"Получение процессов (фильтр: {filter_keyword}, лимит: {limit})")
        
        processes = await self.controller.list_processes(
            filter_keyword=filter_keyword,
            limit=limit
        )
        
        if not processes:
            return "Процессов не найдено"
        
        lines = ["📊 Топ процессов по нагрузке:", ""]
        
        for proc in processes[:limit]:
            name = proc['name'][:30]  # Обрезаем длинные имена
            cpu = proc['cpu']
            mem = proc['memory']
            
            lines.append(f"• {name:<30} CPU: {cpu:5.1f}%  RAM: {mem:5.1f}%")
        
        return "\n".join(lines)


class SearchAppsTool(BaseTool):
    """
    Поиск приложений

    ПРИМЕЧАНИЕ: Не зарегистрирован в main.py по умолчанию.
    """
    
    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_apps",
            description="Ищет установленные приложения на компьютере",
            required_args=["query"],
            arg_types={"query": str},
            examples=[
                'search_apps("steam")',
                'search_apps("chrome")',
                'search_apps("офис")'
            ]
        )
    
    async def execute(self, query: str) -> str:
        logger.info(f"Поиск приложений: {query}")
        
        result = await self.controller.search_apps(query)
        
        return result["message"]

# GetCurrentTimeTool и GetWeatherTool — УДАЛЕНЫ (дубликаты).
# Рабочие реализации находятся в tools/web_tools.py