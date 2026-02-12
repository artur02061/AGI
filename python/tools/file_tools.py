"""
Инструменты для работы с файлами
"""

from pathlib import Path
from typing import Optional
from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_file_path
import config

logger = get_logger("file_tools")


class SearchFilesTool(BaseTool):
    """Поиск файлов"""
    
    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_files",
            description="Ищет файлы на всех дисках компьютера",
            required_args=["filename"],
            arg_types={"filename": str},
            examples=[
                'search_files("document.txt")',
                'search_files("отчёт")',
                'search_files("*.pdf")'
            ]
        )
    
    async def execute(self, filename: str) -> str:
        logger.info(f"Поиск файла: {filename}")
        
        result = await self.controller.search_file(filename)
        
        if result["success"]:
            files = result.get("files", [])
            
            if len(files) > config.FILE_SEARCH_MAX_RESULTS:
                return (
                    f"Найдено {len(files)} файлов (показываю первые {config.FILE_SEARCH_MAX_RESULTS}):\n\n"
                    + result["message"]
                )
            
            return result["message"]
        
        return result["message"]


class ReadFileTool(BaseTool):
    """Чтение файла"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Читает содержимое текстового файла",
            required_args=["filepath"],
            arg_types={"filepath": str},
            examples=[
                'read_file("C:/Users/User/Desktop/notes.txt")',
                'read_file("~/Documents/report.txt")'
            ]
        )
    
    async def execute(self, filepath: str) -> str:
        logger.info(f"Чтение файла: {filepath}")
        
        path = Path(filepath).expanduser()
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Доступ запрещён: {reason}")
            return f"❌ Доступ запрещён: {reason}"
        
        if not path.exists():
            return f"Файл не найден: {filepath}"
        
        if not path.is_file():
            return f"Это не файл: {filepath}"
        
        try:
            # Пытаемся прочитать как текст
            content = path.read_text(encoding='utf-8')
            
            # Ограничиваем размер
            max_size = 5000
            if len(content) > max_size:
                content = content[:max_size] + f"\n\n[...файл обрезан, полный размер: {len(content)} символов]"
            
            logger.info(f"✅ Файл прочитан ({len(content)} символов)")
            return content
        
        except UnicodeDecodeError:
            return f"Файл не является текстовым или использует неподдерживаемую кодировку"
        
        except Exception as e:
            logger.error(f"Ошибка чтения: {e}")
            return f"Ошибка чтения файла: {e}"


class DeleteFileTool(BaseTool):
    """Удаление файла (через корзину)"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_file",
            description="Удаляет файл (перемещает в корзину)",
            required_args=["filepath"],
            arg_types={"filepath": str},
            examples=[
                'delete_file("C:/Users/User/Desktop/temp.txt")',
                'delete_file("Логи.txt")'
            ]
        )
    
    async def execute(self, filepath: str) -> str:
        logger.warning(f"Попытка удаления файла: {filepath}")
        
        path = Path(filepath).expanduser()
        
        # Если это просто имя файла — ищем его сначала
        if not path.is_absolute():
            logger.info(f"Получено имя файла, ищу на рабочем столе: {filepath}")
            
            # Ищем на рабочем столе
            desktop = Path.home() / "Desktop"
            potential_path = desktop / filepath
            
            if potential_path.exists():
                logger.info(f"Файл найден: {potential_path}")
                path = potential_path
            else:
                return f"Файл '{filepath}' не найден на рабочем столе. Укажи полный путь."
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Удаление запрещено: {reason}")
            return f"❌ Удаление запрещено: {reason}"
        
        if not path.exists():
            return f"Файл не найден: {filepath}"
        
        try:
            import send2trash
            
            send2trash.send2trash(str(path))
            
            logger.info(f"✅ Файл перемещён в корзину: {path.name}")
            return f"✅ Файл '{path.name}' перемещён в корзину (можно восстановить)"
        
        except ImportError:
            logger.error("Библиотека send2trash не установлена")
            return "ERROR: Библиотека send2trash не установлена. Установи: pip install send2trash"
        
        except Exception as e:
            logger.error(f"Ошибка удаления: {e}")
            return f"Ошибка удаления файла: {e}"


class ListDirectoryTool(BaseTool):
    """Список файлов в директории"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="Показывает список файлов и папок в директории",
            required_args=["directory"],
            optional_args=["pattern"],
            arg_types={"directory": str, "pattern": str},
            examples=[
                'list_directory("C:/Users/User/Desktop")',
                'list_directory("~/Documents", "*.txt")'
            ]
        )
    
    async def execute(self, directory: str, pattern: str = "*") -> str:
        logger.info(f"Список директории: {directory}")
        
        path = Path(directory).expanduser()
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            return f"❌ Доступ запрещён: {reason}"
        
        if not path.exists():
            return f"Директория не найдена: {directory}"
        
        if not path.is_dir():
            return f"Это не директория: {directory}"
        
        try:
            items = []
            
            for item in path.glob(pattern):
                # Проверяем каждый элемент
                is_item_safe, _ = validate_file_path(item)
                if not is_item_safe:
                    continue
                
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size_kb = item.stat().st_size / 1024
                    items.append(f"📄 {item.name} ({size_kb:.1f} KB)")
            
            if not items:
                return f"Папка пуста или файлы не соответствуют паттерну '{pattern}'"
            
            result = f"📂 {path.name}/ ({len(items)} элементов):\n\n"
            result += "\n".join(items[:50])  # Максимум 50
            
            if len(items) > 50:
                result += f"\n\n... и ещё {len(items) - 50} элементов"
            
            return result
        
        except Exception as e:
            logger.error(f"Ошибка чтения директории: {e}")
            return f"Ошибка: {e}"

class CreateFileTool(BaseTool):
    """Создание файла с содержимым"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="create_file",
            description="Создаёт новый файл с указанным содержимым",
            required_args=["filepath", "content"],
            arg_types={"filepath": str, "content": str},
            examples=[
                'create_file("C:/Users/User/Desktop/test.txt", "Hello World")',
                'create_file("Логи.txt", "Привет")'
            ]
        )
    
    async def execute(self, filepath: str, content: str) -> str:
        logger.info(f"Создание файла: {filepath}")
        
        path = Path(filepath).expanduser()
        
        # Если это просто имя файла — создаём на рабочем столе
        if not path.is_absolute():
            logger.info(f"Получено имя файла, создаю на рабочем столе: {filepath}")
            desktop = Path.home() / "Desktop"
            path = desktop / filepath
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Создание запрещено: {reason}")
            return f"❌ Создание запрещено: {reason}"
        
        # Проверяем, что директория существует
        if not path.parent.exists():
            return f"❌ Директория не существует: {path.parent}"
        
        try:
            # Создаём файл
            path.write_text(content, encoding='utf-8')
            
            logger.info(f"✅ Файл создан: {path.name}")
            return f"✅ Файл '{path.name}' создан по пути:\n{path}\n\nСодержимое:\n{content[:100]}"
        
        except Exception as e:
            logger.error(f"Ошибка создания файла: {e}")
            return f"❌ Ошибка создания файла: {e}"


class WriteFileTool(BaseTool):
    """Запись в существующий файл"""
    
    def __init__(self):
        super().__init__()
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Записывает содержимое в существующий файл (перезаписывает)",
            required_args=["filepath", "content"],
            arg_types={"filepath": str, "content": str},
            examples=[
                'write_file("C:/Users/User/Desktop/notes.txt", "New content")',
                'write_file("Логи.txt", "Обновлённый текст")'
            ]
        )
    
    async def execute(self, filepath: str, content: str) -> str:
        logger.info(f"Запись в файл: {filepath}")
        
        path = Path(filepath).expanduser()
        
        # Если имя — ищем на рабочем столе
        if not path.is_absolute():
            desktop = Path.home() / "Desktop"
            path = desktop / filepath
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Запись запрещена: {reason}")
            return f"❌ Запись запрещена: {reason}"
        
        if not path.exists():
            return f"❌ Файл не найден: {filepath}"
        
        try:
            # Записываем
            path.write_text(content, encoding='utf-8')
            
            logger.info(f"✅ Содержимое записано в: {path.name}")
            return f"✅ Содержимое записано в файл '{path.name}'"
        
        except Exception as e:
            logger.error(f"Ошибка записи: {e}")
            return f"❌ Ошибка записи: {e}"