"""
Инструменты для работы с файлами — JARVIS Edition v6.1

Полный набор файловых операций:
- Поиск, чтение, создание, запись, дополнение
- Копирование, перемещение, переименование
- Удаление (через корзину)
- Информация о файле/директории
- Создание директорий
- Архивация и распаковка (zip, tar.gz)
- Листинг директорий с фильтрами
"""

import os
import shutil
import stat
from pathlib import Path
from typing import Optional
from datetime import datetime

from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_file_path, validate_file_write
import config

logger = get_logger("file_tools")


# ═══════════════════════════════════════════════════════════════
#                         ПОИСК
# ═══════════════════════════════════════════════════════════════

class SearchFilesTool(BaseTool):
    """Поиск файлов по имени/паттерну"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_files",
            description="Ищет файлы на компьютере по имени или паттерну. Поддерживает wildcards: *.pdf, report*, *.py",
            required_args=["filename"],
            optional_args=["directory"],
            arg_types={"filename": str, "directory": str},
            arg_descriptions={
                "filename": "Имя файла или паттерн для поиска (например: '*.pdf', 'report', 'config.yaml')",
                "directory": "Директория для поиска (по умолчанию — домашняя папка)",
            },
            category="file",
            examples=[
                'search_files("document.txt")',
                'search_files("*.pdf", "/home/user/Documents")',
                'search_files("отчёт")',
            ],
        )

    async def execute(self, filename: str, directory: str = None) -> str:
        logger.info(f"Поиск файла: {filename}" + (f" в {directory}" if directory else ""))

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


# ═══════════════════════════════════════════════════════════════
#                         ЧТЕНИЕ
# ═══════════════════════════════════════════════════════════════

class ReadFileTool(BaseTool):
    """Чтение содержимого файла"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Читает содержимое текстового файла. Поддерживает UTF-8, Latin-1, CP1251. Большие файлы обрезаются.",
            required_args=["filepath"],
            optional_args=["encoding", "lines"],
            arg_types={"filepath": str, "encoding": str, "lines": int},
            arg_descriptions={
                "filepath": "Полный или относительный путь к файлу",
                "encoding": "Кодировка (utf-8, cp1251, latin-1). По умолчанию — автоопределение",
                "lines": "Количество строк для чтения (0 = весь файл). По умолчанию — весь файл",
            },
            category="file",
            examples=[
                'read_file("/home/user/notes.txt")',
                'read_file("config.yaml")',
                'read_file("/var/log/syslog", lines=50)',
            ],
        )

    async def execute(self, filepath: str, encoding: str = None, lines: int = 0) -> str:
        logger.info(f"Чтение файла: {filepath}")

        path = Path(filepath).expanduser()

        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Доступ запрещён: {reason}")
            return f"Доступ запрещён: {reason}"

        if not path.exists():
            return f"Файл не найден: {filepath}"

        if not path.is_file():
            return f"Это не файл: {filepath}"

        # Автоопределение кодировки
        encodings_to_try = [encoding] if encoding else ["utf-8", "cp1251", "latin-1"]

        content = None
        used_encoding = None

        for enc in encodings_to_try:
            try:
                content = path.read_text(encoding=enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            # Попробуем как бинарный
            size = path.stat().st_size
            return f"Бинарный файл ({_human_size(size)}). Невозможно отобразить как текст."

        # Фильтр по строкам
        if lines > 0:
            content_lines = content.split('\n')
            content = '\n'.join(content_lines[:lines])
            if len(content_lines) > lines:
                content += f"\n\n[...показано {lines} из {len(content_lines)} строк]"

        # Ограничиваем размер
        max_size = getattr(config, 'FILE_READ_MAX_SIZE', 50000)
        if len(content) > max_size:
            content = content[:max_size] + f"\n\n[...файл обрезан, полный размер: {len(content)} символов]"

        logger.info(f"Файл прочитан ({len(content)} символов, {used_encoding})")
        return content


# ═══════════════════════════════════════════════════════════════
#                     СОЗДАНИЕ / ЗАПИСЬ
# ═══════════════════════════════════════════════════════════════

class CreateFileTool(BaseTool):
    """Создание нового файла"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="create_file",
            description="Создаёт новый файл с указанным содержимым. Если файл уже существует — не перезаписывает (используй write_file).",
            required_args=["filepath", "content"],
            arg_types={"filepath": str, "content": str},
            arg_descriptions={
                "filepath": "Путь к новому файлу. Если указано только имя — создаёт на рабочем столе",
                "content": "Содержимое файла",
            },
            category="file",
            examples=[
                'create_file("/home/user/scripts/hello.py", "print(\'Hello World\')")',
                'create_file("заметка.txt", "Не забыть купить молоко")',
            ],
        )

    async def execute(self, filepath: str, content: str) -> str:
        logger.info(f"Создание файла: {filepath}")

        path = Path(filepath).expanduser()

        # Если это просто имя файла — создаём в домашней директории
        if not path.is_absolute():
            home_dir = Path.home()
            desktop = home_dir / "Desktop"
            if not desktop.exists():
                desktop = home_dir / "Рабочий стол"
            if not desktop.exists():
                desktop = home_dir
            path = desktop / filepath

        # Проверка безопасности (запись)
        is_safe, reason = validate_file_write(path)
        if not is_safe:
            logger.warning(f"Создание запрещено: {reason}")
            return f"Создание запрещено: {reason}"

        if path.exists():
            return f"Файл уже существует: {path}. Используй write_file для перезаписи."

        # Создаём родительскую директорию если нужно
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content, encoding='utf-8')
            logger.info(f"Файл создан: {path}")
            return f"Файл '{path.name}' создан: {path}\nРазмер: {_human_size(len(content.encode('utf-8')))}"
        except Exception as e:
            logger.error(f"Ошибка создания файла: {e}")
            return f"Ошибка создания файла: {e}"


class WriteFileTool(BaseTool):
    """Запись/перезапись файла"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Записывает содержимое в файл (полная перезапись). Файл должен существовать.",
            required_args=["filepath", "content"],
            arg_types={"filepath": str, "content": str},
            arg_descriptions={
                "filepath": "Путь к файлу",
                "content": "Новое содержимое (перезаписывает полностью)",
            },
            category="file",
            danger_level="warning",
            examples=[
                'write_file("/home/user/config.yaml", "key: value")',
            ],
        )

    async def execute(self, filepath: str, content: str) -> str:
        logger.info(f"Запись в файл: {filepath}")

        path = Path(filepath).expanduser()

        # Если имя — ищем в стандартных местах
        if not path.is_absolute():
            for d in [Path.home() / "Desktop", Path.home() / "Рабочий стол", Path.home()]:
                candidate = d / filepath
                if candidate.exists():
                    path = candidate
                    break
            else:
                path = Path.home() / filepath

        # Проверка безопасности (запись)
        is_safe, reason = validate_file_write(path)
        if not is_safe:
            logger.warning(f"Запись запрещена: {reason}")
            return f"Запись запрещена: {reason}"

        if not path.exists():
            return f"Файл не найден: {filepath}. Используй create_file для создания нового."

        try:
            path.write_text(content, encoding='utf-8')
            logger.info(f"Содержимое записано в: {path.name}")
            return f"Содержимое записано в файл '{path.name}'"
        except Exception as e:
            logger.error(f"Ошибка записи: {e}")
            return f"Ошибка записи: {e}"


class AppendFileTool(BaseTool):
    """Дополнение файла (добавление в конец)"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="append_file",
            description="Добавляет текст в конец существующего файла, не затирая предыдущее содержимое.",
            required_args=["filepath", "content"],
            arg_types={"filepath": str, "content": str},
            arg_descriptions={
                "filepath": "Путь к файлу",
                "content": "Текст для добавления в конец файла",
            },
            category="file",
            examples=[
                'append_file("/home/user/log.txt", "\\n2025-01-15: Новая запись")',
                'append_file("заметки.txt", "\\n- Ещё один пункт")',
            ],
        )

    async def execute(self, filepath: str, content: str) -> str:
        logger.info(f"Дополнение файла: {filepath}")

        path = Path(filepath).expanduser()

        if not path.is_absolute():
            for d in [Path.home() / "Desktop", Path.home() / "Рабочий стол", Path.home()]:
                candidate = d / filepath
                if candidate.exists():
                    path = candidate
                    break
            else:
                return f"Файл не найден: {filepath}"

        is_safe, reason = validate_file_write(path)
        if not is_safe:
            return f"Запись запрещена: {reason}"

        if not path.exists():
            return f"Файл не найден: {filepath}. Используй create_file для создания."

        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Текст добавлен в: {path.name}")
            return f"Текст добавлен в конец файла '{path.name}' (+{len(content)} символов)"
        except Exception as e:
            logger.error(f"Ошибка дополнения: {e}")
            return f"Ошибка: {e}"


# ═══════════════════════════════════════════════════════════════
#                   КОПИРОВАНИЕ / ПЕРЕМЕЩЕНИЕ
# ═══════════════════════════════════════════════════════════════

class CopyFileTool(BaseTool):
    """Копирование файла или директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="copy_file",
            description="Копирует файл или директорию в указанное место. Поддерживает рекурсивное копирование папок.",
            required_args=["source", "destination"],
            arg_types={"source": str, "destination": str},
            arg_descriptions={
                "source": "Путь к исходному файлу или директории",
                "destination": "Путь назначения (куда копировать)",
            },
            category="file",
            examples=[
                'copy_file("/home/user/report.pdf", "/home/user/backup/report.pdf")',
                'copy_file("/home/user/project", "/home/user/project_backup")',
            ],
        )

    async def execute(self, source: str, destination: str) -> str:
        logger.info(f"Копирование: {source} → {destination}")

        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        is_safe, reason = validate_file_path(src)
        if not is_safe:
            return f"Доступ к источнику запрещён: {reason}"

        is_safe, reason = validate_file_write(dst)
        if not is_safe:
            return f"Запись в назначение запрещена: {reason}"

        if not src.exists():
            return f"Источник не найден: {source}"

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)

            if src.is_dir():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                item_count = sum(1 for _ in dst.rglob('*'))
                logger.info(f"Директория скопирована: {src.name} ({item_count} элементов)")
                return f"Директория '{src.name}' скопирована в {dst} ({item_count} элементов)"
            else:
                shutil.copy2(str(src), str(dst))
                logger.info(f"Файл скопирован: {src.name}")
                return f"Файл '{src.name}' скопирован в {dst}"
        except Exception as e:
            logger.error(f"Ошибка копирования: {e}")
            return f"Ошибка копирования: {e}"


class MoveFileTool(BaseTool):
    """Перемещение файла или директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="move_file",
            description="Перемещает файл или директорию в новое место (вырезать-вставить).",
            required_args=["source", "destination"],
            arg_types={"source": str, "destination": str},
            arg_descriptions={
                "source": "Путь к исходному файлу/директории",
                "destination": "Новый путь (куда переместить)",
            },
            category="file",
            danger_level="warning",
            examples=[
                'move_file("/home/user/old_report.pdf", "/home/user/archive/report_2024.pdf")',
            ],
        )

    async def execute(self, source: str, destination: str) -> str:
        logger.info(f"Перемещение: {source} → {destination}")

        src = Path(source).expanduser()
        dst = Path(destination).expanduser()

        is_safe, reason = validate_file_path(src)
        if not is_safe:
            return f"Доступ к источнику запрещён: {reason}"

        is_safe, reason = validate_file_write(dst)
        if not is_safe:
            return f"Запись в назначение запрещена: {reason}"

        # Дополнительно: проверяем, можно ли удалить из исходного места
        is_safe, reason = validate_file_write(src)
        if not is_safe:
            return f"Удаление из источника запрещено: {reason}"

        if not src.exists():
            return f"Источник не найден: {source}"

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logger.info(f"Перемещён: {src.name} → {dst}")
            return f"'{src.name}' перемещён в {dst}"
        except Exception as e:
            logger.error(f"Ошибка перемещения: {e}")
            return f"Ошибка перемещения: {e}"


class RenameFileTool(BaseTool):
    """Переименование файла или директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="rename_file",
            description="Переименовывает файл или директорию (не перемещая из текущей папки).",
            required_args=["filepath", "new_name"],
            arg_types={"filepath": str, "new_name": str},
            arg_descriptions={
                "filepath": "Путь к файлу или директории",
                "new_name": "Новое имя (только имя, без пути)",
            },
            category="file",
            examples=[
                'rename_file("/home/user/old_name.txt", "new_name.txt")',
                'rename_file("/home/user/project", "project_v2")',
            ],
        )

    async def execute(self, filepath: str, new_name: str) -> str:
        logger.info(f"Переименование: {filepath} → {new_name}")

        path = Path(filepath).expanduser()

        is_safe, reason = validate_file_write(path)
        if not is_safe:
            return f"Переименование запрещено: {reason}"

        if not path.exists():
            return f"Не найдено: {filepath}"

        # Новый путь — в той же директории
        new_path = path.parent / new_name

        is_safe, reason = validate_file_write(new_path)
        if not is_safe:
            return f"Переименование запрещено: {reason}"

        if new_path.exists():
            return f"Файл/директория с именем '{new_name}' уже существует"

        try:
            path.rename(new_path)
            logger.info(f"Переименован: {path.name} → {new_name}")
            return f"'{path.name}' переименован в '{new_name}'"
        except Exception as e:
            logger.error(f"Ошибка переименования: {e}")
            return f"Ошибка переименования: {e}"


# ═══════════════════════════════════════════════════════════════
#                         УДАЛЕНИЕ
# ═══════════════════════════════════════════════════════════════

class DeleteFileTool(BaseTool):
    """Удаление файла (через корзину)"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_file",
            description="Удаляет файл или папку, перемещая в корзину (можно восстановить). Безопасное удаление.",
            required_args=["filepath"],
            arg_types={"filepath": str},
            arg_descriptions={
                "filepath": "Путь к файлу или директории для удаления",
            },
            category="file",
            danger_level="warning",
            requires_confirmation=True,
            examples=[
                'delete_file("/home/user/Desktop/temp.txt")',
                'delete_file("/home/user/old_project")',
            ],
        )

    async def execute(self, filepath: str) -> str:
        logger.warning(f"Попытка удаления: {filepath}")

        path = Path(filepath).expanduser()

        # Если это просто имя файла — ищем в стандартных местах
        if not path.is_absolute():
            search_dirs = [
                Path.home() / "Desktop",
                Path.home() / "Рабочий стол",
                Path.home() / "Downloads",
                Path.home(),
            ]
            found = False
            for d in search_dirs:
                potential_path = d / filepath
                if potential_path.exists():
                    path = potential_path
                    found = True
                    break
            if not found:
                return f"Файл '{filepath}' не найден. Укажи полный путь."

        is_safe, reason = validate_file_write(path)
        if not is_safe:
            logger.warning(f"Удаление запрещено: {reason}")
            return f"Удаление запрещено: {reason}"

        if not path.exists():
            return f"Не найдено: {filepath}"

        try:
            import send2trash
            send2trash.send2trash(str(path))
            logger.info(f"Перемещён в корзину: {path.name}")
            return f"'{path.name}' перемещён в корзину (можно восстановить)"
        except ImportError:
            logger.error("send2trash не установлен")
            return "Библиотека send2trash не установлена. Установи: pip install send2trash"
        except Exception as e:
            logger.error(f"Ошибка удаления: {e}")
            return f"Ошибка удаления: {e}"


# ═══════════════════════════════════════════════════════════════
#                     ИНФОРМАЦИЯ / ЛИСТИНГ
# ═══════════════════════════════════════════════════════════════

class ListDirectoryTool(BaseTool):
    """Список файлов в директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_directory",
            description="Показывает содержимое директории с размерами файлов. Поддерживает фильтры по паттерну.",
            required_args=["directory"],
            optional_args=["pattern", "show_hidden"],
            arg_types={"directory": str, "pattern": str, "show_hidden": bool},
            arg_descriptions={
                "directory": "Путь к директории",
                "pattern": "Фильтр файлов (например: '*.py', '*.txt'). По умолчанию — все файлы",
                "show_hidden": "Показывать скрытые файлы (начинающиеся с точки). По умолчанию — нет",
            },
            category="file",
            examples=[
                'list_directory("/home/user")',
                'list_directory("/home/user/project", "*.py")',
                'list_directory("/etc", show_hidden=True)',
            ],
        )

    async def execute(self, directory: str, pattern: str = "*", show_hidden: bool = False) -> str:
        logger.info(f"Список директории: {directory}")

        path = Path(directory).expanduser()

        is_safe, reason = validate_file_path(path)
        if not is_safe:
            return f"Доступ запрещён: {reason}"

        if not path.exists():
            return f"Директория не найдена: {directory}"

        if not path.is_dir():
            return f"Это не директория: {directory}"

        try:
            dirs = []
            files = []

            for item in sorted(path.glob(pattern)):
                # Пропускаем скрытые если не запрошены
                if not show_hidden and item.name.startswith('.'):
                    continue

                is_item_safe, _ = validate_file_path(item)
                if not is_item_safe:
                    continue

                if item.is_dir():
                    # Считаем элементы внутри
                    try:
                        child_count = sum(1 for _ in item.iterdir())
                    except PermissionError:
                        child_count = "?"
                    dirs.append(f"  [DIR]  {item.name}/  ({child_count} элементов)")
                else:
                    size = _human_size(item.stat().st_size)
                    files.append(f"  [FILE] {item.name}  ({size})")

            total = len(dirs) + len(files)
            if total == 0:
                return f"Папка пуста или нет файлов по паттерну '{pattern}'"

            lines = [f"Директория: {path}  ({total} элементов)", ""]

            if dirs:
                lines.append(f"--- Папки ({len(dirs)}) ---")
                lines.extend(dirs[:30])
                if len(dirs) > 30:
                    lines.append(f"  ... и ещё {len(dirs) - 30} папок")
                lines.append("")

            if files:
                lines.append(f"--- Файлы ({len(files)}) ---")
                lines.extend(files[:50])
                if len(files) > 50:
                    lines.append(f"  ... и ещё {len(files) - 50} файлов")

            return "\n".join(lines)
        except PermissionError:
            return f"Нет прав доступа к: {directory}"
        except Exception as e:
            logger.error(f"Ошибка чтения директории: {e}")
            return f"Ошибка: {e}"


class FileInfoTool(BaseTool):
    """Подробная информация о файле или директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_info",
            description="Показывает подробную информацию: размер, даты создания/изменения, права, тип файла.",
            required_args=["filepath"],
            arg_types={"filepath": str},
            arg_descriptions={
                "filepath": "Путь к файлу или директории",
            },
            category="file",
            examples=[
                'file_info("/home/user/report.pdf")',
                'file_info("/home/user/project")',
            ],
        )

    async def execute(self, filepath: str) -> str:
        logger.info(f"Информация о: {filepath}")

        path = Path(filepath).expanduser()

        is_safe, reason = validate_file_path(path)
        if not is_safe:
            return f"Доступ запрещён: {reason}"

        if not path.exists():
            return f"Не найдено: {filepath}"

        try:
            st = path.stat()

            lines = [f"Информация: {path}", ""]

            # Тип
            if path.is_dir():
                lines.append("Тип: Директория")
                # Размер директории
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                total_files = sum(1 for f in path.rglob('*') if f.is_file())
                total_dirs = sum(1 for f in path.rglob('*') if f.is_dir())
                lines.append(f"Содержимое: {total_files} файлов, {total_dirs} папок")
                lines.append(f"Общий размер: {_human_size(total_size)}")
            elif path.is_file():
                lines.append(f"Тип: Файл ({path.suffix or 'без расширения'})")
                lines.append(f"Размер: {_human_size(st.st_size)}")
            elif path.is_symlink():
                target = path.resolve()
                lines.append(f"Тип: Символическая ссылка → {target}")

            # Даты
            created = datetime.fromtimestamp(st.st_ctime).strftime('%d.%m.%Y %H:%M:%S')
            modified = datetime.fromtimestamp(st.st_mtime).strftime('%d.%m.%Y %H:%M:%S')
            accessed = datetime.fromtimestamp(st.st_atime).strftime('%d.%m.%Y %H:%M:%S')

            lines.append(f"Создан: {created}")
            lines.append(f"Изменён: {modified}")
            lines.append(f"Доступ: {accessed}")

            # Права (Linux)
            if os.name != 'nt':
                mode = stat.filemode(st.st_mode)
                lines.append(f"Права: {mode}")
                lines.append(f"Владелец UID: {st.st_uid}, GID: {st.st_gid}")

            # Полный путь
            lines.append(f"Полный путь: {path.resolve()}")

            return "\n".join(lines)

        except PermissionError:
            return f"Нет прав доступа к: {filepath}"
        except Exception as e:
            logger.error(f"Ошибка получения информации: {e}")
            return f"Ошибка: {e}"


# ═══════════════════════════════════════════════════════════════
#                      ДИРЕКТОРИИ
# ═══════════════════════════════════════════════════════════════

class CreateDirectoryTool(BaseTool):
    """Создание директории"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="create_directory",
            description="Создаёт новую директорию (включая все промежуточные папки).",
            required_args=["directory"],
            arg_types={"directory": str},
            arg_descriptions={
                "directory": "Путь к новой директории",
            },
            category="file",
            examples=[
                'create_directory("/home/user/projects/new_project")',
                'create_directory("/tmp/build/output")',
            ],
        )

    async def execute(self, directory: str) -> str:
        logger.info(f"Создание директории: {directory}")

        path = Path(directory).expanduser()

        is_safe, reason = validate_file_write(path)
        if not is_safe:
            return f"Создание запрещено: {reason}"

        if path.exists():
            return f"Директория уже существует: {directory}"

        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Директория создана: {path}")
            return f"Директория создана: {path}"
        except Exception as e:
            logger.error(f"Ошибка создания директории: {e}")
            return f"Ошибка: {e}"


# ═══════════════════════════════════════════════════════════════
#                        АРХИВЫ
# ═══════════════════════════════════════════════════════════════

class ArchiveTool(BaseTool):
    """Создание архива (zip, tar.gz)"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="archive",
            description="Создаёт архив из файла или директории. Поддерживает форматы: zip, tar.gz, tar.bz2.",
            required_args=["source", "archive_path"],
            optional_args=["format"],
            arg_types={"source": str, "archive_path": str, "format": str},
            arg_descriptions={
                "source": "Путь к файлу или директории для архивации",
                "archive_path": "Путь для создания архива (например: /home/user/backup.zip)",
                "format": "Формат архива: zip, tar.gz, tar.bz2. По умолчанию определяется из расширения",
            },
            arg_enums={"format": ["zip", "tar.gz", "tar.bz2"]},
            category="file",
            examples=[
                'archive("/home/user/project", "/home/user/project_backup.zip")',
                'archive("/home/user/logs", "/tmp/logs.tar.gz", "tar.gz")',
            ],
        )

    async def execute(self, source: str, archive_path: str, format: str = None) -> str:
        logger.info(f"Архивация: {source} → {archive_path}")

        src = Path(source).expanduser()
        dst = Path(archive_path).expanduser()

        is_safe, reason = validate_file_path(src)
        if not is_safe:
            return f"Доступ к источнику запрещён: {reason}"

        is_safe, reason = validate_file_write(dst)
        if not is_safe:
            return f"Запись архива запрещена: {reason}"

        if not src.exists():
            return f"Источник не найден: {source}"

        # Определяем формат
        if format is None:
            if str(dst).endswith('.tar.gz') or str(dst).endswith('.tgz'):
                format = "tar.gz"
            elif str(dst).endswith('.tar.bz2'):
                format = "tar.bz2"
            else:
                format = "zip"

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)

            if format == "zip":
                import zipfile
                with zipfile.ZipFile(str(dst), 'w', zipfile.ZIP_DEFLATED) as zf:
                    if src.is_dir():
                        for file in src.rglob('*'):
                            if file.is_file():
                                zf.write(str(file), file.relative_to(src))
                    else:
                        zf.write(str(src), src.name)

            elif format in ("tar.gz", "tar.bz2"):
                import tarfile
                mode = "w:gz" if format == "tar.gz" else "w:bz2"
                with tarfile.open(str(dst), mode) as tf:
                    tf.add(str(src), arcname=src.name)

            size = _human_size(dst.stat().st_size)
            logger.info(f"Архив создан: {dst} ({size})")
            return f"Архив создан: {dst}\nФормат: {format}\nРазмер: {size}"

        except Exception as e:
            logger.error(f"Ошибка архивации: {e}")
            return f"Ошибка архивации: {e}"


class ExtractArchiveTool(BaseTool):
    """Распаковка архива"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="extract_archive",
            description="Распаковывает архив (zip, tar.gz, tar.bz2, tar) в указанную директорию.",
            required_args=["archive_path"],
            optional_args=["destination"],
            arg_types={"archive_path": str, "destination": str},
            arg_descriptions={
                "archive_path": "Путь к архиву",
                "destination": "Директория для распаковки. По умолчанию — рядом с архивом",
            },
            category="file",
            examples=[
                'extract_archive("/home/user/backup.zip")',
                'extract_archive("/home/user/data.tar.gz", "/home/user/extracted")',
            ],
        )

    async def execute(self, archive_path: str, destination: str = None) -> str:
        logger.info(f"Распаковка: {archive_path}")

        src = Path(archive_path).expanduser()

        is_safe, reason = validate_file_path(src)
        if not is_safe:
            return f"Доступ к архиву запрещён: {reason}"

        if not src.exists():
            return f"Архив не найден: {archive_path}"

        # Определяем назначение
        if destination:
            dst = Path(destination).expanduser()
        else:
            dst = src.parent / src.stem
            # Для .tar.gz убираем двойное расширение
            if src.name.endswith('.tar.gz') or src.name.endswith('.tar.bz2'):
                dst = src.parent / src.name.rsplit('.tar', 1)[0]

        is_safe, reason = validate_file_write(dst)
        if not is_safe:
            return f"Запись в назначение запрещена: {reason}"

        try:
            dst.mkdir(parents=True, exist_ok=True)
            name_lower = src.name.lower()

            if name_lower.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(str(src), 'r') as zf:
                    zf.extractall(str(dst))
                    count = len(zf.namelist())

            elif name_lower.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tar')):
                import tarfile
                with tarfile.open(str(src), 'r:*') as tf:
                    tf.extractall(str(dst), filter='data')
                    count = len(tf.getnames())
            else:
                return f"Неподдерживаемый формат архива: {src.suffix}"

            logger.info(f"Распакован: {src.name} → {dst} ({count} файлов)")
            return f"Архив распакован в: {dst}\nИзвлечено: {count} элементов"

        except Exception as e:
            logger.error(f"Ошибка распаковки: {e}")
            return f"Ошибка распаковки: {e}"


# ═══════════════════════════════════════════════════════════════
#                        УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

def _human_size(size_bytes: int) -> str:
    """Человекочитаемый размер файла"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
