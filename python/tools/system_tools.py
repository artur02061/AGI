"""
Системные инструменты — JARVIS Edition v6.1

Полный контроль над системой:
- Запуск/завершение приложений и процессов
- Мониторинг CPU/RAM/GPU/Диска
- Выполнение shell-команд
- Информация о системе, сети, дисках
- Открытие файлов системными приложениями
- Управление буфером обмена
"""

import os
import platform
import asyncio
from typing import Dict, Any

from modules.system_control.controller import SystemController
from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
from utils.validators import validate_process_name, validate_shell_command
import config

logger = get_logger("system_tools")


# ═══════════════════════════════════════════════════════════════
#                     ПРИЛОЖЕНИЯ
# ═══════════════════════════════════════════════════════════════

class LaunchAppTool(BaseTool):
    """Запускает приложение"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="launch_app",
            description="Запускает приложение или игру на компьютере по имени.",
            required_args=["app_name"],
            arg_types={"app_name": str},
            arg_descriptions={
                "app_name": "Название приложения (например: steam, chrome, discord, vscode, telegram)",
            },
            category="system",
            examples=[
                'launch_app("steam")',
                'launch_app("chrome")',
                'launch_app("discord")',
            ],
        )

    async def execute(self, app_name: str) -> str:
        logger.info(f"Запуск приложения: {app_name}")
        result = await self.controller.launch_app(app_name)
        if result["success"]:
            logger.info(f"Запущено: {result['message']}")
        else:
            logger.warning(f"Ошибка запуска: {result['message']}")
        return result["message"]


class SearchAppsTool(BaseTool):
    """Поиск установленных приложений"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_apps",
            description="Ищет установленные приложения на компьютере по названию.",
            required_args=["query"],
            arg_types={"query": str},
            arg_descriptions={
                "query": "Название или часть названия приложения для поиска",
            },
            category="system",
            examples=[
                'search_apps("steam")',
                'search_apps("браузер")',
                'search_apps("офис")',
            ],
        )

    async def execute(self, query: str) -> str:
        logger.info(f"Поиск приложений: {query}")
        result = await self.controller.search_apps(query)
        return result["message"]


class OpenFileTool(BaseTool):
    """Открытие файла системным приложением"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="open_file",
            description="Открывает файл или URL системным приложением по умолчанию (браузер, текстовый редактор, просмотрщик и т.д.).",
            required_args=["filepath"],
            arg_types={"filepath": str},
            arg_descriptions={
                "filepath": "Путь к файлу или URL для открытия",
            },
            category="system",
            examples=[
                'open_file("/home/user/report.pdf")',
                'open_file("~/Documents/image.png")',
                'open_file("https://github.com")',
            ],
        )

    async def execute(self, filepath: str) -> str:
        logger.info(f"Открытие: {filepath}")
        result = await self.controller.open_file(filepath)
        if result["success"]:
            logger.info(f"Открыт: {result['message']}")
        else:
            logger.warning(f"Ошибка: {result['message']}")
        return result["message"]


# ═══════════════════════════════════════════════════════════════
#                     ПРОЦЕССЫ
# ═══════════════════════════════════════════════════════════════

class ListProcessesTool(BaseTool):
    """Список процессов"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_processes",
            description="Показывает список запущенных процессов с информацией о нагрузке CPU и RAM.",
            required_args=[],
            optional_args=["filter_keyword", "limit"],
            arg_types={"filter_keyword": str, "limit": int},
            arg_descriptions={
                "filter_keyword": "Фильтр по имени процесса (например: 'chrome', 'python')",
                "limit": "Максимальное количество процессов для отображения (по умолчанию 10)",
            },
            category="system",
            examples=[
                'list_processes()',
                'list_processes("chrome")',
                'list_processes(limit=20)',
            ],
        )

    async def execute(self, filter_keyword: str = None, limit: int = 10) -> str:
        logger.debug(f"Получение процессов (фильтр: {filter_keyword}, лимит: {limit})")

        processes = await self.controller.list_processes(
            filter_keyword=filter_keyword,
            limit=limit,
        )

        if not processes:
            if filter_keyword:
                return f"Процессов с именем '{filter_keyword}' не найдено"
            return "Не удалось получить список процессов"

        lines = ["Топ процессов по нагрузке:", ""]
        lines.append(f"{'Имя':<30} {'CPU':>7} {'RAM':>7}")
        lines.append("-" * 46)

        for proc in processes[:limit]:
            name = proc['name'][:30]
            cpu = proc['cpu']
            mem = proc['memory']
            lines.append(f"{name:<30} {cpu:5.1f}%  {mem:5.1f}%")

        return "\n".join(lines)


class KillProcessTool(BaseTool):
    """Завершение процесса"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="kill_process",
            description="Завершает процесс по имени. Защищённые системные процессы не могут быть завершены.",
            required_args=["process_name"],
            arg_types={"process_name": str},
            arg_descriptions={
                "process_name": "Имя процесса для завершения (например: 'chrome', 'firefox')",
            },
            category="system",
            danger_level="warning",
            requires_confirmation=True,
            examples=[
                'kill_process("chrome")',
                'kill_process("firefox")',
            ],
        )

    async def execute(self, process_name: str) -> str:
        logger.info(f"Завершение процесса: {process_name}")

        is_valid, reason = validate_process_name(process_name)
        if not is_valid:
            return f"Отказано: {reason}"

        result = await self.controller.kill_process(process_name)
        if result["success"]:
            logger.info(f"Завершён: {result['message']}")
        else:
            logger.warning(f"Ошибка: {result['message']}")
        return result["message"]


# ═══════════════════════════════════════════════════════════════
#                     МОНИТОРИНГ
# ═══════════════════════════════════════════════════════════════

class SystemStatusTool(BaseTool):
    """Показывает статус системы"""

    def __init__(self, system_controller):
        super().__init__()
        self.controller = system_controller

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="system_status",
            description="Показывает текущую нагрузку системы: CPU, RAM, GPU, температура.",
            required_args=[],
            category="system",
            examples=['system_status()'],
        )

    async def execute(self) -> str:
        logger.debug("Получение статуса системы")

        status = await self.controller.get_system_status()

        cpu = status['cpu']['usage_percent']
        ram = status['ram']['usage_percent']

        result = f"CPU: {cpu:.1f}%\nRAM: {ram:.1f}%"

        # GPU если доступен
        gpu = status.get('gpu', {})
        if 'usage_percent' in gpu and gpu.get('error') != "Недоступен":
            result += f"\nGPU: {gpu['usage_percent']}%"
            if 'temperature_c' in gpu:
                result += f" ({gpu['temperature_c']}°C)"

        # Предупреждения
        warnings = []
        if cpu > config.CPU_WARNING_THRESHOLD:
            warnings.append("CPU перегружен!")
        if ram > config.RAM_WARNING_THRESHOLD:
            warnings.append("RAM заполнена!")

        if warnings:
            result += "\n\nВнимание: " + ", ".join(warnings)

        return result


class SystemInfoTool(BaseTool):
    """Подробная информация о системе"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="system_info",
            description="Показывает подробную информацию о системе: ОС, процессор, RAM, uptime, Python версия.",
            required_args=[],
            category="system",
            examples=['system_info()'],
        )

    async def execute(self) -> str:
        logger.debug("Получение информации о системе")

        import psutil
        from datetime import datetime

        lines = ["=== Информация о системе ===", ""]

        # ОС
        uname = platform.uname()
        lines.append(f"ОС: {uname.system} {uname.release}")
        lines.append(f"Версия: {uname.version}")
        lines.append(f"Имя хоста: {uname.node}")
        lines.append(f"Архитектура: {uname.machine}")

        # CPU
        lines.append(f"\nПроцессор: {uname.processor or platform.processor() or 'N/A'}")
        lines.append(f"Ядра (физические): {psutil.cpu_count(logical=False)}")
        lines.append(f"Ядра (логические): {psutil.cpu_count(logical=True)}")
        try:
            freq = psutil.cpu_freq()
            if freq:
                lines.append(f"Частота: {freq.current:.0f} MHz (макс: {freq.max:.0f} MHz)")
        except Exception:
            pass

        # RAM
        mem = psutil.virtual_memory()
        lines.append(f"\nRAM: {mem.total / (1024**3):.1f} GB всего")
        lines.append(f"RAM используется: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
        lines.append(f"RAM доступно: {mem.available / (1024**3):.1f} GB")

        # Swap
        swap = psutil.swap_memory()
        if swap.total > 0:
            lines.append(f"Swap: {swap.total / (1024**3):.1f} GB (используется: {swap.percent}%)")

        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        lines.append(f"\nUptime: {hours} ч {minutes} мин")
        lines.append(f"Загрузка: {boot_time.strftime('%d.%m.%Y %H:%M')}")

        # Python
        lines.append(f"\nPython: {platform.python_version()}")

        return "\n".join(lines)


class DiskUsageTool(BaseTool):
    """Информация о дисках"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="disk_usage",
            description="Показывает информацию о дисках: размер, использование, свободное место.",
            required_args=[],
            optional_args=["path"],
            arg_types={"path": str},
            arg_descriptions={
                "path": "Путь для проверки конкретного диска/раздела. По умолчанию — все диски",
            },
            category="system",
            examples=[
                'disk_usage()',
                'disk_usage("/home")',
            ],
        )

    async def execute(self, path: str = None) -> str:
        logger.debug("Получение информации о дисках")

        import psutil

        if path:
            try:
                usage = psutil.disk_usage(path)
                total = usage.total / (1024**3)
                used = usage.used / (1024**3)
                free = usage.free / (1024**3)
                return (
                    f"Диск: {path}\n"
                    f"Всего: {total:.1f} GB\n"
                    f"Используется: {used:.1f} GB ({usage.percent}%)\n"
                    f"Свободно: {free:.1f} GB"
                )
            except Exception as e:
                return f"Ошибка: {e}"

        lines = ["=== Диски ===", ""]
        lines.append(f"{'Раздел':<20} {'Размер':>8} {'Исп.':>8} {'Своб.':>8} {'%':>5}")
        lines.append("-" * 53)

        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                total = usage.total / (1024**3)
                used = usage.used / (1024**3)
                free = usage.free / (1024**3)

                mount = part.mountpoint
                if len(mount) > 18:
                    mount = "..." + mount[-15:]

                lines.append(
                    f"{mount:<20} {total:>6.1f}G {used:>6.1f}G {free:>6.1f}G {usage.percent:>4.0f}%"
                )
            except (PermissionError, OSError):
                continue

        return "\n".join(lines)


class NetworkInfoTool(BaseTool):
    """Информация о сети"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="network_info",
            description="Показывает информацию о сетевых интерфейсах, IP-адресах и сетевой активности.",
            required_args=[],
            category="system",
            examples=['network_info()'],
        )

    async def execute(self) -> str:
        logger.debug("Получение сетевой информации")

        import psutil
        import socket

        lines = ["=== Сеть ===", ""]

        # Hostname
        lines.append(f"Hostname: {socket.gethostname()}")

        # Сетевые интерфейсы
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()

        for iface, addr_list in addrs.items():
            iface_stat = stats.get(iface)
            status = "UP" if iface_stat and iface_stat.isup else "DOWN"

            for addr in addr_list:
                if addr.family == socket.AF_INET:  # IPv4
                    lines.append(f"\n{iface} ({status}):")
                    lines.append(f"  IPv4: {addr.address}")
                    if addr.netmask:
                        lines.append(f"  Маска: {addr.netmask}")
                    if iface_stat:
                        speed = iface_stat.speed
                        if speed > 0:
                            lines.append(f"  Скорость: {speed} Mbps")
                    break

        # Сетевой трафик
        counters = psutil.net_io_counters()
        sent_mb = counters.bytes_sent / (1024**2)
        recv_mb = counters.bytes_recv / (1024**2)
        lines.append(f"\nТрафик за сессию:")
        lines.append(f"  Отправлено: {sent_mb:.1f} MB")
        lines.append(f"  Получено: {recv_mb:.1f} MB")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#                     SHELL / КОМАНДЫ
# ═══════════════════════════════════════════════════════════════

class RunCommandTool(BaseTool):
    """Выполнение shell-команды"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description=(
                "Выполняет shell-команду в терминале и возвращает результат. "
                "Позволяет выполнить любую команду: ls, grep, git, pip, apt, docker, и т.д. "
                "Опасные команды (rm -rf /, mkfs) заблокированы."
            ),
            required_args=["command"],
            optional_args=["timeout", "working_dir"],
            arg_types={"command": str, "timeout": int, "working_dir": str},
            arg_descriptions={
                "command": "Shell-команда для выполнения (например: 'ls -la', 'git status', 'pip list')",
                "timeout": "Таймаут выполнения в секундах (по умолчанию 30)",
                "working_dir": "Рабочая директория для выполнения команды",
            },
            category="system",
            danger_level="warning",
            examples=[
                'run_command("ls -la /home/user")',
                'run_command("git status")',
                'run_command("pip list")',
                'run_command("df -h")',
                'run_command("docker ps")',
                'run_command("cat /etc/hostname")',
            ],
        )

    async def execute(self, command: str, timeout: int = None, working_dir: str = None) -> str:
        logger.info(f"Выполнение команды: {command}")

        # Валидация
        is_safe, reason = validate_shell_command(command)
        if not is_safe:
            logger.warning(f"Команда заблокирована: {reason}")
            return f"Команда заблокирована: {reason}"

        shell_timeout = timeout or getattr(config, 'SHELL_TIMEOUT', 30)
        max_output = getattr(config, 'SHELL_MAX_OUTPUT', 10000)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=shell_timeout,
            )

            output_parts = []

            # stdout
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='replace')
                if len(stdout_text) > max_output:
                    stdout_text = stdout_text[:max_output] + f"\n[...обрезано, полный вывод: {len(stdout_text)} символов]"
                output_parts.append(stdout_text)

            # stderr (если есть ошибки)
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='replace')
                if stderr_text.strip():
                    if len(stderr_text) > max_output:
                        stderr_text = stderr_text[:max_output] + "\n[...обрезано]"
                    output_parts.append(f"[stderr]\n{stderr_text}")

            # Exit code
            if process.returncode != 0:
                output_parts.append(f"\n[exit code: {process.returncode}]")

            result = "\n".join(output_parts) if output_parts else "(команда выполнена, вывод пуст)"

            logger.info(f"Команда выполнена (exit: {process.returncode})")
            return result

        except asyncio.TimeoutError:
            logger.warning(f"Таймаут команды ({shell_timeout}с): {command}")
            return f"Команда превысила таймаут ({shell_timeout} секунд)"
        except Exception as e:
            logger.error(f"Ошибка выполнения команды: {e}")
            return f"Ошибка выполнения: {e}"


# ═══════════════════════════════════════════════════════════════
#                     БУФЕР ОБМЕНА
# ═══════════════════════════════════════════════════════════════

class ClipboardReadTool(BaseTool):
    """Чтение из буфера обмена"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="clipboard_read",
            description="Читает текущее содержимое буфера обмена (Ctrl+C).",
            required_args=[],
            category="system",
            examples=['clipboard_read()'],
        )

    async def execute(self) -> str:
        logger.debug("Чтение буфера обмена")

        try:
            # Пробуем разные способы
            if platform.system() == "Linux":
                proc = await asyncio.create_subprocess_shell(
                    "xclip -selection clipboard -o 2>/dev/null || xsel --clipboard --output 2>/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if stdout:
                    content = stdout.decode('utf-8', errors='replace')
                    if len(content) > 5000:
                        content = content[:5000] + "\n[...обрезано]"
                    return f"Буфер обмена:\n{content}"
                return "Буфер обмена пуст или xclip/xsel не установлен"

            elif platform.system() == "Darwin":  # macOS
                proc = await asyncio.create_subprocess_shell(
                    "pbpaste",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                content = stdout.decode('utf-8', errors='replace')
                return f"Буфер обмена:\n{content}" if content else "Буфер обмена пуст"

            elif platform.system() == "Windows":
                proc = await asyncio.create_subprocess_shell(
                    "powershell -command Get-Clipboard",
                    stdout=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                content = stdout.decode('utf-8', errors='replace')
                return f"Буфер обмена:\n{content}" if content else "Буфер обмена пуст"

            return "Чтение буфера обмена не поддерживается на данной ОС"

        except Exception as e:
            logger.error(f"Ошибка чтения буфера: {e}")
            return f"Ошибка чтения буфера обмена: {e}"


class ClipboardWriteTool(BaseTool):
    """Запись в буфер обмена"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="clipboard_write",
            description="Копирует текст в буфер обмена (эквивалент Ctrl+C).",
            required_args=["text"],
            arg_types={"text": str},
            arg_descriptions={
                "text": "Текст для копирования в буфер обмена",
            },
            category="system",
            examples=[
                'clipboard_write("Привет, мир!")',
                'clipboard_write("git commit -m \\"fix: resolved issue\\"")  ',
            ],
        )

    async def execute(self, text: str) -> str:
        logger.debug("Запись в буфер обмена")

        try:
            if platform.system() == "Linux":
                proc = await asyncio.create_subprocess_shell(
                    "xclip -selection clipboard 2>/dev/null || xsel --clipboard --input 2>/dev/null",
                    stdin=asyncio.subprocess.PIPE,
                )
                await proc.communicate(input=text.encode('utf-8'))
                return f"Скопировано в буфер обмена ({len(text)} символов)"

            elif platform.system() == "Darwin":
                proc = await asyncio.create_subprocess_shell(
                    "pbcopy",
                    stdin=asyncio.subprocess.PIPE,
                )
                await proc.communicate(input=text.encode('utf-8'))
                return f"Скопировано в буфер обмена ({len(text)} символов)"

            elif platform.system() == "Windows":
                proc = await asyncio.create_subprocess_shell(
                    "clip",
                    stdin=asyncio.subprocess.PIPE,
                )
                await proc.communicate(input=text.encode('utf-8'))
                return f"Скопировано в буфер обмена ({len(text)} символов)"

            return "Запись в буфер обмена не поддерживается на данной ОС"

        except Exception as e:
            logger.error(f"Ошибка записи в буфер: {e}")
            return f"Ошибка записи в буфер обмена: {e}"


# ═══════════════════════════════════════════════════════════════
#                    ПЕРЕМЕННЫЕ СРЕДЫ
# ═══════════════════════════════════════════════════════════════

class GetEnvTool(BaseTool):
    """Чтение переменных окружения"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="get_env",
            description="Читает переменную окружения по имени или показывает все переменные.",
            required_args=[],
            optional_args=["name"],
            arg_types={"name": str},
            arg_descriptions={
                "name": "Имя переменной (например: PATH, HOME, USER). Если не указано — показывает все",
            },
            category="system",
            examples=[
                'get_env("PATH")',
                'get_env("HOME")',
                'get_env()',
            ],
        )

    async def execute(self, name: str = None) -> str:
        if name:
            value = os.environ.get(name)
            if value is None:
                return f"Переменная '{name}' не найдена"
            # Маскируем потенциально секретные переменные
            secret_keywords = ['password', 'secret', 'token', 'key', 'api_key']
            if any(kw in name.lower() for kw in secret_keywords):
                return f"{name} = ***скрыто***"
            return f"{name} = {value}"

        # Показываем все (кроме секретных)
        lines = ["Переменные окружения:", ""]
        secret_keywords = ['password', 'secret', 'token', 'key', 'api_key', 'credential']

        for key in sorted(os.environ.keys()):
            if any(kw in key.lower() for kw in secret_keywords):
                lines.append(f"  {key} = ***скрыто***")
            else:
                value = os.environ[key]
                if len(value) > 100:
                    value = value[:100] + "..."
                lines.append(f"  {key} = {value}")

        return "\n".join(lines[:50])  # Ограничиваем вывод


# GetCurrentTimeTool и GetWeatherTool — в tools/web_tools.py
