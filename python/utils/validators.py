"""
Валидаторы для инструментов

v6.1 (JARVIS Edition):
- validate_file_path: режим "full" — полный доступ кроме ядерных директорий
- validate_file_write: проверка readonly-директорий
- validate_shell_command: безопасность shell-команд
"""

from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path
import config


class ValidationError(Exception):
    """Ошибка валидации"""
    pass


def validate_tool_call(
    tool_name: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    tool_schema: Dict
) -> Tuple[bool, str]:
    """
    Валидирует вызов инструмента

    Returns:
        (success: bool, error_message: str)
    """

    # Проверяем обязательные аргументы
    required = tool_schema.get('required_args', [])
    total_args = len(args) + len(kwargs)

    if total_args < len(required):
        missing = required[total_args:]
        return False, f"Не хватает аргументов: {', '.join(missing)}"

    # Проверяем типы
    arg_types = tool_schema.get('arg_types', {})

    for i, arg in enumerate(args):
        if i < len(required):
            arg_name = required[i]
            expected_type = arg_types.get(arg_name)

            if expected_type and not isinstance(arg, expected_type):
                return False, f"Аргумент '{arg_name}' должен быть {expected_type.__name__}, получен {type(arg).__name__}"

    return True, "OK"


def _resolve_path(path: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Разрешает путь с проверками безопасности.

    Returns:
        (resolved_path, error_message) — error_message is None on success.
    """
    path_str = str(path)

    # NULL bytes
    if '\x00' in path_str:
        return None, "Путь содержит NULL байт"

    # Длина пути (Linux MAX_PATH = 4096)
    if len(path_str) > 4096:
        return None, "Путь слишком длинный"

    try:
        resolved = path.resolve()
    except Exception as e:
        return None, f"Некорректный путь: {e}"

    # Symlink check
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return None, "Symlink указывает на несуществующий путь"
        is_target_safe, reason = validate_file_path(target)
        if not is_target_safe:
            return None, f"Symlink target небезопасен: {reason}"

    # Path traversal
    for part in path.parts:
        if part == '..':
            return None, "Path traversal запрещён (..)"

    return resolved, None


def _is_under_directory(resolved: Path, directory: Path) -> bool:
    """Проверяет, находится ли resolved внутри directory."""
    try:
        if resolved.is_relative_to(directory):
            return True
    except (ValueError, AttributeError):
        try:
            resolved.relative_to(directory)
            return True
        except ValueError:
            pass
    return False


def validate_file_path(path: Path) -> Tuple[bool, str]:
    """
    Проверяет безопасность пути к файлу (для чтения).

    v6.1: Режим "full" — доступ ко всей системе кроме ядерных директорий.

    Returns:
        (is_safe: bool, reason: str)
    """
    resolved, error = _resolve_path(path)
    if error:
        return False, error

    # Проверка на заблокированные директории (всегда)
    blocked_dirs = getattr(config, 'BLOCKED_DIRECTORIES', [])
    for blocked in blocked_dirs:
        if _is_under_directory(resolved, blocked):
            return False, f"Системная папка запрещена: {blocked}"

    # Проверка расширения
    if resolved.is_file():
        ext = resolved.suffix.lower()
        blocked_ext = getattr(config, 'BLOCKED_EXTENSIONS', [])
        if ext in blocked_ext:
            return False, f"Расширение {ext} запрещено"

    # Проверка режима доступа
    file_access_mode = getattr(config, 'FILE_ACCESS_MODE', 'safe')

    if file_access_mode == "full":
        # v6.1: Полный доступ — только ядерные директории заблокированы (уже проверены выше)
        return True, "OK"

    elif file_access_mode == "safe":
        home = Path.home()
        if not _is_under_directory(resolved, home):
            return False, f"Режим 'safe': доступ разрешён только внутри {home}"

    elif file_access_mode == "restricted":
        allowed_dirs = getattr(config, 'ALLOWED_DIRECTORIES', [])
        is_allowed = any(_is_under_directory(resolved, d) for d in allowed_dirs)
        if not is_allowed:
            return False, "Папка не входит в разрешённые"

    return True, "OK"


def validate_file_write(path: Path) -> Tuple[bool, str]:
    """
    v6.1: Проверяет, можно ли ЗАПИСЫВАТЬ по этому пути.
    Дополнительно к validate_file_path — проверяет readonly-директории.

    Returns:
        (is_safe: bool, reason: str)
    """
    # Сначала базовая проверка
    is_safe, reason = validate_file_path(path)
    if not is_safe:
        return False, reason

    resolved, error = _resolve_path(path)
    if error:
        return False, error

    # Проверка readonly-директорий
    readonly_dirs = getattr(config, 'READONLY_DIRECTORIES', [])
    for ro_dir in readonly_dirs:
        if _is_under_directory(resolved, ro_dir):
            return False, f"Директория только для чтения: {ro_dir}"

    return True, "OK"


def validate_process_name(process_name: str) -> Tuple[bool, str]:
    """Проверяет, можно ли завершить процесс"""

    if not process_name:
        return False, "Имя процесса не указано"

    # Проверка защищённых процессов
    if process_name.lower() in [p.lower() for p in config.PROTECTED_PROCESSES]:
        return False, f"Процесс {process_name} защищён от завершения"

    return True, "OK"


def validate_url(url: str) -> Tuple[bool, str]:
    """Проверяет корректность URL"""

    if not url:
        return False, "URL не указан"

    if not url.startswith(('http://', 'https://')):
        return False, "URL должен начинаться с http:// или https://"

    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False, f"Некорректный URL: {url}"

    # Localhost
    blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
    if host in blocked_hosts:
        return False, f"Доступ к локальным адресам запрещён: {host}"

    # Private IP ranges (RFC 1918 + link-local)
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False, f"Доступ к приватным IP запрещён: {host}"
    except ValueError:
        pass  # hostname, не IP — OK

    return True, "OK"


def validate_shell_command(command: str) -> Tuple[bool, str]:
    """
    v6.1: Проверяет безопасность shell-команды.

    Returns:
        (is_safe: bool, reason: str)
    """
    if not command or not command.strip():
        return False, "Команда не указана"

    if not getattr(config, 'SHELL_ENABLED', False):
        return False, "Выполнение shell-команд отключено в конфигурации"

    cmd_lower = command.lower().strip()

    # Проверка заблокированных команд
    blocked = getattr(config, 'SHELL_BLOCKED_COMMANDS', [])
    for blocked_cmd in blocked:
        if blocked_cmd.lower() in cmd_lower:
            return False, f"Команда заблокирована: {blocked_cmd}"

    # Проверка на опасные паттерны
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf /*",
        "> /dev/sd",
        "mkfs.",
        ":(){ :|:&};:",
        "chmod -R 777 /",
        "chown -R",
    ]

    for pattern in dangerous_patterns:
        if pattern in cmd_lower:
            return False, f"Потенциально опасная команда: содержит '{pattern}'"

    return True, "OK"
