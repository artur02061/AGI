"""
Валидаторы для инструментов
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

def validate_file_path(path: Path) -> Tuple[bool, str]:
    """
    Проверяет безопасность пути к файлу

    ИСПРАВЛЕНО v6.0:
    - ✅ Проверка symlink (resolved vs original)
    - ✅ Проверка path traversal (.. в компонентах)
    - ✅ Проверка NULL bytes
    - ✅ Проверка длины пути

    Returns:
        (is_safe: bool, reason: str)
    """

    path_str = str(path)

    # NULL bytes
    if '\x00' in path_str:
        return False, "Путь содержит NULL байт"

    # Длина пути (Linux MAX_PATH = 4096)
    if len(path_str) > 4096:
        return False, "Путь слишком длинный"

    try:
        resolved = path.resolve()
    except Exception as e:
        return False, f"Некорректный путь: {e}"

    # Symlink check: resolved path должен быть в разрешённой зоне
    # Это предотвращает: symlink /tmp/safe -> /etc/shadow
    if path.is_symlink():
        # Разрешаем symlink только если target тоже безопасен
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return False, "Symlink указывает на несуществующий путь"
        # Рекурсивная проверка target
        is_target_safe, reason = validate_file_path(target)
        if not is_target_safe:
            return False, f"Symlink target небезопасен: {reason}"

    # Path traversal: запрещаем .. в компонентах
    for part in path.parts:
        if part == '..':
            return False, "Path traversal запрещён (..)"

    # Проверка на системные папки
    blocked_dirs = getattr(config, 'BLOCKED_DIRECTORIES', [])
    for blocked in blocked_dirs:
        try:
            if resolved.is_relative_to(blocked):
                return False, f"Системная папка запрещена: {blocked.name}"
        except (ValueError, AttributeError):
            try:
                resolved.relative_to(blocked)
                return False, f"Системная папка запрещена: {blocked.name}"
            except ValueError:
                continue

    # Проверка расширения
    if resolved.is_file():
        ext = resolved.suffix.lower()
        blocked_ext = getattr(config, 'BLOCKED_EXTENSIONS', [])
        if ext in blocked_ext:
            return False, f"Расширение {ext} запрещено"

    # Проверка режима restricted
    file_access_mode = getattr(config, 'FILE_ACCESS_MODE', 'unrestricted')
    if file_access_mode == "restricted":
        is_allowed = False
        allowed_dirs = getattr(config, 'ALLOWED_DIRECTORIES', [])

        for allowed in allowed_dirs:
            try:
                if resolved.is_relative_to(allowed):
                    is_allowed = True
                    break
            except (ValueError, AttributeError):
                try:
                    resolved.relative_to(allowed)
                    is_allowed = True
                    break
                except ValueError:
                    continue

        if not is_allowed:
            return False, "Папка не входит в разрешённые"

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
    
    # Проверка на подозрительные домены (опционально)
    suspicious = ['localhost', '127.0.0.1', '0.0.0.0']
    if any(s in url.lower() for s in suspicious):
        return False, f"Подозрительный URL: {url}"
    
    return True, "OK"