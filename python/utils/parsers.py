# DEPRECATED v6.0: Regex-based parsing removed. v6.0 uses native tool calling.
"""
Парсеры для tool calls

⚠️ DEPRECATED в v6.0: AgentCore теперь использует нативный Ollama tool calling.
Этот модуль оставлен для обратной совместимости с Orchestrator/multi-agent flow.
При полном переходе на native tool calling — можно удалить.
"""

import shlex
import re
from typing import Tuple, List, Dict, Any, Optional

class ParseError(Exception):
    """Ошибка парсинга"""
    pass

def parse_tool_call(action_str: str) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Парсит строку вызова инструмента
    
    Формат: tool_name("arg1", "arg2", kwarg="value")
    
    Returns:
        (tool_name, args, kwargs)
    
    Raises:
        ParseError: Если формат некорректен
    """
    
    # Регулярка: tool_name(...)
    match = re.match(r'(\w+)\((.*)\)', action_str.strip(), re.DOTALL)
    
    if not match:
        raise ParseError(
            f"Неверный формат. Ожидается: tool_name(\"args\")\n"
            f"Получено: {action_str}"
        )
    
    tool_name = match.group(1)
    args_str = match.group(2).strip()
    
    args = []
    kwargs = {}
    
    if args_str:
        try:
            # Используем shlex для безопасного парсинга
            tokens = shlex.split(args_str)
            
            for token in tokens:
                # Keyword argument
                if '=' in token and not token.startswith('http'):
                    key, value = token.split('=', 1)
                    kwargs[key.strip()] = value.strip()
                else:
                    args.append(token)
        
        except ValueError as e:
            raise ParseError(f"Ошибка парсинга аргументов: {e}")
    
    return tool_name, args, kwargs

def extract_action(text: str) -> Optional[str]:
    """
    Извлекает строку ACTION из текста агента
    
    Returns:
        action_str или None
    """
    
    upper_text = text.upper()
    
    if "ACTION:" not in upper_text:
        return None
    
    # Находим позицию ACTION:
    idx = upper_text.index("ACTION:")
    after_action = text[idx + 7:].strip()
    
    # Берём первую строку
    first_line = after_action.split("\n")[0].strip()
    
    # Убираем markdown
    first_line = first_line.replace("**", "").replace("*", "").strip()
    
    return first_line if first_line else None

def extract_final_answer(text: str) -> str:
    """Извлекает FINAL_ANSWER из текста"""
    
    markers = [
        "FINAL_ANSWER:",
        "FINAL ANSWER:",
        "ИТОГОВЫЙ ОТВЕТ:",
        "МОЙ ОТВЕТ:"
    ]
    
    upper_text = text.upper()
    
    for marker in markers:
        if marker in upper_text:
            idx = upper_text.index(marker) + len(marker)
            answer = text[idx:].strip()
            return answer
    
    # Fallback
    return text.strip()

def is_final_answer(text: str) -> bool:
    """Проверяет наличие FINAL_ANSWER"""
    
    upper = text.upper()
    return any(marker in upper for marker in [
        "FINAL_ANSWER:",
        "FINAL ANSWER:",
        "ИТОГОВЫЙ ОТВЕТ:",
        "МОЙ ОТВЕТ:"
    ])

def is_action(text: str) -> bool:
    """Проверяет наличие ACTION"""
    return "ACTION:" in text.upper()