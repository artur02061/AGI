"""
Python Fallback: ToolCallParser

Зеркалит API Rust kristina_core.ToolCallParser:
- parse(input) → (name, args, kwargs)
- extract_action(text) → Optional[str]
- extract_final_answer(text) → Optional[str]
- is_final_answer(text) → bool
- is_action(text) → bool
- set_known_tools(tools)
"""

import re
from typing import List, Dict, Optional, Tuple


class ToolCallParser:
    """Парсер вызовов инструментов — Python fallback для Rust"""

    def __init__(self, known_tools: List[str] = None):
        self.known_tools = known_tools or []

    def parse(self, input_str: str) -> Tuple[str, List[str], Dict[str, str]]:
        """
        Парсит tool_name("arg1", "arg2", key="value")
        Возвращает (name, [args], {kwargs})
        """
        input_str = input_str.strip()

        # Ищем имя инструмента
        paren_pos = input_str.find("(")
        if paren_pos == -1:
            raise ValueError(
                f"Нет скобок в вызове: '{input_str}'. "
                f'Формат: tool_name("аргументы")'
            )

        name = input_str[:paren_pos].strip()

        # Проверяем известность
        if self.known_tools and name not in self.known_tools:
            available = ", ".join(self.known_tools)
            raise ValueError(
                f"Инструмент '{name}' не существует. Доступны: {available}"
            )

        # Находим содержимое скобок
        rest = input_str[paren_pos + 1:]
        close_pos = self._find_matching_paren(rest)
        args_str = rest[:close_pos].strip()

        if not args_str:
            return (name, [], {})

        # Парсим аргументы
        args, kwargs = self._parse_arguments(args_str)
        return (name, args, kwargs)

    def extract_action(self, text: str) -> Optional[str]:
        for line in text.splitlines():
            trimmed = line.strip()
            if trimmed.startswith("ACTION:"):
                action = trimmed[len("ACTION:"):].strip()
                if action:
                    return action
        return None

    def extract_final_answer(self, text: str) -> Optional[str]:
        marker = "FINAL_ANSWER:"
        pos = text.find(marker)
        if pos != -1:
            answer = text[pos + len(marker):].strip()
            if answer:
                return answer
        return None

    def is_final_answer(self, text: str) -> bool:
        return "FINAL_ANSWER:" in text

    def is_action(self, text: str) -> bool:
        return "ACTION:" in text

    def set_known_tools(self, tools: List[str]):
        self.known_tools = tools

    # ── Внутренние ──

    @staticmethod
    def _find_matching_paren(s: str) -> int:
        depth = 0
        in_string = False
        string_char = '"'
        escape_next = False

        for i, ch in enumerate(s):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue
            if in_string:
                if ch == string_char:
                    in_string = False
                continue

            if ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                if depth == 0:
                    return i
                depth -= 1

        raise ValueError("Не найдена закрывающая скобка")

    @staticmethod
    def _split_args(s: str) -> List[str]:
        parts = []
        current = []
        in_string = False
        string_char = '"'
        escape_next = False
        paren_depth = 0

        for ch in s:
            if escape_next:
                current.append(ch)
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                current.append(ch)
                continue
            if in_string:
                current.append(ch)
                if ch == string_char:
                    in_string = False
                continue

            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                current.append(ch)
            elif ch == '(':
                paren_depth += 1
                current.append(ch)
            elif ch == ')':
                paren_depth -= 1
                current.append(ch)
            elif ch == ',' and paren_depth == 0:
                part = ''.join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(ch)

        part = ''.join(current).strip()
        if part:
            parts.append(part)
        return parts

    @staticmethod
    def _unquote(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            inner = s[1:-1]
            return (inner
                    .replace('\\"', '"')
                    .replace("\\'", "'")
                    .replace('\\\\', '\\')
                    .replace('\\n', '\n')
                    .replace('\\t', '\t'))
        return s

    def _parse_arguments(self, s: str) -> Tuple[List[str], Dict[str, str]]:
        args = []
        kwargs = {}
        parts = self._split_args(s)

        for part in parts:
            eq_pos = part.find('=')
            # Проверяем что = не внутри кавычек
            if eq_pos != -1 and not self._eq_in_string(part, eq_pos):
                key = part[:eq_pos].strip()
                value = self._unquote(part[eq_pos + 1:].strip())
                kwargs[key] = value
            else:
                args.append(self._unquote(part))

        return (args, kwargs)

    @staticmethod
    def _eq_in_string(s: str, eq_pos: int) -> bool:
        in_string = False
        string_char = '"'
        for i, ch in enumerate(s):
            if i == eq_pos:
                return in_string
            if not in_string and ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif in_string and ch == string_char:
                in_string = False
        return False
