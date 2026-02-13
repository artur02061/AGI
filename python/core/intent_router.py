"""
Кристина 7.0 — IntentRouter (Трёхуровневый роутер)

Заменяет LLM-вызов director.analyze_request() для большинства запросов.

ТРИ УРОВНЯ (от быстрого к медленному):
  Tier 1: LearnedPatterns — паттерны, выученные у LLM (<10мс)
  Tier 2: RuleEngine     — захардкоженные regex правила (<5мс)
  Tier 3: LLM fallback   — director.analyze_request() (~25с)

Каждый раз когда срабатывает Tier 3, результат ЗАПИСЫВАЕТСЯ
в Tier 1 (LearnedPatterns). Со временем Tier 3 вызывается
всё реже и реже.
"""

import re
from typing import Optional, Dict, List, Any

from utils.logging import get_logger

logger = get_logger("intent_router")


class IntentRouter:
    """
    Детерминированный роутер запросов.
    Не использует LLM. Не использует нейросети.
    Чистые алгоритмы: FTS5 поиск + regex паттерны.
    """

    def __init__(self, learned_patterns, tool_names: List[str] = None):
        """
        Args:
            learned_patterns: LearnedPatterns instance (SQLite база паттернов)
            tool_names: список доступных инструментов для валидации
        """
        self.learned = learned_patterns
        self.tool_names = set(tool_names or [])
        self._build_rules()

        logger.info(
            f"🧭 IntentRouter: {len(self._rules)} правил, "
            f"{len(self.tool_names)} инструментов"
        )

    def _build_rules(self):
        """
        Захардкоженные правила (Tier 2).
        Это НАЧАЛЬНОЕ ЗНАНИЕ — как словарь для ребёнка.
        Со временем LearnedPatterns перекроет большинство из них.
        """
        self._rules = [
            # ── Файлы ──
            (re.compile(
                r'(?:создай|сделай|напиши|сгенерируй)\s+'
                r'(?:(?:текстовый|новый)\s+)?'
                r'(?:файл|документ|текст)',
                re.I),
             "create_file", "executor"),

            (re.compile(
                r'(?:удали|убери|сотри|удалить)\s+'
                r'(?:этот\s+)?(?:файл|документ)',
                re.I),
             "delete_file", "executor"),

            (re.compile(
                r'(?:прочитай|прочти|открой|покажи|что\s+в)\s+'
                r'(?:файл[ае]?|документ)',
                re.I),
             "read_file", "executor"),

            (re.compile(
                r'(?:запиши|допиши|добавь)\s+(?:в|к)\s+(?:файл|документ)',
                re.I),
             "append_file", "executor"),

            (re.compile(
                r'(?:скопируй|копируй|копировать)\s+(?:файл|документ)',
                re.I),
             "copy_file", "executor"),

            (re.compile(
                r'(?:перемести|перенеси|перемещ)\s+(?:файл|документ)',
                re.I),
             "move_file", "executor"),

            (re.compile(
                r'(?:переименуй|переименовать)\s+(?:файл|документ)',
                re.I),
             "rename_file", "executor"),

            (re.compile(
                r'(?:покажи|список|что\s+в)\s+'
                r'(?:папк[еу]|директори[юи]|каталог[еу]|рабочем\s+столе)',
                re.I),
             "list_directory", "executor"),

            (re.compile(
                r'(?:создай|сделай)\s+(?:папку|директорию|каталог)',
                re.I),
             "create_directory", "executor"),

            (re.compile(
                r'(?:найди|поищи|поиск)\s+(?:файл[ыа]?)',
                re.I),
             "search_files", "executor"),

            (re.compile(
                r'(?:информаци[яю]|размер|вес|дата)\s+'
                r'(?:о\s+)?(?:файл[ае])',
                re.I),
             "file_info", "executor"),

            (re.compile(
                r'(?:заархивируй|упакуй|архив)',
                re.I),
             "archive", "executor"),

            # ── Система ──
            (re.compile(
                r'(?:запусти|открой|запустить|включи)\s+'
                r'(?:приложение\s+)?(?!файл)([\wа-яёА-ЯЁ]+)',
                re.I),
             "launch_app", "executor"),

            (re.compile(
                r'(?:закрой|заверши|убей|останови)\s+'
                r'(?:процесс|приложение)\s+',
                re.I),
             "kill_process", "executor"),

            (re.compile(
                r'(?:статус|состояние|нагрузка)\s*'
                r'(?:систем|компьютер|пк)?',
                re.I),
             "system_status", "executor"),

            (re.compile(
                r'(?:информаци[яю]|инфо)\s*(?:о\s+)?'
                r'(?:систем[еу]|компьютер[еу]|пк)',
                re.I),
             "system_info", "executor"),

            (re.compile(
                r'(?:процесс[ыа]|запущенные|список\s+процесс)',
                re.I),
             "list_processes", "executor"),

            (re.compile(
                r'(?:мест[оа]\s+на\s+диск|дисков|свободн[оа]\s+на\s+диск)',
                re.I),
             "disk_usage", "executor"),

            (re.compile(
                r'(?:выполни\s+команд|терминал|командн\w+\s+строк)',
                re.I),
             "run_command", "executor"),

            # ── Время / Погода / Валюта ──
            (re.compile(
                r'(?:врем[яю]|час|который\s+час|сколько\s+врем|дат[ау]|'
                r'какой\s+(?:сегодня\s+)?день)',
                re.I),
             "get_current_time", "executor"),

            (re.compile(
                r'(?:погод[аеу]|температур|градус|на\s+улице)',
                re.I),
             "get_weather", "executor"),

            (re.compile(
                r'(?:курс|стоимость)\s+'
                r'(?:доллар|евро|валют|рубл|юан|фунт|USD|EUR|CNY|GBP)',
                re.I),
             "get_currency_rate", "executor"),

            # ── Память / Заметки ──
            (re.compile(
                r'(?:вспомни|напомни|помнишь|что\s+(?:ты\s+)?знаешь)',
                re.I),
             "recall_memory", "executor"),

            (re.compile(
                r'(?:сохрани|запиши|запомни)\s+(?:заметк|замечани)',
                re.I),
             "save_note", "executor"),

            (re.compile(
                r'(?:покажи|список)\s+(?:замет[ок]|заметки)',
                re.I),
             "list_notes", "executor"),

            # ── Веб ──
            (re.compile(
                r'(?:найди|поищи|загугли|погугли|search|'
                r'что\s+(?:такое|значит)|(?:кто\s+(?:такой|такая)))',
                re.I),
             "web_search", "analyst"),

            (re.compile(
                r'(?:скачай|загрузи|download)\s+(?:файл\s+)?(?:с|из|по)',
                re.I),
             "download_file", "executor"),

            # ── Диалог (director, без инструмента) ──
            (re.compile(
                r'^(?:привет|здравствуй|хай|hello|добр\w+\s+'
                r'(?:утро|день|вечер)|как\s+дела)',
                re.I),
             "greeting", "director"),

            (re.compile(
                r'(?:расскажи|объясни|почему|зачем|как\s+работает)',
                re.I),
             "explanation", "director"),

            (re.compile(
                r'(?:придумай|сочини|напиши\s+(?:стих|рассказ|историю|'
                r'пожелани|поздравлени))',
                re.I),
             "creative", "director"),
        ]

    def route(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Определяет intent без LLM.

        Returns:
            Dict с планом или None (→ нужен LLM).

            Если Dict:
              - intent: str      — имя инструмента
              - agent: str       — executor/analyst/director
              - confidence: float
              - source: str      — 'learned' | 'rule'
              - pattern_id: int? — ID паттерна (для reinforce/weaken)
              - slots: dict      — извлечённые аргументы
        """

        # ── Tier 1: Выученные паттерны (<10мс) ──
        learned_result = self.learned.find_routing(user_input)
        if learned_result and learned_result["confidence"] >= 0.7:
            # Также пытаемся извлечь аргументы
            slots = self.learned.find_slots(
                learned_result["intent"], user_input
            )
            learned_result["slots"] = slots
            logger.debug(
                f"✅ Tier 1 (learned): {learned_result['intent']} "
                f"(conf={learned_result['confidence']:.2f})"
            )
            return learned_result

        # ── Tier 2: Захардкоженные правила (<5мс) ──
        for pattern, intent, agent in self._rules:
            if pattern.search(user_input):
                # Валидация: intent должен быть реальным инструментом
                # (кроме director-специфичных как greeting, explanation, creative)
                if agent == "executor" and intent not in self.tool_names:
                    continue

                slots = self._extract_slots_by_rules(intent, user_input)

                result = {
                    "intent": intent,
                    "agent": agent,
                    "confidence": 0.85,
                    "source": "rule",
                    "pattern_id": None,
                    "slots": slots,
                }
                logger.debug(f"✅ Tier 2 (rule): {intent}")
                return result

        # ── Ничего не нашли → Tier 3 (LLM) ──
        logger.debug(f"⚠️ Tier 1+2 miss, нужен LLM для: '{user_input[:50]}'")
        return None

    def _extract_slots_by_rules(self, intent: str, user_input: str) -> Dict[str, str]:
        """
        Извлечение аргументов из текста правилами.

        Сначала пробует learned slots, потом захардкоженные regex.
        """
        # Tier 1: Выученные slot-паттерны
        slots = self.learned.find_slots(intent, user_input)
        if slots:
            return slots

        # Tier 2: Базовые правила
        slots = {}

        if intent in ("create_file", "read_file", "delete_file",
                       "write_file", "append_file", "file_info"):
            # Ищем имя файла
            match = re.search(
                r'([\wа-яёА-ЯЁ\-]+\.[\wа-яёА-ЯЁ]+)', user_input, re.I
            )
            if match:
                slots["filepath"] = match.group(1)

        if intent == "create_file":
            # Ищем содержимое после ключевых слов
            for pattern in [
                r'(?:с\s+(?:текстом|содержимым|содержанием))\s*[:\-]?\s*(.+)',
                r'(?:напиши|написать)\s*[:\-]?\s*(.+)',
                r'(?:содержимое|текст)\s*[:\-]?\s*(.+)',
            ]:
                match = re.search(pattern, user_input, re.I)
                if match:
                    slots["content"] = match.group(1).strip()
                    break

        if intent == "launch_app":
            match = re.search(
                r'(?:запусти|открой|включи)\s+(?:приложение\s+)?'
                r'([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["app_name"] = match.group(1)

        if intent == "get_weather":
            match = re.search(
                r'(?:погод[аеу]|температур\w*)\s+(?:в\s+)?([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["city"] = match.group(1)

        if intent == "web_search":
            # Всё после ключевого слова — запрос
            match = re.search(
                r'(?:найди|поищи|загугли|погугли)\s+(.+)',
                user_input, re.I
            )
            if match:
                slots["query"] = match.group(1).strip()

        if intent == "kill_process":
            match = re.search(
                r'(?:закрой|заверши|убей)\s+(?:процесс\s+)?'
                r'([\wа-яёА-ЯЁ]+)',
                user_input, re.I
            )
            if match:
                slots["process_name"] = match.group(1)

        if intent == "get_currency_rate":
            match = re.search(
                r'(доллар|евро|юан|фунт|USD|EUR|CNY|GBP|JPY)',
                user_input, re.I
            )
            if match:
                mapping = {
                    "доллар": "USD", "евро": "EUR", "юан": "CNY",
                    "фунт": "GBP",
                }
                raw = match.group(1)
                slots["currency"] = mapping.get(raw.lower(), raw.upper())

        return slots
