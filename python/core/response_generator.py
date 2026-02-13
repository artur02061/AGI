"""
Кристина 7.0 — ResponseGenerator (Ответы без LLM)

Заменяет LLM-вызов director.synthesize_response() для инструментальных задач.

ДВА УРОВНЯ:
  1. LearnedPatterns — шаблоны, выученные у LLM (растут со временем)
  2. BaseTemplates   — начальные шаблоны (как первые слова ребёнка)

Когда LLM генерирует ответ — ResponseGenerator ЗАПОМИНАЕТ шаблон.
Следующий раз для такого же intent → ответит сам, без LLM.
"""

from typing import Optional, Dict

from utils.logging import get_logger

logger = get_logger("response_gen")


class ResponseGenerator:
    """
    Генерация ответов без LLM.
    Использует шаблоны: сначала выученные, потом базовые.
    """

    def __init__(self, learned_patterns):
        self.learned = learned_patterns
        self._build_base_templates()

    def _build_base_templates(self):
        """
        Базовые шаблоны — начальный словарь Кристины.
        Со временем LearnedPatterns перекроет большинство из них
        более естественными формулировками от LLM.
        """
        self.base_templates: Dict[str, Dict[str, str]] = {
            # ── Файлы ──
            "create_file": {
                "success": "Готово! {result}",
                "error": "Не получилось создать файл: {result}",
            },
            "delete_file": {
                "success": "Готово! {result}",
                "error": "Не удалось удалить файл: {result}",
            },
            "read_file": {
                "success": "Вот содержимое файла:\n{result}",
                "error": "Не удалось прочитать файл: {result}",
            },
            "write_file": {
                "success": "Файл обновлён! {result}",
                "error": "Не удалось записать файл: {result}",
            },
            "append_file": {
                "success": "Текст добавлен! {result}",
                "error": "Ошибка дописывания: {result}",
            },
            "copy_file": {
                "success": "Файл скопирован! {result}",
                "error": "Не удалось скопировать: {result}",
            },
            "move_file": {
                "success": "Файл перемещён! {result}",
                "error": "Не удалось переместить: {result}",
            },
            "rename_file": {
                "success": "Файл переименован! {result}",
                "error": "Не удалось переименовать: {result}",
            },
            "list_directory": {
                "success": "Содержимое:\n{result}",
                "error": "Не удалось прочитать директорию: {result}",
            },
            "create_directory": {
                "success": "Папка создана! {result}",
                "error": "Не удалось создать папку: {result}",
            },
            "search_files": {
                "success": "Результаты поиска:\n{result}",
                "error": "Поиск не удался: {result}",
            },
            "file_info": {
                "success": "Информация о файле:\n{result}",
                "error": "Не удалось получить информацию: {result}",
            },
            "archive": {
                "success": "Архив создан! {result}",
                "error": "Ошибка архивации: {result}",
            },

            # ── Система ──
            "launch_app": {
                "success": "{result}",
                "error": "Не удалось запустить: {result}",
            },
            "kill_process": {
                "success": "{result}",
                "error": "Не удалось завершить процесс: {result}",
            },
            "system_status": {
                "success": "Состояние системы:\n{result}",
            },
            "system_info": {
                "success": "{result}",
            },
            "list_processes": {
                "success": "{result}",
            },
            "disk_usage": {
                "success": "{result}",
            },
            "network_info": {
                "success": "{result}",
            },
            "run_command": {
                "success": "Результат:\n{result}",
                "error": "Ошибка выполнения: {result}",
            },

            # ── Время / Погода / Валюта ──
            "get_current_time": {
                "success": "{result}",
            },
            "get_weather": {
                "success": "{result}",
                "error": "Не удалось получить погоду: {result}",
            },
            "get_currency_rate": {
                "success": "{result}",
                "error": "Не удалось получить курс: {result}",
            },

            # ── Веб ──
            "web_search": {
                "success": "{result}",
                "error": "Поиск не удался: {result}",
            },
            "download_file": {
                "success": "{result}",
                "error": "Не удалось скачать: {result}",
            },

            # ── Память ──
            "recall_memory": {
                "success": "{result}",
                "error": "Не нашла ничего в памяти.",
            },
            "save_note": {
                "success": "Заметка сохранена! {result}",
                "error": "Не удалось сохранить заметку: {result}",
            },
            "list_notes": {
                "success": "Твои заметки:\n{result}",
                "empty": "У тебя пока нет заметок.",
            },

            # ── Буфер обмена ──
            "clipboard_read": {
                "success": "{result}",
            },
            "clipboard_write": {
                "success": "{result}",
            },
        }

    def generate(self, intent: str, tool_result: str) -> Optional[str]:
        """
        Генерирует ответ БЕЗ LLM.

        Порядок:
        1. Ищет выученный шаблон (LearnedPatterns)
        2. Использует базовый шаблон
        3. Возвращает None → нужен LLM

        Returns:
            str — готовый ответ, или None если нужен LLM.
        """

        # ── Tier 1: Выученные шаблоны ──
        learned = self.learned.find_response(intent, tool_result)
        if learned:
            logger.debug(f"✅ Response: learned template for {intent}")
            return learned

        # ── Tier 2: Базовые шаблоны ──
        templates = self.base_templates.get(intent)
        if not templates:
            return None

        result_type = self._classify_result(tool_result)
        template = templates.get(result_type)
        if not template:
            template = templates.get("success", "{result}")

        try:
            response = template.format(result=tool_result)
        except (KeyError, IndexError):
            response = template.replace("{result}", tool_result)

        logger.debug(f"✅ Response: base template for {intent}/{result_type}")
        return response

    def _classify_result(self, result: str) -> str:
        """Классифицирует результат инструмента"""
        if not result or not result.strip():
            return "empty"
        if result.startswith("ERROR") or "ошибка" in result.lower()[:50]:
            return "error"
        return "success"
