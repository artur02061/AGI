"""
Инструменты для работы с памятью и заметками — JARVIS Edition v6.1

- Поиск в краткосрочной/долговременной памяти
- Создание, просмотр и удаление персональных заметок
"""

import json
from pathlib import Path
from datetime import datetime

from tools.base import BaseTool, ToolSchema
from utils.logging import get_logger
import config

logger = get_logger("memory_tools")


# ═══════════════════════════════════════════════════════════════
#                     ПАМЯТЬ АГЕНТА
# ═══════════════════════════════════════════════════════════════

class RecallMemoryTool(BaseTool):
    """Поиск в эпизодической памяти"""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="recall_memory",
            description="Ищет информацию в краткосрочной памяти — последние диалоги и взаимодействия.",
            required_args=["query"],
            arg_types={"query": str},
            arg_descriptions={
                "query": "Что искать в памяти (например: 'что мы обсуждали про проект', 'имя кота пользователя')",
            },
            category="memory",
            examples=[
                'recall_memory("что мы обсуждали про погоду")',
                'recall_memory("игра которую я упоминал")',
                'recall_memory("мой день рождения")',
            ],
        )

    async def execute(self, query: str) -> str:
        logger.info(f"Поиск в памяти: {query}")

        context = self.memory.get_relevant_context(query, max_items=5)

        if context == "Нет релевантного контекста.":
            return "Ничего не нашла в краткосрочной памяти по этому запросу."

        return f"Из краткосрочной памяти:\n\n{context}"


class SearchMemoryTool(BaseTool):
    """Поиск в векторной памяти"""

    def __init__(self, vector_memory):
        super().__init__()
        self.vector_memory = vector_memory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="search_memory",
            description="Ищет информацию в долговременной памяти — все прошлые разговоры, факты и события.",
            required_args=["query"],
            optional_args=["timeframe"],
            arg_types={"query": str, "timeframe": str},
            arg_descriptions={
                "query": "Что искать в долговременной памяти",
                "timeframe": "Временной фильтр: 'today', 'last_week', 'last_month'. По умолчанию — без фильтра",
            },
            arg_enums={"timeframe": ["today", "last_week", "last_month"]},
            category="memory",
            examples=[
                'search_memory("проект о котором говорили")',
                'search_memory("курс доллара", "last_week")',
                'search_memory("игры в которые я играл")',
            ],
        )

    async def execute(self, query: str, timeframe: str = None) -> str:
        logger.info(f"Поиск в долговременной памяти: {query} (timeframe: {timeframe})")

        if timeframe:
            results = self.vector_memory.search_by_timeframe(
                query,
                timeframe=timeframe,
                n_results=5,
            )
        else:
            results = self.vector_memory.search(
                query,
                n_results=5,
                filter_metadata={"type": "dialogue"},
            )

        if not results:
            return "Ничего не нашла в долговременной памяти по этому запросу."

        lines = ["Из долговременной памяти:", ""]

        for r in results:
            date = r['metadata'].get('date', 'н/д')
            text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']

            lines.append(f"[{date}]")
            lines.append(f"  {text}")
            lines.append("")

        logger.info(f"Найдено {len(results)} воспоминаний")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#                     ЗАМЕТКИ ПОЛЬЗОВАТЕЛЯ
# ═══════════════════════════════════════════════════════════════

class SaveNoteTool(BaseTool):
    """Сохранение заметки"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="save_note",
            description="Сохраняет заметку с заголовком и содержимым. Заметки сохраняются на диск и доступны между сессиями.",
            required_args=["title", "content"],
            optional_args=["tags"],
            arg_types={"title": str, "content": str, "tags": str},
            arg_descriptions={
                "title": "Заголовок заметки (например: 'Список покупок', 'Идеи для проекта')",
                "content": "Содержимое заметки",
                "tags": "Теги через запятую (например: 'работа, важное, проект')",
            },
            category="memory",
            examples=[
                'save_note("Список покупок", "Молоко, хлеб, яйца")',
                'save_note("Идея проекта", "Сделать чат-бота для...", "работа, идеи")',
            ],
        )

    async def execute(self, title: str, content: str, tags: str = "") -> str:
        logger.info(f"Сохранение заметки: {title}")

        notes_dir = config.NOTES_DIR
        notes_dir.mkdir(parents=True, exist_ok=True)

        notes_file = notes_dir / "notes.json"

        # Загружаем существующие заметки
        notes = []
        if notes_file.exists():
            try:
                notes = json.loads(notes_file.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, Exception):
                notes = []

        # Создаём новую заметку
        note = {
            "id": len(notes) + 1,
            "title": title,
            "content": content,
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
            "created": datetime.now().strftime('%d.%m.%Y %H:%M'),
            "updated": datetime.now().strftime('%d.%m.%Y %H:%M'),
        }

        notes.append(note)

        # Сохраняем
        notes_file.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')

        logger.info(f"Заметка #{note['id']} сохранена")
        tags_str = f" (теги: {', '.join(note['tags'])})" if note['tags'] else ""
        return f"Заметка #{note['id']} '{title}' сохранена{tags_str}"


class ListNotesTool(BaseTool):
    """Просмотр заметок"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_notes",
            description="Показывает список всех сохранённых заметок или фильтрует по тегу/ключевому слову.",
            required_args=[],
            optional_args=["search", "tag"],
            arg_types={"search": str, "tag": str},
            arg_descriptions={
                "search": "Поиск по заголовку и содержимому",
                "tag": "Фильтр по тегу",
            },
            category="memory",
            examples=[
                'list_notes()',
                'list_notes(search="покупки")',
                'list_notes(tag="работа")',
            ],
        )

    async def execute(self, search: str = None, tag: str = None) -> str:
        logger.info(f"Просмотр заметок (search={search}, tag={tag})")

        notes_file = config.NOTES_DIR / "notes.json"

        if not notes_file.exists():
            return "Заметок пока нет. Используй save_note для создания."

        try:
            notes = json.loads(notes_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, Exception):
            return "Ошибка чтения заметок."

        if not notes:
            return "Заметок пока нет."

        # Фильтрация
        filtered = notes

        if search:
            search_lower = search.lower()
            filtered = [
                n for n in filtered
                if search_lower in n['title'].lower() or search_lower in n['content'].lower()
            ]

        if tag:
            tag_lower = tag.lower()
            filtered = [
                n for n in filtered
                if tag_lower in [t.lower() for t in n.get('tags', [])]
            ]

        if not filtered:
            return f"Заметок не найдено по запросу."

        lines = [f"Заметки ({len(filtered)}):", ""]

        for n in filtered:
            tags_str = f" [{', '.join(n.get('tags', []))}]" if n.get('tags') else ""
            preview = n['content'][:80] + "..." if len(n['content']) > 80 else n['content']
            lines.append(f"  #{n['id']}  {n['title']}{tags_str}")
            lines.append(f"       {preview}")
            lines.append(f"       ({n.get('created', 'н/д')})")
            lines.append("")

        return "\n".join(lines)


class DeleteNoteTool(BaseTool):
    """Удаление заметки"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_note",
            description="Удаляет заметку по её номеру (ID).",
            required_args=["note_id"],
            arg_types={"note_id": int},
            arg_descriptions={
                "note_id": "Номер заметки для удаления (можно узнать через list_notes)",
            },
            category="memory",
            danger_level="warning",
            examples=[
                'delete_note(1)',
                'delete_note(5)',
            ],
        )

    async def execute(self, note_id: int) -> str:
        logger.info(f"Удаление заметки #{note_id}")

        notes_file = config.NOTES_DIR / "notes.json"

        if not notes_file.exists():
            return "Заметок нет."

        try:
            notes = json.loads(notes_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, Exception):
            return "Ошибка чтения заметок."

        # Ищем заметку
        found = None
        for i, n in enumerate(notes):
            if n['id'] == note_id:
                found = i
                break

        if found is None:
            return f"Заметка #{note_id} не найдена."

        deleted = notes.pop(found)
        notes_file.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')

        logger.info(f"Заметка #{note_id} удалена")
        return f"Заметка #{note_id} '{deleted['title']}' удалена."


class ReadNoteTool(BaseTool):
    """Чтение конкретной заметки"""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_note",
            description="Показывает полное содержимое заметки по её номеру.",
            required_args=["note_id"],
            arg_types={"note_id": int},
            arg_descriptions={
                "note_id": "Номер заметки для чтения",
            },
            category="memory",
            examples=[
                'read_note(1)',
                'read_note(3)',
            ],
        )

    async def execute(self, note_id: int) -> str:
        logger.info(f"Чтение заметки #{note_id}")

        notes_file = config.NOTES_DIR / "notes.json"

        if not notes_file.exists():
            return "Заметок нет."

        try:
            notes = json.loads(notes_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, Exception):
            return "Ошибка чтения заметок."

        for n in notes:
            if n['id'] == note_id:
                tags_str = f"\nТеги: {', '.join(n.get('tags', []))}" if n.get('tags') else ""
                return (
                    f"Заметка #{n['id']}: {n['title']}\n"
                    f"Создана: {n.get('created', 'н/д')}\n"
                    f"{tags_str}\n"
                    f"{'=' * 40}\n"
                    f"{n['content']}"
                )

        return f"Заметка #{note_id} не найдена."
