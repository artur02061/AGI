"""
Кристина 7.3 — Task Planner (Планировщик задач)

ЗАЧЕМ:
  Claude разбивает сложные задачи на подзадачи автоматически.
  Кристина должна уметь то же самое.

  "Создай веб-приложение" →
    ├─ "Создай backend"
    │   ├─ "Настрой FastAPI"
    │   ├─ "Создай модели"
    │   └─ "Создай роуты"
    ├─ "Создай frontend"
    │   ├─ "Настрой React"
    │   └─ "Создай компоненты"
    └─ "Настрой деплой"

КАК РАБОТАЕТ:
  1. Входная задача → определяем тип и сложность
  2. Декомпозиция → дерево подзадач
  3. Приоритизация → порядок выполнения
     - Зависимости (backend → frontend)
     - Сложность (простое сначала)
     - Критичность (блокирующие задачи первыми)
  4. Выполнение → пошагово с отслеживанием прогресса

ОБУЧЕНИЕ:
  Когда LLM решает сложную задачу, сохраняем структуру разбиения.
  В следующий раз для похожей задачи — разбиваем по шаблону.
"""

import sqlite3
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from utils.logging import get_logger
import config

logger = get_logger("task_planner")


# ═══════════════════════════════════════════════════════════════
#               СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class TaskPriority(Enum):
    CRITICAL = 0    # Блокирует всё остальное
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class TaskNode:
    """Узел дерева задач"""
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    parent_id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)  # ID задач-зависимостей
    children: List[str] = field(default_factory=list)     # ID подзадач
    estimated_complexity: str = "medium"  # "trivial", "simple", "medium", "complex"
    result: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0

    def is_ready(self) -> bool:
        """Можно ли начать задачу (все зависимости выполнены)"""
        return self.status == TaskStatus.PENDING and not self.depends_on

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "depends_on": self.depends_on,
            "children": self.children,
            "estimated_complexity": self.estimated_complexity,
            "result": self.result,
        }


@dataclass
class TaskPlan:
    """Полный план выполнения задачи"""
    root_task: str              # Исходная задача
    nodes: Dict[str, TaskNode]  # id → TaskNode
    execution_order: List[str]  # Топологический порядок выполнения
    total_tasks: int = 0
    completed_tasks: int = 0
    created_at: float = 0.0

    @property
    def progress(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.completed_tasks / self.total_tasks * 100

    def to_dict(self) -> Dict:
        return {
            "root_task": self.root_task,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "execution_order": self.execution_order,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "progress": round(self.progress, 1),
        }


# ═══════════════════════════════════════════════════════════════
#               ШАБЛОНЫ ДЕКОМПОЗИЦИИ
# ═══════════════════════════════════════════════════════════════

# Шаблоны разбиения типичных задач на подзадачи
DECOMPOSITION_LIBRARY = {
    "create_app": {
        "triggers": ["создай приложение", "напиши программу", "разработай",
                      "create app", "build application"],
        "template": {
            "root": "Создать приложение",
            "children": [
                {
                    "title": "Проектирование",
                    "complexity": "medium",
                    "children": [
                        {"title": "Определить требования", "complexity": "simple"},
                        {"title": "Спроектировать архитектуру", "complexity": "medium"},
                        {"title": "Выбрать технологии", "complexity": "simple"},
                    ],
                },
                {
                    "title": "Backend",
                    "complexity": "complex",
                    "depends_on_idx": [0],
                    "children": [
                        {"title": "Настроить окружение", "complexity": "simple"},
                        {"title": "Создать модели данных", "complexity": "medium"},
                        {"title": "Реализовать API", "complexity": "medium"},
                        {"title": "Написать тесты", "complexity": "medium"},
                    ],
                },
                {
                    "title": "Frontend",
                    "complexity": "complex",
                    "depends_on_idx": [0],
                    "children": [
                        {"title": "Настроить UI-фреймворк", "complexity": "simple"},
                        {"title": "Создать компоненты", "complexity": "medium"},
                        {"title": "Подключить к API", "complexity": "medium"},
                    ],
                },
                {
                    "title": "Деплой",
                    "complexity": "medium",
                    "depends_on_idx": [1, 2],
                    "children": [
                        {"title": "Настроить CI/CD", "complexity": "medium"},
                        {"title": "Развернуть", "complexity": "simple"},
                    ],
                },
            ],
        },
    },
    "create_file": {
        "triggers": ["создай файл", "напиши файл", "сгенерируй файл",
                      "create file", "write file"],
        "template": {
            "root": "Создать файл",
            "children": [
                {"title": "Определить формат и содержание", "complexity": "simple"},
                {"title": "Создать файл", "complexity": "simple"},
                {"title": "Проверить результат", "complexity": "trivial"},
            ],
        },
    },
    "fix_bug": {
        "triggers": ["исправь", "почини", "баг", "ошибка", "не работает",
                      "fix", "bug", "error"],
        "template": {
            "root": "Исправить проблему",
            "children": [
                {"title": "Воспроизвести проблему", "complexity": "simple"},
                {"title": "Найти причину", "complexity": "medium"},
                {"title": "Разработать решение", "complexity": "medium"},
                {"title": "Применить исправление", "complexity": "simple"},
                {
                    "title": "Проверить",
                    "complexity": "simple",
                    "depends_on_idx": [3],
                },
            ],
        },
    },
    "analyze_data": {
        "triggers": ["проанализируй", "исследуй", "статистика", "отчёт",
                      "analyze", "report"],
        "template": {
            "root": "Анализ данных",
            "children": [
                {"title": "Собрать данные", "complexity": "medium"},
                {"title": "Очистить и подготовить", "complexity": "medium"},
                {
                    "title": "Провести анализ",
                    "complexity": "complex",
                    "depends_on_idx": [1],
                },
                {
                    "title": "Оформить результаты",
                    "complexity": "simple",
                    "depends_on_idx": [2],
                },
            ],
        },
    },
    "learn_topic": {
        "triggers": ["объясни", "расскажи", "научи", "что такое",
                      "explain", "teach"],
        "template": {
            "root": "Объяснить тему",
            "children": [
                {"title": "Определить уровень сложности", "complexity": "trivial"},
                {"title": "Подобрать аналогии", "complexity": "simple"},
                {"title": "Дать определение", "complexity": "simple"},
                {"title": "Привести примеры", "complexity": "simple"},
                {"title": "Проверить понимание", "complexity": "trivial"},
            ],
        },
    },
    "refactor_code": {
        "triggers": ["рефакторинг", "переписать", "улучши код", "оптимизируй",
                      "refactor", "optimize"],
        "template": {
            "root": "Рефакторинг кода",
            "children": [
                {"title": "Понять текущий код", "complexity": "medium"},
                {"title": "Определить проблемные места", "complexity": "medium"},
                {"title": "Спланировать изменения", "complexity": "simple"},
                {
                    "title": "Применить рефакторинг",
                    "complexity": "complex",
                    "depends_on_idx": [2],
                },
                {
                    "title": "Проверить что ничего не сломалось",
                    "complexity": "medium",
                    "depends_on_idx": [3],
                },
            ],
        },
    },
    "setup_project": {
        "triggers": ["настрой проект", "инициализируй", "создай проект",
                      "setup", "init project"],
        "template": {
            "root": "Настроить проект",
            "children": [
                {"title": "Создать структуру директорий", "complexity": "simple"},
                {"title": "Настроить зависимости", "complexity": "simple"},
                {"title": "Настроить конфигурацию", "complexity": "medium"},
                {"title": "Создать базовые файлы", "complexity": "simple"},
                {
                    "title": "Проверить что проект собирается",
                    "complexity": "simple",
                    "depends_on_idx": [1, 2, 3],
                },
            ],
        },
    },
}

# Слова-маркеры сложности
COMPLEXITY_MARKERS = {
    "trivial": ["простой", "быстро", "легко", "маленький", "один файл"],
    "simple": ["несложный", "базовый", "стандартный", "обычный"],
    "medium": ["средний", "типичный", "нормальный"],
    "complex": ["сложный", "большой", "многокомпонентный", "архитектура",
                "масштабный", "полноценный", "production"],
}


# ═══════════════════════════════════════════════════════════════
#               TASK PLANNER
# ═══════════════════════════════════════════════════════════════


class TaskPlanner:
    """
    Планировщик задач — декомпозиция и управление выполнением.

    Использование:
        planner = TaskPlanner(knowledge_distillation, sentence_embeddings)

        # Создать план
        plan = planner.plan("Создай веб-приложение на FastAPI + React")

        # Следующая задача к выполнению
        next_task = planner.next_task(plan)

        # Отметить задачу выполненной
        planner.complete_task(plan, next_task.id, result="Done")

        # Прогресс
        print(f"Progress: {plan.progress}%")
    """

    def __init__(
        self,
        knowledge_distillation=None,
        sentence_embeddings=None,
        db_path: Path = None,
    ):
        self._kd = knowledge_distillation
        self._sentence = sentence_embeddings

        self._db_path = db_path or (config.config.data_dir / "task_planner.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        self._total_plans = 0
        self._total_tasks_completed = 0
        self._load_stats()

        logger.info(
            f"📋 TaskPlanner: {self._total_plans} планов, "
            f"{self._total_tasks_completed} задач выполнено"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_task TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                total_tasks INTEGER NOT NULL,
                completed_tasks INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                created_at REAL NOT NULL,
                completed_at REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_decompositions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_pattern TEXT NOT NULL,
                decomposition_json TEXT NOT NULL,
                usage_count INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS planner_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _load_stats(self):
        for key, attr in [("total_plans", "_total_plans"),
                          ("total_tasks_completed", "_total_tasks_completed")]:
            row = self._conn.execute(
                "SELECT value FROM planner_stats WHERE key = ?", (key,)
            ).fetchone()
            if row:
                setattr(self, attr, int(row["value"]))

    def _save_stats(self):
        for key, val in [
            ("total_plans", str(self._total_plans)),
            ("total_tasks_completed", str(self._total_tasks_completed)),
        ]:
            self._conn.execute("""
                INSERT INTO planner_stats (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, val, val))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           СОЗДАНИЕ ПЛАНА
    # ═══════════════════════════════════════════════════════════════

    def plan(self, task_description: str) -> TaskPlan:
        """
        Создаёт план выполнения задачи.

        1. Ищет подходящий шаблон декомпозиции
        2. Адаптирует под конкретную задачу
        3. Определяет зависимости и порядок
        4. Возвращает готовый план

        Returns:
            TaskPlan с деревом задач и порядком выполнения
        """
        now = time.time()
        self._total_plans += 1

        # 1. Определяем тип задачи и ищем шаблон
        template = self._find_template(task_description)

        # 2. Строим дерево задач
        if template:
            plan = self._build_from_template(task_description, template, now)
        else:
            # Нет шаблона — создаём простой линейный план
            plan = self._build_simple_plan(task_description, now)

        # 3. Определяем порядок выполнения (топологическая сортировка)
        plan.execution_order = self._topological_sort(plan)

        # 4. Сохраняем в БД
        self._conn.execute("""
            INSERT INTO plans (root_task, plan_json, total_tasks, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            task_description,
            json.dumps(plan.to_dict(), ensure_ascii=False),
            plan.total_tasks,
            now,
        ))
        self._save_stats()

        logger.info(
            f"📋 Plan: '{task_description[:50]}...' → "
            f"{plan.total_tasks} задач, "
            f"order={len(plan.execution_order)}"
        )

        return plan

    def _find_template(self, task_description: str) -> Optional[Dict]:
        """Ищет подходящий шаблон декомпозиции"""
        text = task_description.lower()

        # 1. Из встроенной библиотеки
        best_template = None
        best_score = 0

        for key, entry in DECOMPOSITION_LIBRARY.items():
            score = sum(1 for trigger in entry["triggers"] if trigger in text)
            if score > best_score:
                best_score = score
                best_template = entry["template"]

        if best_template and best_score > 0:
            return best_template

        # 2. Из выученных декомпозиций
        learned = self._find_learned_decomposition(task_description)
        if learned:
            return learned

        # 3. Из KnowledgeDistillation (цепочки рассуждений)
        if self._kd:
            reasoning = self._kd.find_reasoning(task_description)
            if reasoning and reasoning["confidence"] >= 0.6:
                return self._reasoning_to_template(reasoning)

        return None

    def _find_learned_decomposition(self, task_description: str) -> Optional[Dict]:
        """Ищет выученную декомпозицию по похожести"""
        if not self._sentence:
            return None

        rows = self._conn.execute("""
            SELECT task_pattern, decomposition_json, success_rate
            FROM learned_decompositions
            WHERE success_rate >= 0.5
            ORDER BY usage_count DESC
            LIMIT 20
        """).fetchall()

        best = None
        best_sim = 0.0

        for row in rows:
            sim = self._sentence.similarity(task_description, row["task_pattern"])
            if sim > best_sim:
                best_sim = sim
                best = row

        if best and best_sim >= 0.5:
            try:
                return json.loads(best["decomposition_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _reasoning_to_template(self, reasoning: Dict) -> Optional[Dict]:
        """Конвертирует цепочку рассуждений в шаблон декомпозиции"""
        steps = reasoning.get("steps", [])
        if len(steps) < 2:
            return None

        children = []
        for step in steps:
            children.append({
                "title": step.get("text", "Шаг"),
                "complexity": "medium",
            })

        return {
            "root": "Выполнить задачу",
            "children": children,
        }

    def _build_from_template(
        self,
        task_description: str,
        template: Dict,
        now: float,
    ) -> TaskPlan:
        """Строит дерево задач из шаблона"""
        nodes: Dict[str, TaskNode] = {}
        counter = [0]

        def _gen_id() -> str:
            counter[0] += 1
            return f"task_{counter[0]}"

        def _build_node(
            data: Dict,
            parent_id: Optional[str] = None,
            sibling_ids: List[str] = None,
        ) -> str:
            node_id = _gen_id()
            node = TaskNode(
                id=node_id,
                title=data.get("title", "Задача"),
                description=data.get("description", ""),
                priority=TaskPriority.MEDIUM,
                parent_id=parent_id,
                estimated_complexity=data.get("complexity", "medium"),
                created_at=now,
            )

            # Зависимости от sibling по индексу
            if sibling_ids and "depends_on_idx" in data:
                for idx in data["depends_on_idx"]:
                    if idx < len(sibling_ids):
                        node.depends_on.append(sibling_ids[idx])

            nodes[node_id] = node

            # Рекурсивно строим детей
            children_data = data.get("children", [])
            child_ids = []
            for child_data in children_data:
                child_id = _build_node(child_data, parent_id=node_id, sibling_ids=child_ids)
                child_ids.append(child_id)
                node.children.append(child_id)

            return node_id

        # Строим от корня
        root_children = template.get("children", [])
        root_id = _gen_id()
        root_node = TaskNode(
            id=root_id,
            title=template.get("root", task_description),
            description=task_description,
            priority=TaskPriority.HIGH,
            estimated_complexity="complex",
            created_at=now,
        )
        nodes[root_id] = root_node

        child_ids = []
        for child_data in root_children:
            child_id = _build_node(child_data, parent_id=root_id, sibling_ids=child_ids)
            child_ids.append(child_id)
            root_node.children.append(child_id)

        return TaskPlan(
            root_task=task_description,
            nodes=nodes,
            execution_order=[],
            total_tasks=len(nodes),
            created_at=now,
        )

    def _build_simple_plan(self, task_description: str, now: float) -> TaskPlan:
        """Строит простой линейный план (для неизвестных задач)"""
        complexity = self._estimate_complexity(task_description)
        nodes: Dict[str, TaskNode] = {}

        # Корень
        root = TaskNode(
            id="task_1",
            title=task_description,
            description=task_description,
            priority=TaskPriority.HIGH,
            estimated_complexity=complexity,
            created_at=now,
        )
        nodes["task_1"] = root

        if complexity in ("medium", "complex"):
            # Добавляем стандартные шаги
            steps = [
                ("task_2", "Понять задачу", "simple"),
                ("task_3", "Выполнить", "medium"),
                ("task_4", "Проверить результат", "simple"),
            ]

            prev_id = None
            for task_id, title, comp in steps:
                node = TaskNode(
                    id=task_id,
                    title=title,
                    parent_id="task_1",
                    estimated_complexity=comp,
                    created_at=now,
                )
                if prev_id:
                    node.depends_on.append(prev_id)
                nodes[task_id] = node
                root.children.append(task_id)
                prev_id = task_id

        return TaskPlan(
            root_task=task_description,
            nodes=nodes,
            execution_order=[],
            total_tasks=len(nodes),
            created_at=now,
        )

    def _estimate_complexity(self, task: str) -> str:
        """Оценивает сложность задачи по ключевым словам"""
        text = task.lower()
        for complexity, markers in COMPLEXITY_MARKERS.items():
            for marker in markers:
                if marker in text:
                    return complexity
        # По длине описания
        if len(task) > 100:
            return "complex"
        if len(task) > 40:
            return "medium"
        return "simple"

    # ═══════════════════════════════════════════════════════════════
    #           ТОПОЛОГИЧЕСКАЯ СОРТИРОВКА
    # ═══════════════════════════════════════════════════════════════

    def _topological_sort(self, plan: TaskPlan) -> List[str]:
        """
        Определяет порядок выполнения задач с учётом зависимостей.
        Задачи без зависимостей идут первыми.
        Листовые задачи (без детей) — это реально выполняемые.
        """
        # Берём только листовые задачи (без children)
        leaves = [
            nid for nid, node in plan.nodes.items()
            if not node.children
        ]

        # Topological sort (Kahn's algorithm)
        in_degree: Dict[str, int] = {}
        for nid in leaves:
            deps = plan.nodes[nid].depends_on
            in_degree[nid] = len([d for d in deps if d in set(leaves)])

        queue = [nid for nid in leaves if in_degree.get(nid, 0) == 0]
        # Сортируем очередь по приоритету
        queue.sort(key=lambda nid: plan.nodes[nid].priority.value)

        order = []
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            order.append(current)

            # Разблокируем задачи, зависящие от current
            for nid in leaves:
                if nid in visited:
                    continue
                if current in plan.nodes[nid].depends_on:
                    in_degree[nid] = in_degree.get(nid, 1) - 1
                    if in_degree[nid] <= 0:
                        queue.append(nid)

            queue.sort(key=lambda nid: plan.nodes[nid].priority.value)

        # Добавляем оставшиеся (если есть циклические зависимости)
        for nid in leaves:
            if nid not in visited:
                order.append(nid)

        return order

    # ═══════════════════════════════════════════════════════════════
    #           ВЫПОЛНЕНИЕ ПЛАНА
    # ═══════════════════════════════════════════════════════════════

    def next_task(self, plan: TaskPlan) -> Optional[TaskNode]:
        """
        Возвращает следующую задачу для выполнения.
        Учитывает зависимости: задача доступна только если
        все её зависимости выполнены.
        """
        completed = {
            nid for nid, node in plan.nodes.items()
            if node.status == TaskStatus.COMPLETED
        }

        for task_id in plan.execution_order:
            node = plan.nodes.get(task_id)
            if not node or node.status != TaskStatus.PENDING:
                continue

            # Проверяем зависимости
            deps_met = all(
                dep in completed
                for dep in node.depends_on
            )

            if deps_met:
                return node

        return None

    def complete_task(
        self,
        plan: TaskPlan,
        task_id: str,
        result: str = "",
        success: bool = True,
    ):
        """Отмечает задачу как выполненную"""
        node = plan.nodes.get(task_id)
        if not node:
            return

        node.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        node.result = result
        node.completed_at = time.time()
        plan.completed_tasks += 1
        self._total_tasks_completed += 1

        # Если все дети выполнены — отмечаем родителя
        if node.parent_id:
            parent = plan.nodes.get(node.parent_id)
            if parent:
                children_done = all(
                    plan.nodes[cid].status == TaskStatus.COMPLETED
                    for cid in parent.children
                    if cid in plan.nodes
                )
                if children_done:
                    parent.status = TaskStatus.COMPLETED
                    parent.completed_at = time.time()

        logger.debug(
            f"📋 Task completed: '{node.title}' "
            f"({plan.completed_tasks}/{plan.total_tasks})"
        )

    def get_plan_status(self, plan: TaskPlan) -> Dict:
        """Текущий статус плана"""
        by_status = {}
        for node in plan.nodes.values():
            s = node.status.value
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "root_task": plan.root_task[:80],
            "total": plan.total_tasks,
            "completed": plan.completed_tasks,
            "progress": round(plan.progress, 1),
            "by_status": by_status,
            "next_task": self.next_task(plan).title if self.next_task(plan) else None,
        }

    def format_plan(self, plan: TaskPlan) -> str:
        """Форматирует план как дерево для отображения"""
        lines = [f"📋 План: {plan.root_task[:60]}"]
        lines.append(f"   Прогресс: {plan.progress:.0f}% ({plan.completed_tasks}/{plan.total_tasks})")
        lines.append("")

        def _format_node(node_id: str, indent: int = 0):
            node = plan.nodes.get(node_id)
            if not node:
                return

            status_icons = {
                TaskStatus.PENDING: "○",
                TaskStatus.IN_PROGRESS: "◐",
                TaskStatus.COMPLETED: "●",
                TaskStatus.BLOCKED: "◌",
                TaskStatus.FAILED: "✕",
            }
            icon = status_icons.get(node.status, "?")
            prefix = "  " * indent

            deps = ""
            if node.depends_on:
                deps = f" [ждёт: {', '.join(node.depends_on)}]"

            lines.append(f"{prefix}{icon} {node.title}{deps}")

            for child_id in node.children:
                _format_node(child_id, indent + 1)

        # Начинаем с корня (task_1)
        root_ids = [
            nid for nid, node in plan.nodes.items()
            if node.parent_id is None
        ]
        for root_id in root_ids:
            _format_node(root_id)

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    #           ОБУЧЕНИЕ ИЗ LLM
    # ═══════════════════════════════════════════════════════════════

    def learn_decomposition(
        self,
        task_description: str,
        decomposition: Dict,
        success: bool = True,
    ):
        """
        Запоминает декомпозицию для будущего использования.
        Вызывается когда LLM успешно решил сложную задачу.
        """
        now = time.time()

        # Ищем похожую
        existing = None
        if self._sentence:
            rows = self._conn.execute(
                "SELECT id, task_pattern FROM learned_decompositions"
            ).fetchall()
            for row in rows:
                sim = self._sentence.similarity(task_description, row["task_pattern"])
                if sim >= 0.8:
                    existing = row["id"]
                    break

        if existing:
            # Обновляем
            sr_delta = 0.1 if success else -0.2
            self._conn.execute("""
                UPDATE learned_decompositions
                SET usage_count = usage_count + 1,
                    success_rate = MAX(0, MIN(1, success_rate + ?)),
                    updated_at = ?
                WHERE id = ?
            """, (sr_delta, now, existing))
        else:
            # Новая
            self._conn.execute("""
                INSERT INTO learned_decompositions
                (task_pattern, decomposition_json, success_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                task_description,
                json.dumps(decomposition, ensure_ascii=False),
                1.0 if success else 0.5,
                now, now,
            ))

        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        learned = self._conn.execute(
            "SELECT COUNT(*) as c FROM learned_decompositions"
        ).fetchone()["c"]

        active_plans = self._conn.execute(
            "SELECT COUNT(*) as c FROM plans WHERE status = 'active'"
        ).fetchone()["c"]

        return {
            "total_plans": self._total_plans,
            "total_tasks_completed": self._total_tasks_completed,
            "learned_decompositions": learned,
            "active_plans": active_plans,
        }

    def close(self):
        self._save_stats()
        self._conn.close()
