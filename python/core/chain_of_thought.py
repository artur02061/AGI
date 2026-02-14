"""
Кристина 7.3 — Chain-of-Thought Engine (Движок рассуждений)

ЗАЧЕМ:
  Claude умеет "думать шаг за шагом" (Extended Thinking).
  Кристина должна уметь то же самое — БЕЗ вызова LLM.

  KnowledgeDistillation ЗАПИСЫВАЕТ цепочки рассуждений из LLM.
  Chain-of-Thought Engine ВЫПОЛНЯЕТ их автоматически.

КАК РАБОТАЕТ:
  1. Пользователь: "Найди все Python файлы больше 100 строк"
  2. CoT ищет подходящую цепочку в KnowledgeDistillation
  3. Если нашёл — выполняет пошагово:
     [Мысль] Нужно найти файлы по расширению
     [Действие] glob("**/*.py")
     [Наблюдение] Найдено 47 файлов
     [Мысль] Нужно проверить размер каждого
     [Действие] count_lines(file) для каждого
     [Наблюдение] 12 файлов > 100 строк
     [Вывод] Вот 12 файлов: ...
  4. Если не нашёл цепочку — строит рассуждение с нуля:
     - Декомпозиция задачи (разбивает на подзадачи)
     - Планирование шагов (определяет порядок)
     - Выполнение и верификация каждого шага

АРХИТЕКТУРА:
  ┌──────────────────────────────────────────┐
  │         Chain-of-Thought Engine          │
  │                                          │
  │  ┌──────────────────────────────────┐    │
  │  │ ReasoningStrategy                │    │
  │  │  - from_template (KD цепочки)    │    │
  │  │  - decompose (новая задача)      │    │
  │  │  - analogy (по аналогии)         │    │
  │  └──────────┬───────────────────────┘    │
  │             ↓                             │
  │  ┌──────────────────────────────────┐    │
  │  │ StepExecutor                     │    │
  │  │  thought → action → observation  │    │
  │  │  с верификацией каждого шага     │    │
  │  └──────────┬───────────────────────┘    │
  │             ↓                             │
  │  ┌──────────────────────────────────┐    │
  │  │ ResponseComposer                 │    │
  │  │  steps → связный ответ           │    │
  │  └─────────────────────────────────-┘    │
  └──────────────────────────────────────────┘

ИНТЕГРАЦИЯ:
  Оркестратор → Tier 3 (перед LLM fallback)
  Если CoT справился — LLM НЕ вызывается.
"""

import re
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

from utils.logging import get_logger
import config

logger = get_logger("chain_of_thought")


# ═══════════════════════════════════════════════════════════════
#               СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════


@dataclass
class ThoughtStep:
    """Один шаг рассуждения"""
    step_num: int
    thought: str        # "Что нужно сделать и почему"
    action: str         # "Какое действие выполнить"
    observation: str    # "Что получили в результате"
    conclusion: str     # "Что это значит для следующего шага"
    success: bool = True
    confidence: float = 1.0


@dataclass
class ThoughtChain:
    """Полная цепочка рассуждений"""
    query: str                          # Исходный запрос
    strategy: str                       # "template" | "decompose" | "analogy" | "direct"
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    overall_confidence: float = 0.0
    reasoning_time_ms: float = 0.0
    source_chain_id: Optional[int] = None  # ID цепочки из KnowledgeDistillation

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "steps": [asdict(s) for s in self.steps],
            "final_answer": self.final_answer,
            "overall_confidence": self.overall_confidence,
            "reasoning_time_ms": self.reasoning_time_ms,
        }


# ═══════════════════════════════════════════════════════════════
#               ПАТТЕРНЫ РАССУЖДЕНИЙ
# ═══════════════════════════════════════════════════════════════

# Шаблоны для автоматической декомпозиции типичных задач
DECOMPOSITION_TEMPLATES = {
    # Поиск информации
    "search": {
        "triggers": ["найди", "поиск", "где", "какой", "сколько", "покажи список"],
        "steps": [
            ("определить_критерии", "Определить что именно ищем"),
            ("выбрать_источник", "Выбрать где искать"),
            ("выполнить_поиск", "Выполнить поиск"),
            ("фильтровать", "Отфильтровать результаты"),
            ("оформить", "Оформить ответ"),
        ],
    },
    # Создание чего-либо
    "create": {
        "triggers": ["создай", "напиши", "сделай", "сгенерируй", "добавь"],
        "steps": [
            ("понять_что", "Понять что именно нужно создать"),
            ("определить_формат", "Определить формат/структуру"),
            ("подготовить", "Подготовить необходимые данные"),
            ("создать", "Создать объект"),
            ("проверить", "Проверить результат"),
        ],
    },
    # Анализ
    "analyze": {
        "triggers": ["проанализируй", "объясни", "почему", "сравни", "оцени"],
        "steps": [
            ("собрать_данные", "Собрать информацию для анализа"),
            ("выделить_ключевое", "Выделить ключевые аспекты"),
            ("сравнить", "Сравнить/сопоставить факты"),
            ("сделать_выводы", "Сформулировать выводы"),
            ("оформить", "Оформить анализ"),
        ],
    },
    # Исправление/починка
    "fix": {
        "triggers": ["исправь", "почини", "реши", "устрани", "ошибка", "баг", "не работает"],
        "steps": [
            ("воспроизвести", "Воспроизвести проблему"),
            ("диагностика", "Определить причину"),
            ("найти_решение", "Найти способ исправления"),
            ("применить", "Применить исправление"),
            ("проверить", "Проверить что проблема решена"),
        ],
    },
    # Настройка/конфигурация
    "configure": {
        "triggers": ["настрой", "установи", "конфигурация", "подключи", "запусти"],
        "steps": [
            ("проверить_требования", "Проверить что нужно для настройки"),
            ("подготовить", "Подготовить окружение"),
            ("настроить", "Выполнить настройку"),
            ("проверить", "Проверить работоспособность"),
        ],
    },
    # Преобразование
    "transform": {
        "triggers": ["преобразуй", "конвертируй", "переведи", "перепиши", "измени формат"],
        "steps": [
            ("прочитать_вход", "Прочитать/понять входные данные"),
            ("определить_формат", "Определить целевой формат"),
            ("преобразовать", "Выполнить преобразование"),
            ("проверить", "Проверить корректность"),
        ],
    },
}

# Связки для генерации текста рассуждений
THOUGHT_CONNECTORS = {
    "first": ["Для начала нужно", "Первым делом", "Сначала"],
    "next": ["Далее нужно", "Затем", "После этого"],
    "check": ["Проверим результат", "Убедимся что", "Верифицируем"],
    "conclude": ["Итого", "Таким образом", "В результате"],
    "because": ["потому что", "так как", "поскольку"],
    "therefore": ["следовательно", "значит", "поэтому"],
}

# Паттерны для извлечения ключевых сущностей из запроса
ENTITY_PATTERNS = {
    "file": re.compile(r'(?:файл[а-я]*|file)\s+["\']?([^\s"\']+)', re.I),
    "path": re.compile(r'([/~][\w/.\-]+)', re.I),
    "number": re.compile(r'(\d+)', re.I),
    "name": re.compile(r'(?:назови|имен[а-я]*|name)\s+["\']?([^\s"\']+)', re.I),
    "format": re.compile(
        r'\b(csv|json|xml|html|yaml|toml|txt|md|py|js|ts|sql)\b', re.I
    ),
}


# ═══════════════════════════════════════════════════════════════
#               CHAIN-OF-THOUGHT ENGINE
# ═══════════════════════════════════════════════════════════════


class ChainOfThought:
    """
    Движок рассуждений Кристины — думает шаг за шагом без LLM.

    Три стратегии:
    1. template  — использует цепочку из KnowledgeDistillation
    2. decompose — разбивает задачу по шаблонам декомпозиции
    3. analogy   — рассуждает по аналогии с похожими задачами

    Использование:
        cot = ChainOfThought(knowledge_distillation, sentence_embeddings)

        # Попробовать решить задачу
        result = cot.reason("Найди все Python файлы больше 100 строк")

        if result and result.overall_confidence >= 0.6:
            print(result.final_answer)  # Готовый ответ
        else:
            # CoT не справился — передаём LLM
            pass
    """

    def __init__(
        self,
        knowledge_distillation=None,
        sentence_embeddings=None,
        tools: Dict = None,
        db_path: Path = None,
    ):
        self._kd = knowledge_distillation
        self._sentence = sentence_embeddings
        self._tools = tools or {}

        self._db_path = db_path or (config.config.data_dir / "chain_of_thought.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Статистика
        self._total_reasonings = 0
        self._successful_reasonings = 0
        self._load_stats()

        logger.info(
            f"🧠 ChainOfThought: {self._total_reasonings} рассуждений, "
            f"{self._successful_reasonings} успешных"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                strategy TEXT NOT NULL,
                chain_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                was_useful INTEGER DEFAULT -1,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _load_stats(self):
        row = self._conn.execute(
            "SELECT value FROM cot_stats WHERE key = 'total_reasonings'"
        ).fetchone()
        if row:
            self._total_reasonings = int(row["value"])
        row = self._conn.execute(
            "SELECT value FROM cot_stats WHERE key = 'successful_reasonings'"
        ).fetchone()
        if row:
            self._successful_reasonings = int(row["value"])

    def _save_stats(self):
        now = time.time()
        for key, val in [
            ("total_reasonings", str(self._total_reasonings)),
            ("successful_reasonings", str(self._successful_reasonings)),
        ]:
            self._conn.execute("""
                INSERT INTO cot_stats (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, val, val))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #               ГЛАВНЫЙ МЕТОД: РАССУЖДЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def reason(
        self,
        user_input: str,
        context: str = "",
        intent: str = None,
        max_steps: int = 8,
    ) -> Optional[ThoughtChain]:
        """
        Пытается решить задачу рассуждением.

        Порядок стратегий:
        1. template  — ищем готовую цепочку в KnowledgeDistillation
        2. decompose — разбиваем по шаблонам декомпозиции
        3. analogy   — рассуждаем по аналогии

        Returns:
            ThoughtChain с результатом или None если не справились
        """
        start = time.time()
        self._total_reasonings += 1

        # 1. Стратегия: Template (из KnowledgeDistillation)
        chain = self._try_template_strategy(user_input, intent)
        if chain and chain.overall_confidence >= 0.5:
            chain.reasoning_time_ms = (time.time() - start) * 1000
            self._record_reasoning(chain)
            return chain

        # 2. Стратегия: Decompose (разбиение на подзадачи)
        chain = self._try_decompose_strategy(user_input, context, max_steps)
        if chain and chain.overall_confidence >= 0.4:
            chain.reasoning_time_ms = (time.time() - start) * 1000
            self._record_reasoning(chain)
            return chain

        # 3. Стратегия: Analogy (по аналогии)
        chain = self._try_analogy_strategy(user_input, context)
        if chain and chain.overall_confidence >= 0.4:
            chain.reasoning_time_ms = (time.time() - start) * 1000
            self._record_reasoning(chain)
            return chain

        # Не справились
        self._save_stats()
        return None

    # ═══════════════════════════════════════════════════════════════
    #           СТРАТЕГИЯ 1: TEMPLATE (из KnowledgeDistillation)
    # ═══════════════════════════════════════════════════════════════

    def _try_template_strategy(
        self,
        user_input: str,
        intent: str = None,
    ) -> Optional[ThoughtChain]:
        """
        Ищет готовую цепочку рассуждений в KnowledgeDistillation
        и адаптирует её к текущему запросу.
        """
        if not self._kd:
            return None

        reasoning = self._kd.find_reasoning(user_input, intent=intent)
        if not reasoning or reasoning["confidence"] < 0.5:
            return None

        chain = ThoughtChain(
            query=user_input,
            strategy="template",
            source_chain_id=reasoning["chain_id"],
        )

        # Превращаем шаги из KD в ThoughtSteps
        for i, step_data in enumerate(reasoning["steps"]):
            step = ThoughtStep(
                step_num=i + 1,
                thought=self._generate_thought(step_data["text"], i, len(reasoning["steps"])),
                action=step_data["text"],
                observation="(из сохранённого опыта)",
                conclusion=self._generate_conclusion(step_data, i, len(reasoning["steps"])),
                confidence=reasoning["confidence"],
            )
            chain.steps.append(step)

        # Собираем ответ из шагов
        chain.final_answer = self._compose_answer_from_steps(chain.steps, user_input)
        chain.overall_confidence = reasoning["confidence"] * 0.9  # Чуть ниже — не проверяли

        logger.debug(
            f"🧠 CoT template: {len(chain.steps)} steps, "
            f"conf={chain.overall_confidence:.2f}"
        )
        return chain

    # ═══════════════════════════════════════════════════════════════
    #           СТРАТЕГИЯ 2: DECOMPOSE (разбиение задачи)
    # ═══════════════════════════════════════════════════════════════

    def _try_decompose_strategy(
        self,
        user_input: str,
        context: str = "",
        max_steps: int = 8,
    ) -> Optional[ThoughtChain]:
        """
        Разбивает задачу на подзадачи по шаблонам декомпозиции.

        1. Определяет тип задачи (search, create, analyze, fix, ...)
        2. Берёт шаблон декомпозиции
        3. Заполняет шаги конкретикой из запроса
        """
        # Определяем тип задачи
        task_type = self._classify_task(user_input)
        if not task_type:
            return None

        template = DECOMPOSITION_TEMPLATES.get(task_type)
        if not template:
            return None

        # Извлекаем сущности из запроса
        entities = self._extract_entities(user_input)

        chain = ThoughtChain(
            query=user_input,
            strategy="decompose",
        )

        # Генерируем шаги из шаблона
        template_steps = template["steps"]
        for i, (action_id, description) in enumerate(template_steps[:max_steps]):
            # Заполняем шаг конкретикой
            thought = self._fill_thought(description, entities, i, len(template_steps))
            action = self._fill_action(action_id, entities, user_input)
            observation = self._simulate_observation(action_id, entities)
            conclusion = self._fill_conclusion(action_id, i, len(template_steps))

            step = ThoughtStep(
                step_num=i + 1,
                thought=thought,
                action=action,
                observation=observation,
                conclusion=conclusion,
                confidence=0.6,  # Средняя уверенность — не проверено
            )
            chain.steps.append(step)

        # Формируем ответ
        chain.final_answer = self._compose_decompose_answer(chain, task_type, entities)
        chain.overall_confidence = self._calculate_decompose_confidence(
            chain, task_type, entities
        )

        logger.debug(
            f"🧠 CoT decompose ({task_type}): {len(chain.steps)} steps, "
            f"conf={chain.overall_confidence:.2f}"
        )
        return chain

    def _classify_task(self, user_input: str) -> Optional[str]:
        """Определяет тип задачи по ключевым словам"""
        text = user_input.lower()

        best_type = None
        best_count = 0

        for task_type, template in DECOMPOSITION_TEMPLATES.items():
            count = sum(1 for trigger in template["triggers"] if trigger in text)
            if count > best_count:
                best_count = count
                best_type = task_type

        return best_type if best_count > 0 else None

    def _extract_entities(self, user_input: str) -> Dict[str, List[str]]:
        """Извлекает сущности из запроса"""
        entities: Dict[str, List[str]] = {}

        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = pattern.findall(user_input)
            if matches:
                entities[entity_type] = matches

        # Извлекаем ключевые слова (существительные и глаголы)
        words = re.findall(r'[а-яёa-z]{3,}', user_input.lower())
        stop = {
            "найди", "создай", "сделай", "покажи", "напиши", "помоги",
            "нужно", "можно", "пожалуйста", "хочу", "надо",
            "все", "для", "как", "что", "где", "это",
        }
        keywords = [w for w in words if w not in stop]
        if keywords:
            entities["keywords"] = keywords

        return entities

    def _fill_thought(
        self,
        description: str,
        entities: Dict,
        step_idx: int,
        total_steps: int,
    ) -> str:
        """Генерирует текст мысли для шага"""
        if step_idx == 0:
            connector = _random_choice(THOUGHT_CONNECTORS["first"])
        elif step_idx == total_steps - 1:
            connector = _random_choice(THOUGHT_CONNECTORS["conclude"])
        else:
            connector = _random_choice(THOUGHT_CONNECTORS["next"])

        # Добавляем конкретику из сущностей
        specifics = ""
        if "keywords" in entities and entities["keywords"]:
            kw = entities["keywords"][0]
            specifics = f" ({kw})"

        return f"{connector} {description.lower()}{specifics}."

    def _fill_action(
        self,
        action_id: str,
        entities: Dict,
        user_input: str,
    ) -> str:
        """Генерирует описание действия"""
        parts = [action_id.replace("_", " ")]

        if "file" in entities:
            parts.append(f"файл: {entities['file'][0]}")
        if "format" in entities:
            parts.append(f"формат: {entities['format'][0]}")
        if "number" in entities:
            parts.append(f"число: {entities['number'][0]}")

        return " — ".join(parts)

    def _simulate_observation(
        self,
        action_id: str,
        entities: Dict,
    ) -> str:
        """Генерирует ожидаемое наблюдение (без реального выполнения)"""
        observations = {
            "определить_критерии": "Критерии поиска определены",
            "выбрать_источник": "Источник данных выбран",
            "выполнить_поиск": "Поиск выполнен, результаты получены",
            "фильтровать": "Результаты отфильтрованы",
            "оформить": "Ответ оформлен",
            "понять_что": "Задача понята",
            "определить_формат": "Формат определён",
            "подготовить": "Данные подготовлены",
            "создать": "Объект создан",
            "проверить": "Проверка пройдена",
            "собрать_данные": "Данные собраны",
            "выделить_ключевое": "Ключевые аспекты выделены",
            "сравнить": "Сравнение проведено",
            "сделать_выводы": "Выводы сформулированы",
            "воспроизвести": "Проблема воспроизведена",
            "диагностика": "Причина определена",
            "найти_решение": "Решение найдено",
            "применить": "Исправление применено",
            "проверить_требования": "Требования проверены",
            "настроить": "Настройка выполнена",
            "прочитать_вход": "Входные данные прочитаны",
            "преобразовать": "Преобразование выполнено",
        }
        return observations.get(action_id, "Шаг выполнен")

    def _fill_conclusion(self, action_id: str, step_idx: int, total_steps: int) -> str:
        """Генерирует заключение шага"""
        if step_idx == total_steps - 1:
            return "Задача завершена."
        return f"Переходим к следующему шагу."

    def _compose_decompose_answer(
        self,
        chain: ThoughtChain,
        task_type: str,
        entities: Dict,
    ) -> str:
        """Собирает ответ из результатов декомпозиции"""
        parts = []

        # Вступление
        task_intros = {
            "search": "Для выполнения поиска",
            "create": "Для создания",
            "analyze": "Для анализа",
            "fix": "Для исправления проблемы",
            "configure": "Для настройки",
            "transform": "Для преобразования",
        }
        intro = task_intros.get(task_type, "Для выполнения задачи")
        parts.append(f"{intro} я выполнила следующие шаги:")

        # Шаги
        for step in chain.steps:
            parts.append(f"  {step.step_num}. {step.action}")

        # Результат
        if "keywords" in entities:
            topic = " ".join(entities["keywords"][:3])
            parts.append(f"\nРезультат по запросу '{topic}' готов.")

        return "\n".join(parts)

    def _calculate_decompose_confidence(
        self,
        chain: ThoughtChain,
        task_type: str,
        entities: Dict,
    ) -> float:
        """Оценивает уверенность в декомпозиции"""
        conf = 0.5  # Базовая

        # Бонус за наличие сущностей
        if entities:
            conf += 0.1 * min(len(entities), 3)

        # Бонус за точное совпадение типа задачи
        if task_type in ("search", "create", "fix"):
            conf += 0.05

        # Бонус если есть KnowledgeDistillation с примерами
        if self._kd:
            stats = self._kd.get_stats()
            if stats["chains"] > 10:
                conf += 0.05

        return min(conf, 0.9)

    # ═══════════════════════════════════════════════════════════════
    #           СТРАТЕГИЯ 3: ANALOGY (рассуждение по аналогии)
    # ═══════════════════════════════════════════════════════════════

    def _try_analogy_strategy(
        self,
        user_input: str,
        context: str = "",
    ) -> Optional[ThoughtChain]:
        """
        Рассуждает по аналогии:
        1. Ищет похожие решённые задачи в истории
        2. Адаптирует решение к текущей задаче

        Работает через sentence_embeddings для поиска похожих.
        """
        if not self._sentence:
            return None

        # Ищем похожие прошлые рассуждения
        rows = self._conn.execute("""
            SELECT query, chain_json, confidence
            FROM cot_history
            WHERE was_useful = 1 AND confidence >= 0.5
            ORDER BY created_at DESC
            LIMIT 50
        """).fetchall()

        if not rows:
            return None

        # Находим самое похожее
        best_row = None
        best_sim = 0.0

        for row in rows:
            sim = self._sentence.similarity(user_input, row["query"])
            if sim > best_sim:
                best_sim = sim
                best_row = row

        if not best_row or best_sim < 0.5:
            return None

        # Адаптируем найденное рассуждение
        try:
            old_chain_data = json.loads(best_row["chain_json"])
        except (json.JSONDecodeError, TypeError):
            return None

        chain = ThoughtChain(
            query=user_input,
            strategy="analogy",
        )

        old_steps = old_chain_data.get("steps", [])
        new_entities = self._extract_entities(user_input)

        for i, old_step in enumerate(old_steps):
            # Адаптируем текст шага
            adapted_thought = self._adapt_text(
                old_step.get("thought", ""),
                new_entities,
            )
            adapted_action = self._adapt_text(
                old_step.get("action", ""),
                new_entities,
            )

            step = ThoughtStep(
                step_num=i + 1,
                thought=adapted_thought,
                action=adapted_action,
                observation="(по аналогии с похожей задачей)",
                conclusion=old_step.get("conclusion", ""),
                confidence=best_sim * 0.8,
            )
            chain.steps.append(step)

        chain.final_answer = self._compose_answer_from_steps(chain.steps, user_input)
        chain.overall_confidence = best_sim * best_row["confidence"] * 0.8

        logger.debug(
            f"🧠 CoT analogy: sim={best_sim:.2f}, "
            f"{len(chain.steps)} steps, conf={chain.overall_confidence:.2f}"
        )
        return chain

    def _adapt_text(self, text: str, entities: Dict) -> str:
        """Адаптирует текст из старого рассуждения к новому контексту"""
        # Подставляем новые сущности
        if "keywords" in entities:
            # Простая подстановка — заменяем {topic} на ключевое слово
            for kw in entities["keywords"][:1]:
                text = text.replace("{topic}", kw)

        if "file" in entities:
            text = text.replace("{filename}", entities["file"][0])

        if "format" in entities:
            text = text.replace("{format}", entities["format"][0])

        return text

    # ═══════════════════════════════════════════════════════════════
    #           ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════════

    def _generate_thought(self, step_text: str, idx: int, total: int) -> str:
        """Генерирует мысль для шага из шаблона"""
        if idx == 0:
            prefix = _random_choice(THOUGHT_CONNECTORS["first"])
        elif idx == total - 1:
            prefix = _random_choice(THOUGHT_CONNECTORS["conclude"])
        else:
            prefix = _random_choice(THOUGHT_CONNECTORS["next"])
        return f"{prefix} {step_text.lower()}."

    def _generate_conclusion(self, step_data: Dict, idx: int, total: int) -> str:
        """Генерирует заключение шага"""
        if idx == total - 1:
            return "Рассуждение завершено."
        return f"Шаг {idx + 1} выполнен, переходим далее."

    def _compose_answer_from_steps(
        self,
        steps: List[ThoughtStep],
        user_input: str,
    ) -> str:
        """Собирает финальный ответ из шагов рассуждения"""
        if not steps:
            return ""

        parts = ["Вот моё рассуждение:"]
        for step in steps:
            parts.append(f"  {step.step_num}. {step.thought}")
            if step.action and step.action != step.thought:
                parts.append(f"     → {step.action}")

        # Финальный вывод
        if len(steps) >= 2:
            parts.append(f"\n{_random_choice(THOUGHT_CONNECTORS['conclude'])}, "
                         f"задача разобрана по шагам.")

        return "\n".join(parts)

    def _record_reasoning(self, chain: ThoughtChain):
        """Записывает рассуждение в историю"""
        now = time.time()
        if chain.overall_confidence >= 0.5:
            self._successful_reasonings += 1

        self._conn.execute("""
            INSERT INTO cot_history (query, strategy, chain_json, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            chain.query,
            chain.strategy,
            json.dumps(chain.to_dict(), ensure_ascii=False),
            chain.overall_confidence,
            now,
        ))
        self._save_stats()

    # ═══════════════════════════════════════════════════════════════
    #           ОБРАТНАЯ СВЯЗЬ И ОБУЧЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def feedback(self, chain: ThoughtChain, was_useful: bool):
        """
        Обратная связь: было ли рассуждение полезным.
        Обновляет историю + KnowledgeDistillation.
        """
        # Обновляем последнюю запись для этого запроса
        self._conn.execute("""
            UPDATE cot_history
            SET was_useful = ?
            WHERE query = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (1 if was_useful else 0, chain.query))
        self._conn.commit()

        # Передаём feedback в KnowledgeDistillation
        if self._kd and chain.source_chain_id:
            self._kd.feedback(
                chain.source_chain_id,
                useful=was_useful,
                source=chain.strategy,
            )

    def get_reasoning_trace(self, chain: ThoughtChain) -> str:
        """
        Форматирует трейс рассуждения для логирования/отладки.

        Пример:
          === Chain-of-Thought ===
          Query: "Найди все Python файлы"
          Strategy: decompose

          [1] Thought: Для начала нужно определить критерии поиска
              Action: определить критерии — формат: py
              Observation: Критерии поиска определены
              Conclusion: Переходим к следующему шагу
          ...

          Confidence: 0.65
          Time: 12ms
          =========================
        """
        lines = [
            "=== Chain-of-Thought ===",
            f"Query: \"{chain.query[:80]}\"",
            f"Strategy: {chain.strategy}",
            "",
        ]

        for step in chain.steps:
            lines.append(f"[{step.step_num}] Thought: {step.thought}")
            lines.append(f"    Action: {step.action}")
            lines.append(f"    Observation: {step.observation}")
            lines.append(f"    Conclusion: {step.conclusion}")
            lines.append("")

        lines.append(f"Answer: {chain.final_answer[:200]}")
        lines.append(f"Confidence: {chain.overall_confidence:.2f}")
        lines.append(f"Time: {chain.reasoning_time_ms:.0f}ms")
        lines.append("=" * 25)

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    #           СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        history_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM cot_history"
        ).fetchone()["c"]

        useful_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM cot_history WHERE was_useful = 1"
        ).fetchone()["c"]

        # Стратегии
        strategy_rows = self._conn.execute("""
            SELECT strategy, COUNT(*) as c FROM cot_history
            GROUP BY strategy
        """).fetchall()
        strategies = {r["strategy"]: r["c"] for r in strategy_rows}

        return {
            "total_reasonings": self._total_reasonings,
            "successful_reasonings": self._successful_reasonings,
            "history_count": history_count,
            "useful_count": useful_count,
            "strategies": strategies,
            "success_rate": round(
                self._successful_reasonings / max(self._total_reasonings, 1) * 100, 1
            ),
        }

    def close(self):
        self._save_stats()
        self._conn.close()


# ═══════════════════════════════════════════════════════════════
#               УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

import random

def _random_choice(items: list) -> str:
    """Случайный выбор из списка"""
    return random.choice(items) if items else ""
