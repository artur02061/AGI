"""
Кристина 7.3 — Conditional Generation (Условная генерация)

ЗАЧЕМ:
  Один и тот же вопрос → разные ответы в зависимости от УСЛОВИЙ:
  - Стиль: формальный, разговорный, технический
  - Настроение: радостное, нейтральное, сочувствующее
  - Тема: код, общение, анализ
  - Формат: текст, список, пошагово

КАК РАБОТАЕТ:
  К вводу трансформера добавляются УСЛОВНЫЕ ТОКЕНЫ:

    [STYLE:formal] [MOOD:happy] [TOPIC:code] Объясни рекурсию

  Модель учится генерировать по-разному в зависимости от условий.
  Условные токены — это обучаемые эмбеддинги, отдельные от словаря.

АРХИТЕКТУРА:
  ┌─────────────────────────────────────────────┐
  │ Условия:                                    │
  │   style=formal, mood=happy, topic=code       │
  └──────────────┬──────────────────────────────┘
                 ↓
  ┌─────────────────────────────────────────────┐
  │ ConditionEncoder                            │
  │   → condition_vec [d_model]                 │
  │   (обучаемый эмбеддинг для каждого условия) │
  └──────────────┬──────────────────────────────┘
                 ↓
  ┌─────────────────────────────────────────────┐
  │ [cond_vec] + [token_1] + [token_2] + ...    │
  │              ↓                               │
  │ MicroTransformer → generate()                │
  │              ↓                               │
  │ "Рекурсия — это приём..."                    │
  └─────────────────────────────────────────────┘

ОБУЧЕНИЕ:
  При каждом диалоге определяем условия → обучаем с ними.
  Так модель связывает: formal + code → "Рекурсия — это метод..."
                         casual + code → "Ну, рекурсия это когда..."
"""

import json
import math
import random
import re
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from utils.logging import get_logger
import config

logger = get_logger("conditional_gen")


# ═══════════════════════════════════════════════════════════════
#               УСЛОВИЯ ГЕНЕРАЦИИ
# ═══════════════════════════════════════════════════════════════

# Все поддерживаемые условия и их значения
CONDITIONS = {
    "style": {
        "formal": 0,       # Формальный ("Рекурсия — это метод...")
        "casual": 1,       # Разговорный ("Ну, рекурсия это когда...")
        "technical": 2,    # Технический ("Рекурсивная функция f(n) = ...")
        "friendly": 3,     # Дружелюбный ("Привет! Рекурсия — это...")
    },
    "mood": {
        "neutral": 0,      # Нейтральное
        "happy": 1,        # Радостное
        "empathetic": 2,   # Сочувствующее
        "enthusiastic": 3, # Воодушевлённое
    },
    "topic": {
        "general": 0,      # Общая тема
        "code": 1,         # Программирование
        "system": 2,       # Системные задачи
        "creative": 3,     # Творчество
        "analysis": 4,     # Анализ
    },
    "format": {
        "text": 0,         # Обычный текст
        "list": 1,         # Список
        "steps": 2,        # Пошагово
        "brief": 3,        # Кратко
        "detailed": 4,     # Подробно
    },
}

# Сколько всего уникальных condition значений
TOTAL_CONDITION_VALUES = sum(len(v) for v in CONDITIONS.values())

# Маркеры для авто-определения условий из текста
STYLE_MARKERS = {
    "formal": ["объясни", "определи", "расскажи подробно", "опиши"],
    "casual": ["ну", "прикинь", "короче", "чё", "как бы"],
    "technical": ["реализуй", "алгоритм", "функция", "класс", "api", "код"],
    "friendly": ["привет", "помоги", "подскажи", "будь добра"],
}

MOOD_MARKERS = {
    "happy": ["отлично", "здорово", "круто", "супер", "ура"],
    "empathetic": ["грустно", "плохо", "устал", "тяжело", "проблема"],
    "enthusiastic": ["давай", "классно", "wow", "обожаю", "хочу"],
}

TOPIC_MARKERS = {
    "code": ["код", "python", "функция", "класс", "программ", "скрипт",
             "баг", "ошибка", "файл", "git", "api"],
    "system": ["запусти", "установи", "настрой", "терминал", "система",
               "сервер", "docker", "процесс"],
    "creative": ["напиши стих", "история", "сказка", "придумай", "фантазия"],
    "analysis": ["проанализируй", "сравни", "статистика", "данные", "отчёт"],
}

FORMAT_MARKERS = {
    "list": ["список", "перечисли", "варианты", "пункты"],
    "steps": ["пошагово", "по шагам", "инструкция", "как сделать"],
    "brief": ["кратко", "коротко", "в двух словах", "суть"],
    "detailed": ["подробно", "детально", "полностью", "развёрнуто"],
}


# ═══════════════════════════════════════════════════════════════
#               CONDITION ENCODER
# ═══════════════════════════════════════════════════════════════


@dataclass
class GenerationConditions:
    """Условия для генерации"""
    style: str = "friendly"
    mood: str = "neutral"
    topic: str = "general"
    format: str = "text"

    def to_dict(self) -> Dict[str, str]:
        return {
            "style": self.style,
            "mood": self.mood,
            "topic": self.topic,
            "format": self.format,
        }

    def __repr__(self) -> str:
        return f"[STYLE:{self.style}] [MOOD:{self.mood}] [TOPIC:{self.topic}] [FMT:{self.format}]"


class ConditionEncoder:
    """
    Кодирует условия генерации в вектор [d_model].

    Каждое значение условия имеет обучаемый эмбеддинг.
    Итоговый condition_vec = сумма эмбеддингов всех условий.
    """

    def __init__(self, d_model: int = 128):
        self.d_model = d_model

        # Обучаемые эмбеддинги для каждого condition value
        self._embeddings: Dict[str, Dict[str, List[float]]] = {}
        scale = math.sqrt(1.0 / d_model)

        for cond_type, values in CONDITIONS.items():
            self._embeddings[cond_type] = {}
            for value_name in values:
                self._embeddings[cond_type][value_name] = [
                    random.gauss(0, scale) for _ in range(d_model)
                ]

    def encode(self, conditions: GenerationConditions) -> List[float]:
        """
        Кодирует условия в один вектор [d_model].
        Суммирует эмбеддинги всех заданных условий.
        """
        result = [0.0] * self.d_model

        for cond_type, value in conditions.to_dict().items():
            emb = self._embeddings.get(cond_type, {}).get(value)
            if emb:
                for i in range(self.d_model):
                    result[i] += emb[i]

        # Нормализуем
        norm = math.sqrt(sum(x * x for x in result) + 1e-10)
        if norm > 0:
            scale = math.sqrt(self.d_model) / norm
            result = [x * scale for x in result]

        return result

    def get_embeddings_data(self) -> Dict:
        """Сериализует эмбеддинги для сохранения"""
        return {
            cond_type: {
                value: emb
                for value, emb in values.items()
            }
            for cond_type, values in self._embeddings.items()
        }

    def load_embeddings_data(self, data: Dict):
        """Загружает эмбеддинги из сохранения"""
        for cond_type, values in data.items():
            if cond_type in self._embeddings:
                for value, emb in values.items():
                    if value in self._embeddings[cond_type]:
                        if len(emb) == self.d_model:
                            self._embeddings[cond_type][value] = emb


# ═══════════════════════════════════════════════════════════════
#               CONDITIONAL GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════


class ConditionalGeneration:
    """
    Условная генерация: генерирует текст с учётом стиля, настроения, темы.

    Использование:
        cg = ConditionalGeneration(micro_transformer, bpe_tokenizer)

        # Авто-определение условий
        conditions = cg.detect_conditions("Привет! Напиши код сортировки")
        # → style=friendly, mood=neutral, topic=code, format=text

        # Генерация с условиями
        text = cg.generate(
            prompt="Объясни рекурсию",
            conditions=GenerationConditions(style="technical", topic="code"),
        )

        # Обучение
        cg.train(
            text="Рекурсия — это приём программирования...",
            conditions=conditions,
        )
    """

    def __init__(
        self,
        micro_transformer=None,
        bpe_tokenizer=None,
        d_model: int = 128,
        db_path: Path = None,
    ):
        self._transformer = micro_transformer
        self._tokenizer = bpe_tokenizer
        self.d_model = d_model

        # Condition encoder
        self.condition_encoder = ConditionEncoder(d_model)

        # Persistence
        self._db_path = db_path or (config.config.data_dir / "conditional_gen.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Stats
        self._total_generations = 0
        self._condition_usage: Dict[str, int] = {}
        self._load_state()

        logger.info(
            f"🎭 ConditionalGen: {TOTAL_CONDITION_VALUES} condition values, "
            f"{self._total_generations} generations"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cond_gen_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cond_gen_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                output_len INTEGER,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def _load_state(self):
        row = self._conn.execute(
            "SELECT value FROM cond_gen_state WHERE key = 'total_generations'"
        ).fetchone()
        if row:
            self._total_generations = int(row["value"])

        row = self._conn.execute(
            "SELECT value FROM cond_gen_state WHERE key = 'condition_embeddings'"
        ).fetchone()
        if row:
            try:
                data = json.loads(row["value"])
                self.condition_encoder.load_embeddings_data(data)
            except (json.JSONDecodeError, TypeError):
                pass

        row = self._conn.execute(
            "SELECT value FROM cond_gen_state WHERE key = 'condition_usage'"
        ).fetchone()
        if row:
            try:
                self._condition_usage = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_state(self):
        now = time.time()
        data = [
            ("total_generations", str(self._total_generations)),
            ("condition_embeddings", json.dumps(
                self.condition_encoder.get_embeddings_data()
            )),
            ("condition_usage", json.dumps(self._condition_usage)),
        ]
        for key, val in data:
            self._conn.execute("""
                INSERT INTO cond_gen_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, val, val))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           ОПРЕДЕЛЕНИЕ УСЛОВИЙ
    # ═══════════════════════════════════════════════════════════════

    def detect_conditions(
        self,
        user_input: str,
        mood: str = None,
        context: str = "",
    ) -> GenerationConditions:
        """
        Автоматически определяет условия генерации из текста.

        Анализирует ключевые слова, настроение, тему.
        """
        text = user_input.lower()
        conditions = GenerationConditions()

        # Style
        best_style = "friendly"
        best_style_score = 0
        for style, markers in STYLE_MARKERS.items():
            score = sum(1 for m in markers if m in text)
            if score > best_style_score:
                best_style_score = score
                best_style = style
        conditions.style = best_style

        # Mood
        if mood:
            conditions.mood = mood
        else:
            best_mood = "neutral"
            best_mood_score = 0
            for m, markers in MOOD_MARKERS.items():
                score = sum(1 for marker in markers if marker in text)
                if score > best_mood_score:
                    best_mood_score = score
                    best_mood = m
            conditions.mood = best_mood

        # Topic
        best_topic = "general"
        best_topic_score = 0
        for topic, markers in TOPIC_MARKERS.items():
            score = sum(1 for m in markers if m in text)
            if score > best_topic_score:
                best_topic_score = score
                best_topic = topic
        conditions.topic = best_topic

        # Format
        best_format = "text"
        best_format_score = 0
        for fmt, markers in FORMAT_MARKERS.items():
            score = sum(1 for m in markers if m in text)
            if score > best_format_score:
                best_format_score = score
                best_format = fmt
        conditions.format = best_format

        # Track usage
        key = repr(conditions)
        self._condition_usage[key] = self._condition_usage.get(key, 0) + 1

        return conditions

    # ═══════════════════════════════════════════════════════════════
    #           ГЕНЕРАЦИЯ С УСЛОВИЯМИ
    # ═══════════════════════════════════════════════════════════════

    def generate(
        self,
        prompt: str,
        conditions: GenerationConditions = None,
        max_len: int = 50,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.9,
    ) -> Optional[str]:
        """
        Генерирует текст с учётом условий.

        1. Кодирует условия в condition_vec
        2. Модифицирует эмбеддинги промпта condition_vec-ом
        3. Генерирует через MicroTransformer
        4. Пост-обрабатывает в соответствии с format

        Returns:
            Сгенерированный текст или None
        """
        if not self._transformer or not self._tokenizer:
            return None

        if self._transformer._training_steps < 20:
            return None  # Модель недостаточно обучена

        if conditions is None:
            conditions = self.detect_conditions(prompt)

        self._total_generations += 1

        # 1. Кодируем условия
        cond_vec = self.condition_encoder.encode(conditions)

        # 2. Токенизируем промпт
        prompt_ids = self._tokenizer.encode(prompt)
        if not prompt_ids or len(prompt_ids) < 1:
            return None

        # 3. Адаптируем temperature и параметры по условиям
        temperature = self._adjust_temperature(temperature, conditions)
        max_len = self._adjust_max_len(max_len, conditions)

        # 4. Генерируем через трансформер
        # Внедряем condition: bias на эмбеддинги
        try:
            generated_ids = self._generate_with_condition(
                prompt_ids, cond_vec, max_len, temperature, top_k, top_p,
            )
        except Exception as e:
            logger.debug(f"ConditionalGen generation failed: {e}")
            return None

        if not generated_ids:
            return None

        # 5. Декодируем
        new_ids = generated_ids[len(prompt_ids):]
        if not new_ids:
            return None

        text = self._tokenizer.decode(new_ids).strip()
        if len(text) < 3:
            return None

        # 6. Пост-обработка по формату
        text = self._postprocess(text, conditions)

        # 7. Логируем
        self._conn.execute("""
            INSERT INTO cond_gen_log (prompt, conditions_json, output_len, created_at)
            VALUES (?, ?, ?, ?)
        """, (prompt[:200], json.dumps(conditions.to_dict()), len(text), time.time()))

        if self._total_generations % 20 == 0:
            self._save_state()

        logger.debug(
            f"🎭 Generated: {conditions} → {len(text)} chars"
        )

        return text

    def _generate_with_condition(
        self,
        prompt_ids: List[int],
        cond_vec: List[float],
        max_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        """
        Генерация с condition bias.

        Condition vector добавляется к каждому эмбеддингу как bias,
        смещая распределение в нужную сторону.
        """
        # Сохраняем оригинальные bias output
        original_bias = list(self._transformer.output_bias)

        # Добавляем condition bias к output bias
        # Это смещает распределение токенов в соответствии с условиями
        cond_projection = self._project_condition_to_vocab(cond_vec)
        for i in range(min(len(self._transformer.output_bias), len(cond_projection))):
            self._transformer.output_bias[i] += cond_projection[i] * 0.1

        try:
            generated = self._transformer.generate(
                prompt_ids,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        finally:
            # Восстанавливаем оригинальные bias
            self._transformer.output_bias = original_bias

        return generated

    def _project_condition_to_vocab(self, cond_vec: List[float]) -> List[float]:
        """
        Проецирует condition vector [d_model] → [vocab_size].
        Использует embedding weight как проекцию (tied).
        """
        vocab_size = self._transformer.vocab_size
        d_model = len(cond_vec)

        # cond_vec @ embedding.T → [vocab_size]
        result = [0.0] * vocab_size
        for token_id in range(min(vocab_size, len(self._transformer.embedding.weight))):
            emb = self._transformer.embedding.weight[token_id]
            result[token_id] = sum(
                cond_vec[j] * emb[j]
                for j in range(min(d_model, len(emb)))
            )

        return result

    def _adjust_temperature(
        self,
        base_temp: float,
        conditions: GenerationConditions,
    ) -> float:
        """Адаптирует temperature под условия"""
        temp = base_temp

        # Formal → ниже temperature (точнее)
        if conditions.style == "formal":
            temp *= 0.7
        elif conditions.style == "casual":
            temp *= 1.2
        elif conditions.style == "technical":
            temp *= 0.6

        # Brief → ниже (точнее)
        if conditions.format == "brief":
            temp *= 0.8
        elif conditions.format == "detailed":
            temp *= 1.1

        # Enthusiastic → выше (разнообразнее)
        if conditions.mood == "enthusiastic":
            temp *= 1.15

        return max(0.1, min(1.5, temp))

    def _adjust_max_len(
        self,
        base_len: int,
        conditions: GenerationConditions,
    ) -> int:
        """Адаптирует максимальную длину"""
        length = base_len

        if conditions.format == "brief":
            length = min(length, 20)
        elif conditions.format == "detailed":
            length = max(length, 80)
        elif conditions.format == "steps":
            length = max(length, 60)

        return length

    def _postprocess(self, text: str, conditions: GenerationConditions) -> str:
        """Пост-обработка текста по условиям формата"""
        # Для формата "list" — добавляем маркеры если нет
        if conditions.format == "list" and not re.search(r'^\s*[-•\d]', text):
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
            if len(sentences) > 1:
                text = "\n".join(f"• {s}" for s in sentences)

        # Для формата "steps" — нумеруем
        if conditions.format == "steps" and not re.search(r'^\s*\d', text):
            sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
            if len(sentences) > 1:
                text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

        # Для brief — обрезаем
        if conditions.format == "brief":
            sentences = re.split(r'[.!?]\s+', text)
            if len(sentences) > 2:
                text = ". ".join(sentences[:2]) + "."

        return text

    # ═══════════════════════════════════════════════════════════════
    #           ОБУЧЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def train(
        self,
        text: str,
        conditions: GenerationConditions,
    ):
        """
        Обучает модель генерировать с заданными условиями.

        Добавляет condition bias при обучении, так модель ассоциирует
        условия с определёнными стилями текста.
        """
        if not self._transformer or not self._tokenizer:
            return

        token_ids = self._tokenizer.encode(text)
        if len(token_ids) < 3:
            return

        # Кодируем условия
        cond_vec = self.condition_encoder.encode(conditions)

        # Добавляем condition bias на время обучения
        original_bias = list(self._transformer.output_bias)
        cond_projection = self._project_condition_to_vocab(cond_vec)

        for i in range(min(len(self._transformer.output_bias), len(cond_projection))):
            self._transformer.output_bias[i] += cond_projection[i] * 0.05

        try:
            self._transformer.train_step(token_ids)
        finally:
            self._transformer.output_bias = original_bias

    # ═══════════════════════════════════════════════════════════════
    #           СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        # Top conditions
        top_conditions = sorted(
            self._condition_usage.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_generations": self._total_generations,
            "condition_types": len(CONDITIONS),
            "condition_values": TOTAL_CONDITION_VALUES,
            "top_conditions": top_conditions,
        }

    def close(self):
        self._save_state()
        self._conn.close()
