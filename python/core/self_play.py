"""
Кристина 7.3 — Self-Play Engine (Самооценка через LLM)

ЗАЧЕМ:
  Кристина генерирует ответ → LLM оценивает его → Кристина учится на оценке.
  Это аналог RLHF (Reinforcement Learning from Human Feedback), но:
  - Feedback приходит от LLM-учителя, а не от человека
  - Работает автоматически, без участия пользователя
  - Каждая оценка улучшает ВСЕ компоненты Кристины

КАК РАБОТАЕТ:
  ┌──────────────────────────────────────────────────────┐
  │                Self-Play Loop                        │
  │                                                      │
  │  1. Берём вопрос (реальный или синтетический)        │
  │  2. Кристина генерирует ответ (без LLM)             │
  │  3. LLM оценивает: 1-10 + объяснение ошибок         │
  │  4. score >= порог → reinforce паттерн               │
  │     score < порог  → weaken + запомнить правильный   │
  │  5. Порог постепенно растёт: 5 → 6 → 7 → 8         │
  │                                                      │
  │  Режимы:                                             │
  │  - online: оценка после каждого ответа (1 LLM-call) │
  │  - batch:  оценка N ответов за раз (1 LLM-call)     │
  │  - exam:   тест на синтетических вопросах            │
  └──────────────────────────────────────────────────────┘

ИНТЕГРАЦИЯ:
  - Оркестратор вызывает self_play.evaluate() после каждого ответа Tier 1-3
  - Батчевая оценка: раз в N диалогов
  - Обучение: reinforcement → LearnedPatterns, NeuralEngine, KD
"""

import sqlite3
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from utils.logging import get_logger
import config

logger = get_logger("self_play")


# ═══════════════════════════════════════════════════════════════
#               СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════

@dataclass
class Evaluation:
    """Результат оценки одного ответа"""
    question: str
    kristina_answer: str
    score: float            # 1-10
    feedback: str           # Объяснение от LLM
    strengths: List[str]    # Что хорошо
    weaknesses: List[str]   # Что плохо
    correct_answer: str     # Эталонный ответ от LLM (если score < порога)
    source_tier: str        # Откуда ответ: "tier1", "tier2", "tier3"
    reinforced: bool        # Был ли reinforced
    timestamp: float


@dataclass
class ExamResult:
    """Результат экзамена (серии вопросов)"""
    total_questions: int
    avg_score: float
    pass_rate: float        # % ответов выше порога
    by_category: Dict[str, float]  # Средний балл по категориям
    improvements: List[str]  # Области для улучшения
    timestamp: float


# ═══════════════════════════════════════════════════════════════
#               ШАБЛОНЫ ПРОМПТОВ
# ═══════════════════════════════════════════════════════════════

EVAL_PROMPT_TEMPLATE = """Оцени ответ ИИ-ассистента "Кристина" на вопрос пользователя.

Вопрос пользователя: {question}

Ответ Кристины: {answer}

Оцени по шкале 1-10:
- 1-3: Неправильный или вредный ответ
- 4-5: Частично правильный, есть существенные ошибки
- 6-7: В целом правильный, но неполный или неточный
- 8-9: Хороший ответ с минимальными замечаниями
- 10: Идеальный ответ

Ответь СТРОГО в формате JSON:
{{
  "score": <число 1-10>,
  "feedback": "<краткое объяснение оценки>",
  "strengths": ["<что хорошо>"],
  "weaknesses": ["<что плохо>"],
  "correct_answer": "<правильный ответ, если score < 7, иначе пустая строка>"
}}"""

BATCH_EVAL_PROMPT_TEMPLATE = """Оцени ответы ИИ-ассистента "Кристина". Для каждого дай оценку 1-10.

{qa_pairs}

Ответь в формате JSON-массив:
[
  {{"index": 0, "score": <1-10>, "feedback": "<кратко>", "weaknesses": ["<что плохо>"]}},
  ...
]"""

# Синтетические вопросы для экзамена
EXAM_QUESTIONS = {
    "greeting": [
        "Привет!",
        "Добрый день",
        "Здравствуй, как дела?",
    ],
    "self_awareness": [
        "Кто ты?",
        "Как тебя зовут?",
        "Что ты умеешь?",
    ],
    "help": [
        "Помоги мне создать файл",
        "Можешь объяснить что такое рекурсия?",
        "Покажи пример кода на Python",
    ],
    "emotion": [
        "Мне грустно сегодня",
        "У меня отличное настроение!",
        "Я устал от работы",
    ],
    "knowledge": [
        "Что такое машинное обучение?",
        "Объясни разницу между списком и словарём в Python",
        "Как работает интернет?",
    ],
}


# ═══════════════════════════════════════════════════════════════
#               SELF-PLAY ENGINE
# ═══════════════════════════════════════════════════════════════

class SelfPlay:
    """
    Self-Play: Кристина учится через самооценку с помощью LLM-учителя.

    Три режима:
    1. online  — оценка после каждого ответа Tier 1-3
    2. batch   — накопить N ответов → оценить за 1 LLM-вызов
    3. exam    — тест на синтетических вопросах

    Использование:
        sp = SelfPlay(director_agent, learned_patterns, neural_engine, kd)

        # После ответа Кристины (без LLM):
        evaluation = await sp.evaluate(
            question="Привет!",
            kristina_answer="Привет! Рада тебя видеть!",
            source_tier="tier1",
        )

        # Батчевая оценка:
        results = await sp.evaluate_batch()

        # Экзамен:
        exam = await sp.run_exam(generate_fn=orchestrator.generate_without_llm)
    """

    def __init__(
        self,
        director=None,
        learned_patterns=None,
        neural_engine=None,
        knowledge_distillation=None,
        chain_of_thought=None,
        db_path: Path = None,
    ):
        self._director = director
        self._patterns = learned_patterns
        self._neural = neural_engine
        self._kd = knowledge_distillation
        self._cot = chain_of_thought

        self._db_path = db_path or (config.config.data_dir / "self_play.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Настройки
        self._threshold = 6.0       # Минимальный балл для reinforce
        self._batch_size = 10       # Сколько ответов накопить для batch
        self._batch_buffer: List[Dict] = []  # Буфер для batch-оценки

        # Статистика
        self._total_evals = 0
        self._total_score = 0.0
        self._reinforced_count = 0
        self._weakened_count = 0
        self._load_state()

        logger.info(
            f"🎮 SelfPlay: {self._total_evals} оценок, "
            f"avg={self._avg_score:.1f}, threshold={self._threshold}, "
            f"reinforced={self._reinforced_count}, weakened={self._weakened_count}"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                kristina_answer TEXT NOT NULL,
                score REAL NOT NULL,
                feedback TEXT,
                strengths_json TEXT DEFAULT '[]',
                weaknesses_json TEXT DEFAULT '[]',
                correct_answer TEXT DEFAULT '',
                source_tier TEXT DEFAULT 'unknown',
                reinforced INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS exam_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_questions INTEGER NOT NULL,
                avg_score REAL NOT NULL,
                pass_rate REAL NOT NULL,
                by_category_json TEXT DEFAULT '{}',
                improvements_json TEXT DEFAULT '[]',
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS self_play_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_score ON evaluations(score)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_tier ON evaluations(source_tier)"
        )
        self._conn.commit()

    def _load_state(self):
        for key in ("total_evals", "total_score", "reinforced_count",
                     "weakened_count", "threshold"):
            row = self._conn.execute(
                "SELECT value FROM self_play_state WHERE key = ?", (key,)
            ).fetchone()
            if row:
                val = row["value"]
                if key == "total_evals":
                    self._total_evals = int(val)
                elif key == "total_score":
                    self._total_score = float(val)
                elif key == "reinforced_count":
                    self._reinforced_count = int(val)
                elif key == "weakened_count":
                    self._weakened_count = int(val)
                elif key == "threshold":
                    self._threshold = float(val)

    def _save_state(self):
        for key, val in [
            ("total_evals", str(self._total_evals)),
            ("total_score", str(self._total_score)),
            ("reinforced_count", str(self._reinforced_count)),
            ("weakened_count", str(self._weakened_count)),
            ("threshold", str(self._threshold)),
        ]:
            self._conn.execute("""
                INSERT INTO self_play_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, val, val))
        self._conn.commit()

    @property
    def _avg_score(self) -> float:
        if self._total_evals == 0:
            return 0.0
        return self._total_score / self._total_evals

    # ═══════════════════════════════════════════════════════════════
    #           РЕЖИМ 1: ONLINE (оценка одного ответа)
    # ═══════════════════════════════════════════════════════════════

    async def evaluate(
        self,
        question: str,
        kristina_answer: str,
        source_tier: str = "unknown",
    ) -> Optional[Evaluation]:
        """
        Оценивает один ответ Кристины через LLM.

        1. Формирует промпт для оценки
        2. LLM оценивает: score 1-10 + feedback
        3. score >= threshold → reinforce
        4. score < threshold → weaken + запомнить правильный ответ

        Returns:
            Evaluation или None (если LLM недоступен)
        """
        if not self._director:
            return None

        # Формируем промпт
        prompt = EVAL_PROMPT_TEMPLATE.format(
            question=question,
            answer=kristina_answer,
        )

        try:
            # Вызываем LLM для оценки
            raw_response = await self._director.execute(
                {"type": "evaluate", "input": prompt},
            )

            # Парсим JSON из ответа
            eval_data = self._parse_eval_response(raw_response)
            if not eval_data:
                logger.warning("SelfPlay: не удалось спарсить оценку LLM")
                return None

            evaluation = Evaluation(
                question=question,
                kristina_answer=kristina_answer,
                score=eval_data["score"],
                feedback=eval_data.get("feedback", ""),
                strengths=eval_data.get("strengths", []),
                weaknesses=eval_data.get("weaknesses", []),
                correct_answer=eval_data.get("correct_answer", ""),
                source_tier=source_tier,
                reinforced=False,
                timestamp=time.time(),
            )

            # Применяем reinforcement
            self._apply_reinforcement(evaluation)

            # Сохраняем
            self._record_evaluation(evaluation)

            logger.info(
                f"🎮 SelfPlay: score={evaluation.score}/10, "
                f"{'✅ reinforced' if evaluation.reinforced else '❌ weakened'}, "
                f"tier={source_tier}"
            )

            return evaluation

        except Exception as e:
            logger.error(f"SelfPlay evaluate error: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    #           РЕЖИМ 2: BATCH (накопительная оценка)
    # ═══════════════════════════════════════════════════════════════

    def add_to_batch(self, question: str, kristina_answer: str, source_tier: str = "unknown"):
        """
        Добавляет ответ в буфер для батчевой оценки.
        Когда буфер заполнится — нужно вызвать evaluate_batch().
        """
        self._batch_buffer.append({
            "question": question,
            "answer": kristina_answer,
            "source_tier": source_tier,
            "timestamp": time.time(),
        })

    @property
    def batch_ready(self) -> bool:
        """Готов ли буфер к батчевой оценке"""
        return len(self._batch_buffer) >= self._batch_size

    async def evaluate_batch(self) -> List[Evaluation]:
        """
        Оценивает накопленный буфер ответов за один LLM-вызов.
        Экономит API-вызовы (1 вместо N).
        """
        if not self._director or not self._batch_buffer:
            return []

        # Формируем единый промпт
        qa_pairs = []
        for i, item in enumerate(self._batch_buffer):
            qa_pairs.append(
                f"[{i}] Вопрос: {item['question']}\n"
                f"    Ответ: {item['answer']}"
            )

        prompt = BATCH_EVAL_PROMPT_TEMPLATE.format(
            qa_pairs="\n\n".join(qa_pairs)
        )

        try:
            raw_response = await self._director.execute(
                {"type": "evaluate_batch", "input": prompt},
            )

            # Парсим массив оценок
            eval_results = self._parse_batch_response(raw_response)

            evaluations = []
            for i, item in enumerate(self._batch_buffer):
                eval_data = eval_results[i] if i < len(eval_results) else {"score": 5.0}

                evaluation = Evaluation(
                    question=item["question"],
                    kristina_answer=item["answer"],
                    score=eval_data.get("score", 5.0),
                    feedback=eval_data.get("feedback", ""),
                    strengths=[],
                    weaknesses=eval_data.get("weaknesses", []),
                    correct_answer="",
                    source_tier=item["source_tier"],
                    reinforced=False,
                    timestamp=item["timestamp"],
                )

                self._apply_reinforcement(evaluation)
                self._record_evaluation(evaluation)
                evaluations.append(evaluation)

            # Очищаем буфер
            self._batch_buffer.clear()

            avg = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0
            logger.info(
                f"🎮 SelfPlay batch: {len(evaluations)} оценок, "
                f"avg={avg:.1f}, reinforced={sum(1 for e in evaluations if e.reinforced)}"
            )

            return evaluations

        except Exception as e:
            logger.error(f"SelfPlay batch error: {e}")
            return []

    # ═══════════════════════════════════════════════════════════════
    #           РЕЖИМ 3: EXAM (тест на синтетических вопросах)
    # ═══════════════════════════════════════════════════════════════

    async def run_exam(
        self,
        generate_fn=None,
        categories: List[str] = None,
        questions_per_category: int = 3,
    ) -> Optional[ExamResult]:
        """
        Проводит экзамен: генерирует ответы на тестовые вопросы
        и оценивает их через LLM.

        Args:
            generate_fn: async функция для генерации ответа Кристины
                         (без LLM, например orchestrator._generate_local)
            categories: какие категории тестировать (по умолчанию все)
            questions_per_category: сколько вопросов на категорию

        Returns:
            ExamResult с результатами
        """
        if not generate_fn or not self._director:
            return None

        cats = categories or list(EXAM_QUESTIONS.keys())
        by_category: Dict[str, List[float]] = {}

        all_evals = []

        for cat in cats:
            questions = EXAM_QUESTIONS.get(cat, [])[:questions_per_category]
            by_category[cat] = []

            for q in questions:
                # Кристина генерирует ответ
                try:
                    answer = await generate_fn(q)
                except Exception:
                    answer = None

                if not answer:
                    by_category[cat].append(1.0)
                    continue

                # Оцениваем
                evaluation = await self.evaluate(q, answer, source_tier="exam")
                if evaluation:
                    by_category[cat].append(evaluation.score)
                    all_evals.append(evaluation)
                else:
                    by_category[cat].append(5.0)  # Средняя оценка если не удалось

        # Подсчёт результатов
        all_scores = [e.score for e in all_evals]
        if not all_scores:
            return None

        avg_score = sum(all_scores) / len(all_scores)
        pass_rate = sum(1 for s in all_scores if s >= self._threshold) / len(all_scores) * 100

        cat_averages = {
            cat: round(sum(scores) / len(scores), 1) if scores else 0.0
            for cat, scores in by_category.items()
        }

        # Определяем слабые места
        improvements = []
        for cat, avg in cat_averages.items():
            if avg < self._threshold:
                improvements.append(f"{cat}: {avg}/10 (ниже порога {self._threshold})")

        result = ExamResult(
            total_questions=len(all_scores),
            avg_score=round(avg_score, 1),
            pass_rate=round(pass_rate, 1),
            by_category=cat_averages,
            improvements=improvements,
            timestamp=time.time(),
        )

        # Сохраняем результат экзамена
        self._conn.execute("""
            INSERT INTO exam_results
            (total_questions, avg_score, pass_rate, by_category_json,
             improvements_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            result.total_questions,
            result.avg_score,
            result.pass_rate,
            json.dumps(result.by_category, ensure_ascii=False),
            json.dumps(result.improvements, ensure_ascii=False),
            result.timestamp,
        ))
        self._conn.commit()

        # Повышаем порог если сдали хорошо
        if pass_rate >= 80.0 and self._threshold < 9.0:
            old = self._threshold
            self._threshold = min(9.0, self._threshold + 0.5)
            self._save_state()
            logger.info(
                f"🎮 SelfPlay: порог повышен {old} → {self._threshold} "
                f"(pass_rate={pass_rate}%)"
            )

        logger.info(
            f"🎮 SelfPlay exam: {result.total_questions} вопросов, "
            f"avg={result.avg_score}, pass={result.pass_rate}%, "
            f"improvements={len(result.improvements)}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    #           REINFORCEMENT (усиление/ослабление)
    # ═══════════════════════════════════════════════════════════════

    def _apply_reinforcement(self, evaluation: Evaluation):
        """
        Применяет reinforcement к компонентам Кристины.

        score >= threshold → REINFORCE (усиливаем паттерн)
        score < threshold  → WEAKEN (ослабляем) + LEARN (запоминаем правильный)
        """
        if evaluation.score >= self._threshold:
            # ✅ Reinforcement: усиливаем паттерн
            evaluation.reinforced = True
            self._reinforced_count += 1

            # Усиливаем в LearnedPatterns
            if self._patterns:
                self._patterns.reinforce_last_match(
                    boost=0.1 * (evaluation.score / 10.0),
                )

            # Положительная обратная связь в KD
            if self._kd:
                # Если CoT использовал цепочку — усиливаем
                pass  # feedback уже применяется через CoT

            logger.debug(
                f"🎮 Reinforce: score={evaluation.score}, "
                f"q='{evaluation.question[:30]}...'"
            )
        else:
            # ❌ Weaken: ослабляем и учимся
            evaluation.reinforced = False
            self._weakened_count += 1

            # Ослабляем в LearnedPatterns
            if self._patterns:
                self._patterns.weaken_last_match(
                    penalty=0.15 * (1 - evaluation.score / 10.0),
                )

            # Запоминаем правильный ответ
            if evaluation.correct_answer and self._neural:
                # Обучаем NeuralEngine на правильном ответе
                self._neural.learn_from_text(
                    evaluation.correct_answer,
                    source="self_play_correction",
                )

            # Дистиллируем правильный ответ
            if evaluation.correct_answer and self._kd:
                self._kd.distill(
                    user_input=evaluation.question,
                    llm_response=evaluation.correct_answer,
                    intent="self_play_correction",
                    result_success=True,
                )

            logger.debug(
                f"🎮 Weaken: score={evaluation.score}, "
                f"weaknesses={evaluation.weaknesses}, "
                f"q='{evaluation.question[:30]}...'"
            )

    # ═══════════════════════════════════════════════════════════════
    #           ПАРСИНГ ОТВЕТОВ LLM
    # ═══════════════════════════════════════════════════════════════

    def _parse_eval_response(self, raw: str) -> Optional[Dict]:
        """Парсит JSON-оценку из ответа LLM"""
        # Ищем JSON в ответе
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 5))
                score = max(1.0, min(10.0, score))
                data["score"] = score
                return data
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: ищем просто число
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', raw)
        if score_match:
            score = float(score_match.group(1))
            return {
                "score": max(1.0, min(10.0, score)),
                "feedback": raw[:200],
                "strengths": [],
                "weaknesses": [],
                "correct_answer": "",
            }

        # Ищем "оценка: N" или "score: N"
        score_match = re.search(r'(?:оценка|score|балл)[:\s]+(\d+(?:\.\d+)?)', raw, re.I)
        if score_match:
            score = float(score_match.group(1))
            return {
                "score": max(1.0, min(10.0, score)),
                "feedback": raw[:200],
                "strengths": [],
                "weaknesses": [],
                "correct_answer": "",
            }

        return None

    def _parse_batch_response(self, raw: str) -> List[Dict]:
        """Парсит массив оценок из batch-ответа"""
        # Ищем JSON-массив
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    # Нормализуем scores
                    for item in data:
                        if "score" in item:
                            item["score"] = max(1.0, min(10.0, float(item["score"])))
                    return data
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: парсим построчно
        results = []
        for match in re.finditer(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw):
            results.append({"score": max(1.0, min(10.0, float(match.group(1))))})

        return results

    # ═══════════════════════════════════════════════════════════════
    #           ЗАПИСЬ И СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def _record_evaluation(self, evaluation: Evaluation):
        """Записывает оценку в историю"""
        self._total_evals += 1
        self._total_score += evaluation.score

        self._conn.execute("""
            INSERT INTO evaluations
            (question, kristina_answer, score, feedback,
             strengths_json, weaknesses_json, correct_answer,
             source_tier, reinforced, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation.question,
            evaluation.kristina_answer,
            evaluation.score,
            evaluation.feedback,
            json.dumps(evaluation.strengths, ensure_ascii=False),
            json.dumps(evaluation.weaknesses, ensure_ascii=False),
            evaluation.correct_answer,
            evaluation.source_tier,
            1 if evaluation.reinforced else 0,
            evaluation.timestamp,
        ))

        self._save_state()

    def get_stats(self) -> Dict:
        """Статистика Self-Play"""
        # Средние баллы по tier-ам
        tier_stats = {}
        for tier in ("tier1", "tier2", "tier3", "exam"):
            row = self._conn.execute("""
                SELECT AVG(score) as avg, COUNT(*) as cnt
                FROM evaluations WHERE source_tier = ?
            """, (tier,)).fetchone()
            if row and row["cnt"] > 0:
                tier_stats[tier] = {
                    "avg_score": round(row["avg"], 1),
                    "count": row["cnt"],
                }

        # Последний экзамен
        last_exam = self._conn.execute("""
            SELECT * FROM exam_results ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        # Тренд (последние 50 оценок)
        recent = self._conn.execute("""
            SELECT score FROM evaluations
            ORDER BY created_at DESC LIMIT 50
        """).fetchall()
        recent_scores = [r["score"] for r in recent]

        trend = "stable"
        if len(recent_scores) >= 10:
            first_half = sum(recent_scores[len(recent_scores)//2:]) / max(len(recent_scores)//2, 1)
            second_half = sum(recent_scores[:len(recent_scores)//2]) / max(len(recent_scores)//2, 1)
            if second_half > first_half + 0.3:
                trend = "improving"
            elif second_half < first_half - 0.3:
                trend = "declining"

        return {
            "total_evaluations": self._total_evals,
            "avg_score": round(self._avg_score, 1),
            "threshold": self._threshold,
            "reinforced": self._reinforced_count,
            "weakened": self._weakened_count,
            "reinforce_rate": round(
                self._reinforced_count / max(self._total_evals, 1) * 100, 1
            ),
            "tier_stats": tier_stats,
            "trend": trend,
            "batch_buffer_size": len(self._batch_buffer),
            "last_exam": {
                "avg_score": last_exam["avg_score"],
                "pass_rate": last_exam["pass_rate"],
            } if last_exam else None,
        }

    def get_report(self) -> str:
        """Текстовый отчёт о Self-Play"""
        stats = self.get_stats()
        lines = [
            "=== Self-Play Report ===",
            f"Всего оценок: {stats['total_evaluations']}",
            f"Средний балл: {stats['avg_score']}/10",
            f"Порог: {stats['threshold']}",
            f"Reinforced: {stats['reinforced']} ({stats['reinforce_rate']}%)",
            f"Weakened: {stats['weakened']}",
            f"Тренд: {stats['trend']}",
        ]

        if stats['tier_stats']:
            lines.append("\nПо уровням:")
            for tier, data in stats['tier_stats'].items():
                lines.append(f"  {tier}: avg={data['avg_score']}, n={data['count']}")

        if stats['last_exam']:
            lines.append(f"\nПоследний экзамен: avg={stats['last_exam']['avg_score']}, "
                         f"pass={stats['last_exam']['pass_rate']}%")

        lines.append("=" * 24)
        return "\n".join(lines)

    def close(self):
        self._save_state()
        self._conn.close()
