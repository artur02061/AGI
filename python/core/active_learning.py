"""
Кристина 7.2 — Active Learning (Умная неуверенность)

ЗАЧЕМ:
  Claude говорит "я не уверен" и задаёт уточняющие вопросы.
  Кристина должна делать то же самое — ЛУЧШЕ СПРОСИТЬ, ЧЕМ ОШИБИТЬСЯ.

ПРИНЦИП:
  Для каждого запроса Кристина оценивает свою УВЕРЕННОСТЬ:

  confidence >= 0.8  → отвечаю уверенно
  0.5 <= conf < 0.8  → отвечаю + "если неправильно поняла, уточни"
  0.3 <= conf < 0.5  → спрашиваю: "Ты имеешь в виду X или Y?"
  confidence < 0.3   → "Я не уверена, давай уточним..."

ИСТОЧНИКИ УВЕРЕННОСТИ:
  1. IntentRouter confidence (Tier 1/2 score)
  2. Sentence embedding similarity с известными паттернами
  3. Количество неизвестных слов
  4. Неоднозначность (несколько intent-ов с близким score)
  5. История: как часто ошибались на подобных запросах

ОБУЧЕНИЕ:
  - Каждый раз когда Кристина спросила и получила ответ → learn
  - Каждый раз когда ответила неправильно → снизить confidence threshold
  - Каждый раз когда ответила правильно → повысить threshold

ЭФФЕКТ:
  - Меньше ошибок (спрашивает вместо угадывания)
  - Пользователь чувствует что Кристина "думает"
  - Качество ответов растёт через уточнения
"""

import sqlite3
import json
import time
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from utils.logging import get_logger
import config

logger = get_logger("active_learning")

# ═══════════════════════════════════════════════════════════════
#               ПОРОГИ УВЕРЕННОСТИ
# ═══════════════════════════════════════════════════════════════

CONFIDENCE_SURE = 0.80       # Отвечаю уверенно
CONFIDENCE_HEDGED = 0.50     # Отвечаю с оговоркой
CONFIDENCE_ASK = 0.30        # Спрашиваю уточнение
# < CONFIDENCE_ASK            → "Я не уверена..."

# Фразы для разных уровней уверенности
HEDGING_PHRASES = [
    "Если я неправильно поняла, уточни.",
    "Надеюсь, я правильно поняла задачу.",
    "Если нужно по-другому — скажи.",
    "Поправь, если я не так поняла.",
]

CLARIFICATION_TEMPLATES = [
    "Ты имеешь в виду {option_a} или {option_b}?",
    "Уточни: {option_a} или {option_b}?",
    "Мне кажется, ты хочешь {option_a}. Правильно?",
    "Я могу сделать {option_a} или {option_b}. Что именно?",
]

UNCERTAINTY_PHRASES = [
    "Я не совсем уверена, что именно ты хочешь. Можешь уточнить?",
    "Хмм, я не до конца поняла задачу. Расскажи подробнее?",
    "Можешь переформулировать? Хочу понять точнее.",
    "Мне нужно больше деталей, чтобы сделать правильно.",
]


class ActiveLearning:
    """
    Модуль активного обучения — Кристина учится спрашивать.

    Оценивает уверенность в каждом запросе и принимает решение:
    - Ответить уверенно
    - Ответить с оговоркой
    - Задать уточняющий вопрос
    - Признать неуверенность

    Использование:
        al = ActiveLearning(neural_engine, sentence_embeddings)

        # Оценка уверенности
        assessment = al.assess_confidence(user_input, route_result)

        # Получение действия
        if assessment["action"] == "answer":
            # Отвечать уверенно
        elif assessment["action"] == "hedge":
            # Ответить + оговорка
            suffix = assessment["hedge_phrase"]
        elif assessment["action"] == "clarify":
            # Задать уточнение
            question = assessment["clarification"]
        elif assessment["action"] == "uncertain":
            # Признать неуверенность
            response = assessment["uncertainty_phrase"]

        # Обратная связь
        al.feedback(assessment["request_id"], correct=True)
    """

    def __init__(self, neural_engine=None, sentence_embeddings=None, db_path: Path = None):
        self._engine = neural_engine
        self._sentence = sentence_embeddings
        self._db_path = db_path or (config.config.data_dir / "active_learning.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

        # Адаптивные пороги (обучаются через feedback)
        self._thresholds = self._load_thresholds()

        # Кеш ошибочных паттернов (intent → error_count)
        self._error_intents: Counter = Counter()
        self._load_error_stats()

        stats = self.get_stats()
        logger.info(
            f"🎯 ActiveLearning: {stats['total_assessments']} оценок, "
            f"accuracy={stats['accuracy_pct']}%, "
            f"thresholds=({self._thresholds['sure']:.2f}, "
            f"{self._thresholds['hedged']:.2f}, "
            f"{self._thresholds['ask']:.2f})"
        )

    def _create_tables(self):
        cur = self._conn.cursor()

        # История оценок уверенности
        cur.execute("""
            CREATE TABLE IF NOT EXISTS confidence_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                intent TEXT,
                confidence REAL NOT NULL,
                action TEXT NOT NULL,
                was_correct INTEGER DEFAULT -1,
                route_source TEXT,
                details TEXT,
                created_at REAL NOT NULL
            )
        """)

        # Адаптивные пороги
        cur.execute("""
            CREATE TABLE IF NOT EXISTS thresholds (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Статистика ошибок по intent-ам
        cur.execute("""
            CREATE TABLE IF NOT EXISTS intent_errors (
                intent TEXT PRIMARY KEY,
                error_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)

        # Неоднозначные запросы (для обучения)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ambiguous_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                possible_intents TEXT NOT NULL,
                chosen_intent TEXT,
                created_at REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conf_action ON confidence_log(action)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conf_correct ON confidence_log(was_correct)
        """)

        self._conn.commit()

    def _load_thresholds(self) -> Dict[str, float]:
        """Загружает адаптивные пороги"""
        defaults = {
            "sure": CONFIDENCE_SURE,
            "hedged": CONFIDENCE_HEDGED,
            "ask": CONFIDENCE_ASK,
        }
        for key, default in defaults.items():
            row = self._conn.execute(
                "SELECT value FROM thresholds WHERE key = ?", (key,)
            ).fetchone()
            if row:
                defaults[key] = row["value"]
        return defaults

    def _load_error_stats(self):
        """Загружает статистику ошибок по intent-ам"""
        rows = self._conn.execute(
            "SELECT intent, error_count FROM intent_errors WHERE error_count > 0"
        ).fetchall()
        self._error_intents = Counter({row["intent"]: row["error_count"] for row in rows})

    # ═══════════════════════════════════════════════════════════════
    #               ОЦЕНКА УВЕРЕННОСТИ
    # ═══════════════════════════════════════════════════════════════

    def assess_confidence(
        self,
        user_input: str,
        route_result: Optional[Dict] = None,
        alternative_intents: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Оценивает уверенность Кристины в понимании запроса.

        Args:
            user_input: текст запроса пользователя
            route_result: результат IntentRouter.route() (может быть None)
            alternative_intents: альтернативные варианты intent-ов

        Returns:
            Dict с полями:
            - confidence: float (0.0 - 1.0)
            - action: "answer" | "hedge" | "clarify" | "uncertain"
            - request_id: int (для feedback)
            - hedge_phrase: str (если action == "hedge")
            - clarification: str (если action == "clarify")
            - uncertainty_phrase: str (если action == "uncertain")
            - details: Dict (подробности расчёта)
        """
        import random

        # Собираем сигналы уверенности
        signals = self._collect_signals(user_input, route_result, alternative_intents)

        # Вычисляем общую уверенность
        confidence = self._compute_confidence(signals)

        # Определяем действие
        action, extra = self._decide_action(
            confidence, signals, user_input, route_result
        )

        # Логируем
        now = time.time()
        intent = route_result.get("intent", "unknown") if route_result else "none"
        details_json = json.dumps(signals, ensure_ascii=False, default=str)

        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO confidence_log
            (user_input, intent, confidence, action, route_source, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_input, intent, confidence, action,
            route_result.get("source", "none") if route_result else "none",
            details_json, now,
        ))
        request_id = cur.lastrowid
        self._conn.commit()

        result = {
            "confidence": round(confidence, 3),
            "action": action,
            "request_id": request_id,
            "intent": intent,
            "details": signals,
        }
        result.update(extra)

        logger.debug(
            f"🎯 Confidence: {confidence:.2f} → {action} "
            f"for '{user_input[:50]}' (intent={intent})"
        )

        return result

    def _collect_signals(
        self,
        user_input: str,
        route_result: Optional[Dict],
        alternative_intents: Optional[List[Dict]],
    ) -> Dict[str, float]:
        """Собирает все сигналы для оценки уверенности"""
        signals = {}

        # 1. Route confidence (от IntentRouter)
        if route_result:
            signals["route_confidence"] = route_result.get("confidence", 0.0)
            signals["route_source"] = {
                "learned": 0.9,   # Выученный паттерн — высокая уверенность
                "rule": 0.85,     # Regex правило — высокая
            }.get(route_result.get("source", ""), 0.5)
        else:
            signals["route_confidence"] = 0.0
            signals["route_source"] = 0.0

        # 2. Неизвестные слова
        if self._engine:
            analysis = self._engine.understand_sentence(user_input)
            known_pct = analysis.get("understood_pct", 0.0) / 100.0
            signals["known_words"] = known_pct
        else:
            signals["known_words"] = 0.5

        # 3. Длина запроса (очень короткие и очень длинные — менее уверенны)
        words = user_input.split()
        if len(words) <= 1:
            signals["length_signal"] = 0.3   # Слишком короткий
        elif len(words) <= 5:
            signals["length_signal"] = 0.9   # Оптимальный
        elif len(words) <= 15:
            signals["length_signal"] = 0.7   # Нормальный
        else:
            signals["length_signal"] = 0.5   # Длинный, сложный

        # 4. Неоднозначность (несколько intent-ов с близким score)
        if alternative_intents and len(alternative_intents) >= 2:
            scores = sorted(
                [a.get("confidence", 0) for a in alternative_intents],
                reverse=True,
            )
            gap = scores[0] - scores[1] if len(scores) >= 2 else 1.0
            signals["ambiguity"] = min(1.0, gap * 2)  # Большой gap = низкая неоднозначность
        else:
            signals["ambiguity"] = 0.8  # Нет альтернатив = средняя уверенность

        # 5. Историческая точность для этого intent-а
        if route_result:
            intent = route_result.get("intent", "")
            error_count = self._error_intents.get(intent, 0)
            if error_count > 3:
                signals["historical"] = 0.3  # Много ошибок на этом intent-е
            elif error_count > 0:
                signals["historical"] = 0.6
            else:
                signals["historical"] = 0.9
        else:
            signals["historical"] = 0.5

        # 6. Наличие вопросительных слов (запрос = вопрос → проще ответить)
        question_words = {"что", "как", "где", "когда", "зачем", "почему", "кто", "сколько"}
        has_question = any(w in user_input.lower().split() for w in question_words)
        signals["is_question"] = 0.8 if has_question else 0.6

        return signals

    def _compute_confidence(self, signals: Dict[str, float]) -> float:
        """
        Вычисляет общую уверенность из сигналов.
        Взвешенное среднее с приоритетом на route_confidence.
        """
        weights = {
            "route_confidence": 3.0,  # Самый важный сигнал
            "route_source": 1.5,
            "known_words": 1.0,
            "length_signal": 0.5,
            "ambiguity": 2.0,         # Неоднозначность важна
            "historical": 1.5,
            "is_question": 0.3,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in weights.items():
            if key in signals:
                weighted_sum += signals[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return min(1.0, max(0.0, weighted_sum / total_weight))

    def _decide_action(
        self,
        confidence: float,
        signals: Dict,
        user_input: str,
        route_result: Optional[Dict],
    ) -> Tuple[str, Dict]:
        """Решает какое действие предпринять"""
        import random

        if confidence >= self._thresholds["sure"]:
            return "answer", {}

        if confidence >= self._thresholds["hedged"]:
            return "hedge", {
                "hedge_phrase": random.choice(HEDGING_PHRASES),
            }

        if confidence >= self._thresholds["ask"]:
            # Формируем уточняющий вопрос
            clarification = self._generate_clarification(
                user_input, route_result, signals
            )
            return "clarify", {
                "clarification": clarification,
            }

        return "uncertain", {
            "uncertainty_phrase": random.choice(UNCERTAINTY_PHRASES),
        }

    def _generate_clarification(
        self,
        user_input: str,
        route_result: Optional[Dict],
        signals: Dict,
    ) -> str:
        """Генерирует уточняющий вопрос"""
        import random

        intent = route_result.get("intent", "") if route_result else ""

        # Если есть intent но низкая уверенность — спрашиваем подтверждение
        if intent:
            intent_descriptions = {
                "create_file": "создать файл",
                "delete_file": "удалить файл",
                "read_file": "прочитать файл",
                "web_search": "поискать в интернете",
                "launch_app": "запустить приложение",
                "greeting": "просто поболтать",
                "explanation": "объяснить что-то",
                "creative": "написать что-то творческое",
            }
            desc = intent_descriptions.get(intent, intent)
            return f"Мне кажется, ты хочешь {desc}. Правильно?"

        # Если нет intent — общий вопрос
        return random.choice(UNCERTAINTY_PHRASES)

    # ═══════════════════════════════════════════════════════════════
    #               ОБРАТНАЯ СВЯЗЬ
    # ═══════════════════════════════════════════════════════════════

    def feedback(self, request_id: int, correct: bool):
        """
        Обратная связь: правильно ли Кристина поняла запрос.

        Вызывается после завершения обработки:
        - correct=True  → пользователь доволен
        - correct=False → пользователь недоволен / уточнил
        """
        now = time.time()

        # Обновляем лог
        row = self._conn.execute(
            "SELECT intent, confidence, action FROM confidence_log WHERE id = ?",
            (request_id,)
        ).fetchone()

        if not row:
            return

        self._conn.execute(
            "UPDATE confidence_log SET was_correct = ? WHERE id = ?",
            (1 if correct else 0, request_id)
        )

        intent = row["intent"]
        confidence = row["confidence"]
        action = row["action"]

        # Обновляем статистику intent-а
        if correct:
            self._conn.execute("""
                INSERT INTO intent_errors (intent, success_count, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(intent)
                DO UPDATE SET success_count = success_count + 1, updated_at = ?
            """, (intent, now, now))
        else:
            self._error_intents[intent] += 1
            self._conn.execute("""
                INSERT INTO intent_errors (intent, error_count, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(intent)
                DO UPDATE SET error_count = error_count + 1, updated_at = ?
            """, (intent, now, now))

        # Адаптация порогов
        self._adapt_thresholds(confidence, action, correct)

        self._conn.commit()

        logger.debug(
            f"🎯 Feedback: request={request_id}, correct={correct}, "
            f"intent={intent}, action={action}"
        )

    def _adapt_thresholds(self, confidence: float, action: str, correct: bool):
        """
        Адаптирует пороги на основе обратной связи.

        Если Кристина ответила уверенно и ОШИБЛА → повысить порог sure
        Если Кристина спросила и ответ был бы ПРАВИЛЬНЫМ → понизить порог ask
        """
        adjustment = 0.01  # Маленький шаг

        if action == "answer" and not correct:
            # Была слишком уверена → повысить порог
            self._thresholds["sure"] = min(0.95, self._thresholds["sure"] + adjustment)

        elif action == "hedge" and not correct:
            # Даже с оговоркой ошиблась → повысить порог hedged
            self._thresholds["hedged"] = min(
                self._thresholds["sure"] - 0.05,
                self._thresholds["hedged"] + adjustment,
            )

        elif action in ("clarify", "uncertain") and correct:
            # Спросила, но ответ был бы правильным → понизить порог
            self._thresholds["ask"] = max(0.1, self._thresholds["ask"] - adjustment)
            self._thresholds["hedged"] = max(
                self._thresholds["ask"] + 0.05,
                self._thresholds["hedged"] - adjustment,
            )

        elif action == "answer" and correct:
            # Правильно ответила уверенно → можно немного понизить порог
            self._thresholds["sure"] = max(0.6, self._thresholds["sure"] - adjustment * 0.5)

        # Сохраняем
        now = time.time()
        for key, value in self._thresholds.items():
            self._conn.execute("""
                INSERT INTO thresholds (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
            """, (key, value, now, value, now))

    # ═══════════════════════════════════════════════════════════════
    #               СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Статистика активного обучения"""
        total = self._conn.execute(
            "SELECT COUNT(*) as c FROM confidence_log"
        ).fetchone()["c"]

        correct = self._conn.execute(
            "SELECT COUNT(*) as c FROM confidence_log WHERE was_correct = 1"
        ).fetchone()["c"]

        incorrect = self._conn.execute(
            "SELECT COUNT(*) as c FROM confidence_log WHERE was_correct = 0"
        ).fetchone()["c"]

        evaluated = correct + incorrect
        accuracy = round(correct / evaluated * 100, 1) if evaluated > 0 else 0.0

        # Распределение по действиям
        actions = {}
        rows = self._conn.execute(
            "SELECT action, COUNT(*) as c FROM confidence_log GROUP BY action"
        ).fetchall()
        for row in rows:
            actions[row["action"]] = row["c"]

        return {
            "total_assessments": total,
            "evaluated": evaluated,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy_pct": accuracy,
            "actions": actions,
            "thresholds": dict(self._thresholds),
            "problematic_intents": dict(self._error_intents.most_common(5)),
        }

    def get_improvement_suggestions(self) -> List[str]:
        """
        Анализирует ошибки и даёт рекомендации.
        Полезно для self-improvement.
        """
        suggestions = []

        # Проблемные intent-ы
        for intent, count in self._error_intents.most_common(3):
            if count >= 3:
                suggestions.append(
                    f"Intent '{intent}' имеет {count} ошибок — "
                    f"нужно больше обучающих примеров или уточнение правил"
                )

        # Слишком много uncertain
        stats = self.get_stats()
        uncertain_count = stats["actions"].get("uncertain", 0)
        if stats["total_assessments"] > 10 and uncertain_count > stats["total_assessments"] * 0.3:
            suggestions.append(
                "Слишком много неуверенных ответов (>30%) — "
                "нужно расширить базу паттернов"
            )

        # Низкая accuracy
        if stats["accuracy_pct"] < 70 and stats["evaluated"] > 10:
            suggestions.append(
                f"Accuracy {stats['accuracy_pct']}% ниже 70% — "
                f"пороги нужно повысить или добавить обучающих данных"
            )

        return suggestions

    def close(self):
        self._conn.commit()
        self._conn.close()
