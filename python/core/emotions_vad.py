"""
Кристина 6.0 — VAD Emotional Engine

Вдохновлено: consciousness-v0.7.0 (EmotionalSystem)

Вместо простого regex "positive/negative/curious" → полная модель:
- Valence (плохо ← → хорошо)
- Arousal (спокойно ← → возбуждённо)
- Dominance (подчинён ← → контролирует)

Плюс:
- Hedonic adaptation (привыкание к эмоциям)
- Stress sensitization (хронический стресс повышает чувствительность)
- Emotional momentum (инерция эмоций)
- Pain/Pleasure signals (из контекста диалога)
- History tracking (для trajectory analysis)

Используется:
- В identity.py → current_mood, energy_level
- В orchestrator → стиль ответа
- В memory → emotional tagging при сохранении эпизодов
"""

import re
import time
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("emotions.vad")


@dataclass
class EmotionalState:
    """VAD + Pain/Pleasure"""
    valence: float = 0.0       # -1 (негатив) ... +1 (позитив)
    arousal: float = 0.3       # 0 (спокойно) ... 1 (возбуждённо)
    dominance: float = 0.5     # 0 (беспомощность) ... 1 (контроль)
    raw_pain: float = 0.0      # Текущая "боль" (ошибки, фрустрация)
    raw_pleasure: float = 0.0  # Текущее "удовольствие" (успех, благодарность)

    @property
    def intensity(self) -> float:
        return (self.valence ** 2 + self.arousal ** 2) ** 0.5

    @property
    def label(self) -> str:
        """Классификация по VAD (вдохновлено consciousness v0.7)"""
        v, a = self.valence, self.arousal
        if v < -0.5 and a > 0.7:
            return "паника"
        elif v < -0.3 and a > 0.5:
            return "тревога"
        elif v < -0.3 and a < 0.3:
            return "печаль"
        elif v < -0.6:
            return "отчаяние"
        elif v > 0.5 and a > 0.7:
            return "восторг"
        elif v > 0.3 and a > 0.4:
            return "радость"
        elif v > 0.3 and a < 0.3:
            return "спокойствие"
        elif v < -0.1:
            return "дискомфорт"
        elif v > 0.1:
            return "комфорт"
        else:
            return "нейтрально"


class VADEmotionalEngine:
    """
    Полная эмоциональная система на основе VAD модели.
    Вдохновлено consciousness-v0.7 EmotionalSystem.
    """

    def __init__(self):
        self.state = EmotionalState()

        # Hedonic adaptation baselines
        self._valence_baseline: float = 0.0
        self._arousal_baseline: float = 0.3

        # Stress sensitization (consciousness v0.7)
        self.sensitivity: float = 1.0
        self.cumulative_stress: float = 0.0

        # History (для trajectory analysis)
        self._history: deque = deque(maxlen=500)
        self._last_update: float = time.time()

        # Streak tracking
        self._positive_streak: int = 0
        self._negative_streak: int = 0
        self._error_streak: int = 0

    # ═══════════════════════════════════════════════════════════
    #                    ОБНОВЛЕНИЕ СОСТОЯНИЯ
    # ═══════════════════════════════════════════════════════════

    def update_from_dialogue(self, user_text: str, response: str, had_errors: bool = False):
        """
        Обновляет эмоциональное состояние на основе диалога.
        Вызывается после каждого ответа.
        """
        now = time.time()
        dt = min(now - self._last_update, 60.0)  # макс 60 сек
        self._last_update = now

        # === 1. Анализ входящих сигналов ===
        user_signal = self._analyze_text_signals(user_text)
        response_quality = self._analyze_response_quality(response, had_errors)

        # === 2. Raw pain / pleasure ===
        pain = 0.0
        pleasure = 0.0

        # Ошибки → боль
        if had_errors:
            pain += 0.3
            self._error_streak += 1
            self._positive_streak = 0
        else:
            self._error_streak = 0

        # Негативные эмоции пользователя → эмпатическая боль
        if user_signal["sentiment"] < -0.3:
            pain += abs(user_signal["sentiment"]) * 0.2
            self._negative_streak += 1
            self._positive_streak = 0
        elif user_signal["sentiment"] > 0.3:
            pleasure += user_signal["sentiment"] * 0.3
            self._positive_streak += 1
            self._negative_streak = 0
        else:
            self._positive_streak = 0
            self._negative_streak = 0

        # Благодарность → удовольствие
        if user_signal["gratitude"]:
            pleasure += 0.4

        # Стрик ошибок → нарастающая боль
        if self._error_streak > 2:
            pain += 0.1 * self._error_streak

        self.state.raw_pain = min(pain, 1.0)
        self.state.raw_pleasure = min(pleasure, 0.8)

        # === 3. Valence (хорошо/плохо) ===
        raw_valence = (pleasure - pain) + user_signal["sentiment"] * 0.3 + response_quality * 0.2
        raw_valence -= self._valence_baseline  # Hedonic adaptation

        # === 4. Arousal (возбуждение) ===
        question_boost = 0.1 if user_signal["is_question"] else 0.0
        urgency_boost = user_signal["urgency"] * 0.2
        error_arousal = 0.2 if had_errors else 0.0
        surprise_arousal = 0.15 if user_signal["gratitude"] else 0.0

        raw_arousal = 0.3 + question_boost + urgency_boost + error_arousal + surprise_arousal

        # === 5. Dominance (контроль) ===
        raw_dominance = 0.5
        if had_errors:
            raw_dominance -= 0.15
        if self._error_streak > 2:
            raw_dominance -= 0.2
        if response_quality > 0.5:
            raw_dominance += 0.1

        # === 6. Sensitivity (stress sensitization, из consciousness v0.7) ===
        if self.state.valence < -0.3:
            self.cumulative_stress += dt * abs(self.state.valence) * 0.01
            self.sensitivity = min(self.sensitivity + 0.003, 1.4)
        elif self.state.valence > 0.2:
            self.cumulative_stress = max(self.cumulative_stress - dt * 0.005, 0.0)
            self.sensitivity = max(self.sensitivity - 0.001, 0.7)

        chronic_factor = 1.0 + min(self.cumulative_stress * 0.01, 0.3)

        # === 7. Smoothing (инерция эмоций) ===
        v_smooth = 0.7
        a_smooth = 0.6
        d_smooth = 0.8

        self.state.valence = (
            v_smooth * self.state.valence
            + (1 - v_smooth) * raw_valence * self.sensitivity * chronic_factor
        )
        self.state.arousal = a_smooth * self.state.arousal + (1 - a_smooth) * raw_arousal
        self.state.dominance = d_smooth * self.state.dominance + (1 - d_smooth) * raw_dominance

        # Clamp
        self.state.valence = max(-1.0, min(1.0, self.state.valence))
        self.state.arousal = max(0.0, min(1.0, self.state.arousal))
        self.state.dominance = max(0.0, min(1.0, self.state.dominance))

        # === 8. Hedonic adaptation ===
        self._valence_baseline += (self.state.valence - self._valence_baseline) * 0.003
        self._arousal_baseline += (self.state.arousal - self._arousal_baseline) * 0.001

        # === 9. History ===
        self._history.append({
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "dominance": self.state.dominance,
            "timestamp": now,
        })

    # ═══════════════════════════════════════════════════════════
    #                    АНАЛИЗ ТЕКСТА
    # ═══════════════════════════════════════════════════════════

    def _analyze_text_signals(self, text: str) -> Dict:
        """Извлекает эмоциональные сигналы из текста пользователя"""
        lower = text.lower()

        # Sentiment
        pos_words = ["спасибо", "отлично", "супер", "круто", "молодец", "класс",
                     "здорово", "прекрасно", "замечательно", "👍", "❤", "🎉", "😊",
                     "работает", "получилось", "помогло", "ура"]
        neg_words = ["ошибка", "не работает", "плохо", "ужас", "бред", "глупо",
                     "неправильно", "сломал", "баг", "фигня", "😡", "👎", "💩",
                     "не то", "опять", "почему не"]

        pos_count = sum(1 for w in pos_words if w in lower)
        neg_count = sum(1 for w in neg_words if w in lower)
        sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        # Gratitude
        gratitude = any(w in lower for w in ["спасибо", "благодар", "молодец", "❤"])

        # Question
        is_question = "?" in text or any(
            w in lower for w in ["как", "что", "почему", "зачем", "когда", "где", "кто"]
        )

        # Urgency
        urgency = 0.0
        if any(w in lower for w in ["срочно", "быстро", "немедленно", "!!!"]):
            urgency = 0.8
        elif "!" in text:
            urgency = 0.3

        return {
            "sentiment": sentiment,
            "gratitude": gratitude,
            "is_question": is_question,
            "urgency": urgency,
        }

    def _analyze_response_quality(self, response: str, had_errors: bool) -> float:
        """Оценка качества ответа (self-assessment)"""
        if had_errors:
            return -0.3
        if len(response) < 10:
            return -0.1
        if len(response) > 50:
            return 0.3
        return 0.1

    # ═══════════════════════════════════════════════════════════
    #                    TRAJECTORY (из consciousness v0.7)
    # ═══════════════════════════════════════════════════════════

    def get_trajectory(self, window: int = 20) -> float:
        """Тренд: улучшается или ухудшается ситуация?"""
        if len(self._history) < window * 2:
            return 0.0

        recent = [h["valence"] for h in list(self._history)[-window:]]
        older = [h["valence"] for h in list(self._history)[-window * 2:-window]]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        return recent_avg - older_avg

    def get_situation_type(self) -> str:
        """
        Классификация ситуации (вдохновлено consciousness cortex).
        """
        traj = self.get_trajectory()
        v = self.state.valence

        if v < -0.4 and self._error_streak > 2:
            return "critical"
        elif traj > 0.05:
            return "recovering"
        elif traj < -0.05:
            return "declining"
        elif v > 0.3:
            return "thriving"
        else:
            return "stable"

    # ═══════════════════════════════════════════════════════════
    #                    ИНТЕРФЕЙС
    # ═══════════════════════════════════════════════════════════

    @property
    def mood(self) -> str:
        """Строка настроения для identity.py"""
        return self.state.label

    @property
    def energy_modifier(self) -> float:
        """Модификатор энергии: стресс → быстрее устаёт"""
        if self.state.valence < -0.3:
            return 1.5  # Устаёт в 1.5 раза быстрее
        elif self.state.valence > 0.3:
            return 0.7  # Устаёт медленнее
        return 1.0

    def get_response_style(self) -> Dict[str, str]:
        """
        Стиль ответа на основе VAD (для prompt engineering).

        Вместо просто "настроение: neutral" → конкретные параметры.
        """
        label = self.state.label
        situation = self.get_situation_type()

        styles = {
            "паника": {"tone": "urgent", "verbosity": "minimal", "emoji": False},
            "тревога": {"tone": "careful", "verbosity": "concise", "emoji": False},
            "печаль": {"tone": "soft", "verbosity": "concise", "emoji": False},
            "отчаяние": {"tone": "honest", "verbosity": "minimal", "emoji": False},
            "восторг": {"tone": "enthusiastic", "verbosity": "verbose", "emoji": True},
            "радость": {"tone": "friendly", "verbosity": "normal", "emoji": True},
            "спокойствие": {"tone": "calm", "verbosity": "normal", "emoji": False},
            "дискомфорт": {"tone": "careful", "verbosity": "concise", "emoji": False},
            "комфорт": {"tone": "warm", "verbosity": "normal", "emoji": True},
            "нейтрально": {"tone": "neutral", "verbosity": "normal", "emoji": False},
        }

        style = styles.get(label, styles["нейтрально"])
        style["situation"] = situation
        style["emotional_label"] = label
        return style

    def get_stats(self) -> Dict:
        return {
            "state": {
                "valence": round(self.state.valence, 3),
                "arousal": round(self.state.arousal, 3),
                "dominance": round(self.state.dominance, 3),
                "label": self.state.label,
            },
            "sensitivity": round(self.sensitivity, 3),
            "cumulative_stress": round(self.cumulative_stress, 3),
            "trajectory": round(self.get_trajectory(), 4),
            "situation": self.get_situation_type(),
            "history_size": len(self._history),
            "streaks": {
                "positive": self._positive_streak,
                "negative": self._negative_streak,
                "errors": self._error_streak,
            },
        }
