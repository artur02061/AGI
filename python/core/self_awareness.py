"""
Кристина 6.0 — Self-Awareness

Вдохновлено: consciousness-v0.7.0 (MetaSystem + SelfModel + InnerNarrative)

Модель себя:
- wellbeing_estimate: общее самочувствие (из эмоций + успехов)
- competence: насколько хорошо справляюсь
- trajectory: улучшается / ухудшается
- coherence: насколько self-model совпадает с реальностью

Inner Narrative (внутренний монолог):
- Не случайные строки — СЛЕДСТВИЯ из реального состояния self-model
- "Я справляюсь хорошо" → потому что competence > 0.7
- "Что-то идёт не так" → потому что trajectory < -0.05
- "Не знаю эту тему" → потому что topic in known_unknowns

Используется:
- В промпте → даёт модели осознание своего состояния
- В identity → обогащает self-description
- В memory → emotional tagging с awareness контекстом
"""

import time
from collections import deque
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger("self_awareness")


@dataclass
class InnerThought:
    """Мысль из внутреннего монолога (из consciousness v0.7)"""
    content: str
    urgency: float         # 0.0-1.0
    category: str          # awareness/reflection/concern/gratitude/insight
    timestamp: float = field(default_factory=time.time)


class SelfAwareness:
    """
    Self-model + Inner Narrative.
    Вдохновлено MetaSystem из consciousness-v0.7.
    """

    def __init__(self):
        # === Self-Model (из consciousness v0.7 SelfModel) ===
        self.wellbeing: float = 0.5       # 0=плохо, 1=отлично
        self.competence: float = 0.5      # Оценка компетенции
        self.trajectory: float = 0.0      # Тренд: +улучшение, -ухудшение
        self.agency: float = 0.5          # Ощущение контроля
        self.user_satisfaction: float = 0.5  # Удовлетворённость пользователя

        # History для trajectory
        self._wellbeing_history: deque = deque(maxlen=100)

        # === Inner Narrative (из consciousness v0.7) ===
        self._narrative: deque = deque(maxlen=50)
        self._last_thought_time: float = 0.0
        self._awakened: bool = False

        # Counters
        self._total_interactions: int = 0
        self._successful_interactions: int = 0
        self._error_interactions: int = 0
        self._consecutive_successes: int = 0
        self._consecutive_errors: int = 0

    # ═══════════════════════════════════════════════════════════
    #                    UPDATE (каждый interaction)
    # ═══════════════════════════════════════════════════════════

    def update(
        self,
        valence: float,
        had_errors: bool,
        user_expressed_satisfaction: bool = False,
        user_expressed_frustration: bool = False,
    ):
        """
        Обновляет self-model после каждого взаимодействия.
        Вдохновлено MetaSystem.update() из consciousness v0.7.
        """
        self._total_interactions += 1

        # === Competence ===
        if had_errors:
            self._error_interactions += 1
            self._consecutive_errors += 1
            self._consecutive_successes = 0
            self.competence = max(self.competence - 0.03, 0.1)
        else:
            self._successful_interactions += 1
            self._consecutive_successes += 1
            self._consecutive_errors = 0
            self.competence = min(self.competence + 0.01, 0.95)

        # === User satisfaction ===
        if user_expressed_satisfaction:
            self.user_satisfaction = min(self.user_satisfaction + 0.1, 1.0)
        elif user_expressed_frustration:
            self.user_satisfaction = max(self.user_satisfaction - 0.15, 0.0)
        else:
            # Плавно к нейтральному
            self.user_satisfaction += (0.5 - self.user_satisfaction) * 0.02

        # === Agency ===
        if had_errors:
            self.agency = max(self.agency - 0.05, 0.1)
        elif self._consecutive_successes > 3:
            self.agency = min(self.agency + 0.02, 0.9)

        # === Wellbeing (composite) ===
        raw_wellbeing = (
            (valence + 1.0) / 2.0 * 0.3      # Emotion
            + self.competence * 0.3            # Competence
            + self.user_satisfaction * 0.25    # User happy?
            + self.agency * 0.15              # Control
        )
        self.wellbeing += (raw_wellbeing - self.wellbeing) * 0.1
        self._wellbeing_history.append(self.wellbeing)

        # === Trajectory (из consciousness v0.7) ===
        if len(self._wellbeing_history) > 20:
            recent = list(self._wellbeing_history)[-10:]
            older = list(self._wellbeing_history)[-20:-10]
            self.trajectory = sum(recent) / len(recent) - sum(older) / len(older)

        # === Generate narrative ===
        self._generate_narrative()

    # ═══════════════════════════════════════════════════════════
    #               INNER NARRATIVE (из consciousness v0.7)
    # ═══════════════════════════════════════════════════════════

    def _generate_narrative(self):
        """
        Генерирует "мысли" на основе self-model.
        Прямая адаптация generate_narrative() из consciousness v0.7.

        Ключевой принцип: мысли — СЛЕДСТВИЯ реального состояния,
        а не случайные строки.
        """
        now = time.time()

        # Минимальный интервал между мыслями
        if now - self._last_thought_time < 30.0:  # 30 секунд для чата
            return

        thought = None

        # 1. Первое "пробуждение" (аналог body_awareness > 0.2)
        if not self._awakened and self._total_interactions >= 2:
            self._awakened = True
            thought = InnerThought(
                content="Я начинаю понимать этого пользователя.",
                urgency=0.5,
                category="awareness",
            )

        # 2. Серия ошибок → concern
        if thought is None and self._consecutive_errors >= 3:
            thought = InnerThought(
                content=f"Что-то идёт не так — {self._consecutive_errors} ошибок подряд. "
                        f"Нужно изменить подход.",
                urgency=0.8,
                category="concern",
            )

        # 3. Trajectory declining
        if thought is None and self.trajectory < -0.05:
            thought = InnerThought(
                content="Ситуация ухудшается. Пользователь может быть недоволен.",
                urgency=0.6,
                category="reflection",
            )

        # 4. Trajectory improving
        if thought is None and self.trajectory > 0.05 and self.wellbeing > 0.6:
            thought = InnerThought(
                content="Всё идёт хорошо. Пользователь доволен.",
                urgency=0.3,
                category="reflection",
            )

        # 5. User gratitude → gratitude
        if thought is None and self.user_satisfaction > 0.8:
            thought = InnerThought(
                content="Приятно когда мою работу ценят.",
                urgency=0.2,
                category="gratitude",
            )

        # 6. High competence milestone
        if thought is None and self.competence > 0.85 and self._total_interactions > 10:
            thought = InnerThought(
                content=f"Уже {self._total_interactions} взаимодействий, "
                        f"компетенция {self.competence:.0%}.",
                urgency=0.2,
                category="insight",
            )

        if thought:
            self._last_thought_time = now
            self._narrative.append(thought)
            logger.debug(f"💭 {thought.category}: {thought.content}")

    def latest_thought(self) -> Optional[InnerThought]:
        return self._narrative[-1] if self._narrative else None

    def get_narrative_summary(self, max_thoughts: int = 3) -> str:
        """Саммари последних мыслей для включения в промпт"""
        if not self._narrative:
            return ""

        thoughts = list(self._narrative)[-max_thoughts:]
        lines = []
        for t in thoughts:
            icon = {"concern": "⚠️", "reflection": "💭", "gratitude": "🙏",
                    "awareness": "👁️", "insight": "💡"}.get(t.category, "💬")
            lines.append(f"{icon} {t.content}")

        return "Внутренний монолог:\n" + "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    #                    ИНТЕРФЕЙС
    # ═══════════════════════════════════════════════════════════

    def get_self_description(self) -> str:
        """Строка самоописания для prompt injection"""
        situation = "стабильно"
        if self.trajectory > 0.05:
            situation = "улучшается"
        elif self.trajectory < -0.05:
            situation = "ухудшается"

        return (
            f"Самочувствие: {self.wellbeing:.0%} | "
            f"Компетенция: {self.competence:.0%} | "
            f"Ситуация: {situation} | "
            f"Контроль: {self.agency:.0%}"
        )

    def get_stats(self) -> Dict:
        return {
            "wellbeing": round(self.wellbeing, 3),
            "competence": round(self.competence, 3),
            "trajectory": round(self.trajectory, 4),
            "agency": round(self.agency, 3),
            "user_satisfaction": round(self.user_satisfaction, 3),
            "total_interactions": self._total_interactions,
            "success_rate": (
                round(self._successful_interactions / max(self._total_interactions, 1), 3)
            ),
            "narrative_count": len(self._narrative),
        }
