"""
Кристина 7.3 — Meta-Learning (Мета-обучение / Learning-to-Learn)

ЗАЧЕМ:
  Кристина обучает много компонентов одновременно:
    - MicroTransformer (генерация текста)
    - MoE (маршрутизация экспертов)
    - KnowledgeDistillation (дистилляция LLM)
    - ConditionalGen (условная генерация)
    - ResponseGenerator (правила → ответы)
    - IntentRouter (маршрутизация запросов)
    - Word2Vec (эмбеддинги слов)

  Но КАК обучать каждый из них ЛУЧШЕ?
    - Кому сейчас нужно больше данных?
    - Чей learning rate слишком высокий/низкий?
    - Кто застрял на плато?
    - Какие данные каждому полезны?

  META-LEARNING отвечает на эти вопросы.
  Это "обучение обучению" — система, которая ОПТИМИЗИРУЕТ процесс обучения.

АРХИТЕКТУРА:
  ┌─────────────────────────────────────────────────────┐
  │ MetaLearner                                         │
  │                                                     │
  │  ┌──────────────┐    ┌───────────────────────────┐  │
  │  │ LearnerProfile│×N │ SchedulerStrategy         │  │
  │  │ - loss_history│    │ - adaptive_lr             │  │
  │  │ - lr          │    │ - plateau_detection       │  │
  │  │ - is_plateau  │    │ - curriculum_ordering     │  │
  │  │ - importance  │    │ - resource_allocation     │  │
  │  └──────────────┘    └───────────────────────────┘  │
  │                                                     │
  │  ┌──────────────────────────────────────────────┐   │
  │  │ CurriculumScheduler                          │   │
  │  │  - Порядок обучения: от простого к сложному  │   │
  │  │  - Фокус: больше данных туда, где нужнее     │   │
  │  │  - Баланс: exploration vs exploitation       │   │
  │  └──────────────────────────────────────────────┘   │
  │                                                     │
  │  ┌──────────────────────────────────────────────┐   │
  │  │ PerformanceTracker                           │   │
  │  │  - Метрики каждого компонента                │   │
  │  │  - Тренды (улучшается/стагнирует/деградирует)│   │
  │  │  - Корреляции между компонентами             │   │
  │  └──────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────┘
"""

import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import get_logger
import config

logger = get_logger("meta_learning")


# ═══════════════════════════════════════════════════════════════
#               LEARNER PROFILE
# ═══════════════════════════════════════════════════════════════


class Trend(Enum):
    IMPROVING = "improving"
    PLATEAU = "plateau"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class LearnerProfile:
    """
    Профиль одного обучаемого компонента.
    Meta-Learning отслеживает и оптимизирует каждый.
    """
    name: str
    # Learning rate
    base_lr: float = 3e-4
    current_lr: float = 3e-4
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    # Loss tracking
    loss_history: List[float] = field(default_factory=list)
    loss_window: int = 20       # Скользящее окно для тренда
    # Stats
    total_steps: int = 0
    total_improvements: int = 0
    plateau_count: int = 0      # Сколько раз был plateau
    # Importance (meta-learned)
    importance: float = 1.0     # Насколько этот компонент важен
    # Training probability
    train_prob: float = 1.0     # Вероятность обучения на данном шаге
    # Trend
    trend: Trend = Trend.UNKNOWN

    def record_loss(self, loss: float):
        """Записывает loss и обновляет тренд"""
        self.loss_history.append(loss)
        if len(self.loss_history) > 200:
            self.loss_history = self.loss_history[-200:]
        self.total_steps += 1
        self._update_trend()

    def _update_trend(self):
        """Определяет тренд по последним loss-значениям"""
        if len(self.loss_history) < self.loss_window:
            self.trend = Trend.UNKNOWN
            return

        recent = self.loss_history[-self.loss_window:]
        older = self.loss_history[-self.loss_window * 2:-self.loss_window] \
            if len(self.loss_history) >= self.loss_window * 2 \
            else self.loss_history[:self.loss_window]

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        ratio = avg_recent / (avg_older + 1e-10)

        if ratio < 0.95:
            self.trend = Trend.IMPROVING
            self.total_improvements += 1
        elif ratio > 1.05:
            self.trend = Trend.DEGRADING
        else:
            self.trend = Trend.PLATEAU
            self.plateau_count += 1

    def avg_recent_loss(self) -> float:
        if not self.loss_history:
            return float('inf')
        window = min(10, len(self.loss_history))
        return sum(self.loss_history[-window:]) / window

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "current_lr": self.current_lr,
            "total_steps": self.total_steps,
            "avg_loss": round(self.avg_recent_loss(), 6),
            "trend": self.trend.value,
            "importance": round(self.importance, 3),
            "train_prob": round(self.train_prob, 3),
            "plateau_count": self.plateau_count,
        }


# ═══════════════════════════════════════════════════════════════
#               ADAPTIVE LR SCHEDULER
# ═══════════════════════════════════════════════════════════════


class AdaptiveLRScheduler:
    """
    Адаптивный learning rate для каждого компонента.

    Стратегии:
    - Reduce on plateau: lr *= 0.5 при стагнации
    - Warmup: линейный рост lr на первых N шагах
    - Cosine annealing: плавное снижение
    - Importance-weighted: больше lr важным компонентам
    """

    def __init__(self, warmup_steps: int = 50):
        self.warmup_steps = warmup_steps

    def step(self, profile: LearnerProfile) -> float:
        """
        Вычисляет новый learning rate для компонента.

        Returns:
            Обновлённый lr
        """
        lr = profile.current_lr

        # 1. Warmup (первые N шагов)
        if profile.total_steps < self.warmup_steps:
            warmup_factor = (profile.total_steps + 1) / self.warmup_steps
            lr = profile.base_lr * warmup_factor
            profile.current_lr = lr
            return lr

        # 2. Plateau detection → reduce
        if profile.trend == Trend.PLATEAU:
            lr *= 0.8  # Мягкое снижение
            if profile.plateau_count > 3:
                lr *= 0.5  # Более агрессивное при повторных plateau

        # 3. Degradation → значительное снижение
        elif profile.trend == Trend.DEGRADING:
            lr *= 0.5

        # 4. Improving → можно немного увеличить
        elif profile.trend == Trend.IMPROVING:
            lr *= 1.05

        # 5. Cosine component (мягкое затухание со временем)
        decay_steps = max(1, profile.total_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * min(decay_steps / 5000, 1.0)))
        cosine_lr = profile.lr_min + (profile.base_lr - profile.lr_min) * cosine_factor

        # Blend: 70% adaptive + 30% cosine
        lr = 0.7 * lr + 0.3 * cosine_lr

        # Clamp
        lr = max(profile.lr_min, min(profile.lr_max, lr))

        profile.current_lr = lr
        return lr


# ═══════════════════════════════════════════════════════════════
#               CURRICULUM SCHEDULER
# ═══════════════════════════════════════════════════════════════


class CurriculumScheduler:
    """
    Curriculum Learning: определяет порядок и приоритет обучения.

    Принципы:
    1. Компоненты с бОльшим потенциалом улучшения → больше обучения
    2. Компоненты на plateau → меньше обучения (экономия ресурсов)
    3. Exploration: иногда обучаем "забытые" компоненты
    4. Dependency-aware: base components first
    """

    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self._step = 0

    def compute_train_probabilities(
        self,
        profiles: Dict[str, LearnerProfile],
    ) -> Dict[str, float]:
        """
        Вычисляет вероятность обучения для каждого компонента.

        Returns:
            {component_name: probability}
        """
        self._step += 1
        probs = {}

        for name, profile in profiles.items():
            prob = self._compute_single_prob(profile)
            probs[name] = prob
            profile.train_prob = prob

        return probs

    def _compute_single_prob(self, profile: LearnerProfile) -> float:
        """Вычисляет вероятность обучения одного компонента"""
        base_prob = 1.0

        # Trend-based adjustment
        if profile.trend == Trend.IMPROVING:
            base_prob = 1.0  # Продолжаем — работает!
        elif profile.trend == Trend.PLATEAU:
            base_prob = 0.3  # Снижаем — мало пользы
        elif profile.trend == Trend.DEGRADING:
            base_prob = 0.5  # Снижаем, но не убираем (может исправиться)
        else:
            base_prob = 0.8  # Unknown — осторожно обучаем

        # Importance weighting
        base_prob *= profile.importance

        # Exploration: случайный шанс обучить даже "неважный" компонент
        if random.random() < self.exploration_rate:
            base_prob = max(base_prob, 0.5)

        return min(1.0, max(0.05, base_prob))

    def should_train(self, profile: LearnerProfile) -> bool:
        """Решает, обучать ли компонент на этом шаге"""
        return random.random() < profile.train_prob


# ═══════════════════════════════════════════════════════════════
#               PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════


class PerformanceTracker:
    """
    Отслеживает общую производительность системы и вклад компонентов.
    """

    def __init__(self):
        self._response_quality: List[float] = []  # 0-1
        self._tier_distribution: Dict[str, int] = {}
        self._component_contributions: Dict[str, List[float]] = {}

    def record_response(
        self,
        quality: float,
        tier: str,
        contributing_components: List[str] = None,
    ):
        """Записывает качество ответа и участвующие компоненты"""
        self._response_quality.append(quality)
        if len(self._response_quality) > 500:
            self._response_quality = self._response_quality[-500:]

        self._tier_distribution[tier] = self._tier_distribution.get(tier, 0) + 1

        if contributing_components:
            for comp in contributing_components:
                if comp not in self._component_contributions:
                    self._component_contributions[comp] = []
                self._component_contributions[comp].append(quality)
                if len(self._component_contributions[comp]) > 200:
                    self._component_contributions[comp] = \
                        self._component_contributions[comp][-200:]

    def compute_importance(self, profiles: Dict[str, LearnerProfile]):
        """
        Вычисляет importance каждого компонента по его вкладу в качество.
        """
        for name, profile in profiles.items():
            contributions = self._component_contributions.get(name, [])
            if len(contributions) >= 5:
                avg_quality = sum(contributions) / len(contributions)
                # Importance пропорциональна среднему качеству и частоте использования
                frequency = len(contributions) / max(1, len(self._response_quality))
                profile.importance = 0.7 * avg_quality + 0.3 * frequency
            else:
                profile.importance = 0.5  # Default для новых компонентов

    def avg_quality(self, window: int = 50) -> float:
        if not self._response_quality:
            return 0.0
        recent = self._response_quality[-window:]
        return sum(recent) / len(recent)

    def quality_trend(self) -> Trend:
        """Тренд общего качества"""
        if len(self._response_quality) < 20:
            return Trend.UNKNOWN

        recent = self._response_quality[-10:]
        older = self._response_quality[-20:-10]

        avg_r = sum(recent) / len(recent)
        avg_o = sum(older) / len(older)

        if avg_r > avg_o * 1.05:
            return Trend.IMPROVING
        elif avg_r < avg_o * 0.95:
            return Trend.DEGRADING
        return Trend.PLATEAU

    def get_stats(self) -> Dict:
        return {
            "avg_quality": round(self.avg_quality(), 4),
            "quality_trend": self.quality_trend().value,
            "total_responses": len(self._response_quality),
            "tier_distribution": dict(self._tier_distribution),
        }


# ═══════════════════════════════════════════════════════════════
#               META-LEARNER (главный модуль)
# ═══════════════════════════════════════════════════════════════


# Все управляемые компоненты
MANAGED_COMPONENTS = [
    "micro_transformer",
    "moe",
    "conditional_gen",
    "knowledge_distillation",
    "response_generator",
    "intent_router",
    "word2vec",
]


class MetaLearner:
    """
    Meta-Learning: система, которая учится УЧИТЬ другие компоненты.

    Использование:
        meta = MetaLearner()

        # Регистрация компонента
        meta.register("micro_transformer", base_lr=3e-4)

        # Перед обучением — получить рекомендации
        should = meta.should_train("micro_transformer")
        lr = meta.get_lr("micro_transformer")

        # После обучения — сообщить результат
        meta.report_loss("micro_transformer", loss=0.42)

        # После ответа — сообщить качество
        meta.report_response(quality=0.8, tier="tier1",
                            components=["micro_transformer", "moe"])

        # Периодически — оптимизировать
        meta.optimize_step()
    """

    def __init__(self, db_path: Path = None):
        # Profiles
        self.profiles: Dict[str, LearnerProfile] = {}

        # Sub-systems
        self.lr_scheduler = AdaptiveLRScheduler()
        self.curriculum = CurriculumScheduler()
        self.performance = PerformanceTracker()

        # Persistence
        self._db_path = db_path or (config.config.data_dir / "meta_learning.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Global stats
        self._total_meta_steps = 0
        self._load_state()

        # Register default components
        for comp in MANAGED_COMPONENTS:
            if comp not in self.profiles:
                self.register(comp)

        logger.info(
            f"🧬 MetaLearner: {len(self.profiles)} components, "
            f"{self._total_meta_steps} meta-steps"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                component TEXT,
                data TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def _load_state(self):
        row = self._conn.execute(
            "SELECT value FROM meta_state WHERE key = 'state'"
        ).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                self._total_meta_steps = data.get("total_meta_steps", 0)
                for name, pdata in data.get("profiles", {}).items():
                    profile = LearnerProfile(name=name)
                    profile.base_lr = pdata.get("base_lr", 3e-4)
                    profile.current_lr = pdata.get("current_lr", 3e-4)
                    profile.total_steps = pdata.get("total_steps", 0)
                    profile.total_improvements = pdata.get("total_improvements", 0)
                    profile.plateau_count = pdata.get("plateau_count", 0)
                    profile.importance = pdata.get("importance", 1.0)
                    profile.train_prob = pdata.get("train_prob", 1.0)
                    profile.loss_history = pdata.get("loss_history", [])[-100:]
                    trend_str = pdata.get("trend", "unknown")
                    profile.trend = Trend(trend_str)
                    self.profiles[name] = profile
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_state(self):
        data = {
            "total_meta_steps": self._total_meta_steps,
            "profiles": {
                name: {
                    "base_lr": p.base_lr,
                    "current_lr": p.current_lr,
                    "total_steps": p.total_steps,
                    "total_improvements": p.total_improvements,
                    "plateau_count": p.plateau_count,
                    "importance": p.importance,
                    "train_prob": p.train_prob,
                    "loss_history": p.loss_history[-100:],
                    "trend": p.trend.value,
                }
                for name, p in self.profiles.items()
            },
        }
        json_str = json.dumps(data)
        self._conn.execute("""
            INSERT INTO meta_state (key, value) VALUES ('state', ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (json_str, json_str))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           COMPONENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def register(
        self,
        name: str,
        base_lr: float = 3e-4,
        importance: float = 1.0,
    ):
        """Регистрирует компонент для мета-обучения"""
        if name not in self.profiles:
            self.profiles[name] = LearnerProfile(
                name=name,
                base_lr=base_lr,
                current_lr=base_lr,
                importance=importance,
            )

    # ═══════════════════════════════════════════════════════════════
    #           TRAINING DECISIONS
    # ═══════════════════════════════════════════════════════════════

    def should_train(self, component: str) -> bool:
        """Решает, обучать ли компонент сейчас"""
        profile = self.profiles.get(component)
        if not profile:
            return True
        return self.curriculum.should_train(profile)

    def get_lr(self, component: str) -> float:
        """Возвращает текущий learning rate для компонента"""
        profile = self.profiles.get(component)
        if not profile:
            return 3e-4
        return profile.current_lr

    # ═══════════════════════════════════════════════════════════════
    #           REPORTING
    # ═══════════════════════════════════════════════════════════════

    def report_loss(self, component: str, loss: float):
        """Сообщает loss после шага обучения"""
        profile = self.profiles.get(component)
        if not profile:
            self.register(component)
            profile = self.profiles[component]

        profile.record_loss(loss)

        # Update LR
        self.lr_scheduler.step(profile)

    def report_response(
        self,
        quality: float,
        tier: str,
        components: List[str] = None,
    ):
        """Сообщает качество ответа и участвующие компоненты"""
        self.performance.record_response(
            quality=quality,
            tier=tier,
            contributing_components=components,
        )

    # ═══════════════════════════════════════════════════════════════
    #           META-OPTIMIZATION STEP
    # ═══════════════════════════════════════════════════════════════

    def optimize_step(self):
        """
        Один шаг мета-оптимизации.
        Вызывается периодически (каждые N запросов).

        Обновляет:
        1. Training probabilities для каждого компонента
        2. Importance scores
        3. Learning rates
        """
        self._total_meta_steps += 1

        # 1. Update importance from performance
        self.performance.compute_importance(self.profiles)

        # 2. Update training probabilities
        self.curriculum.compute_train_probabilities(self.profiles)

        # 3. Update LRs
        for profile in self.profiles.values():
            self.lr_scheduler.step(profile)

        # 4. Log meta-state
        if self._total_meta_steps % 10 == 0:
            self._log_meta_state()

        # 5. Save
        if self._total_meta_steps % 5 == 0:
            self._save_state()

    def _log_meta_state(self):
        """Логирует текущее мета-состояние"""
        improving = sum(1 for p in self.profiles.values() if p.trend == Trend.IMPROVING)
        plateau = sum(1 for p in self.profiles.values() if p.trend == Trend.PLATEAU)
        degrading = sum(1 for p in self.profiles.values() if p.trend == Trend.DEGRADING)

        avg_q = self.performance.avg_quality()

        logger.info(
            f"🧬 Meta step #{self._total_meta_steps}: "
            f"quality={avg_q:.3f}, "
            f"trends: {improving}↑ {plateau}→ {degrading}↓"
        )

        # Log individual components with issues
        for name, profile in self.profiles.items():
            if profile.trend == Trend.DEGRADING:
                logger.warning(
                    f"🧬 {name}: DEGRADING (lr={profile.current_lr:.6f}, "
                    f"loss={profile.avg_recent_loss():.4f})"
                )
            elif profile.trend == Trend.PLATEAU and profile.plateau_count > 2:
                logger.info(
                    f"🧬 {name}: persistent plateau "
                    f"(count={profile.plateau_count}, lr={profile.current_lr:.6f})"
                )

        # Record event
        self._conn.execute("""
            INSERT INTO meta_events (event_type, data, created_at)
            VALUES ('meta_step', ?, ?)
        """, (json.dumps({
            "step": self._total_meta_steps,
            "avg_quality": avg_q,
            "improving": improving,
            "plateau": plateau,
            "degrading": degrading,
        }), time.time()))

    # ═══════════════════════════════════════════════════════════════
    #           RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════

    def get_recommendations(self) -> List[str]:
        """
        Генерирует рекомендации по обучению.
        """
        recs = []

        for name, profile in self.profiles.items():
            if profile.trend == Trend.DEGRADING:
                recs.append(
                    f"{name}: деградирует — снизить lr или увеличить данные"
                )
            elif profile.trend == Trend.PLATEAU and profile.plateau_count > 5:
                recs.append(
                    f"{name}: длительный plateau — попробовать lr restart"
                )

        # Overall quality
        q_trend = self.performance.quality_trend()
        if q_trend == Trend.DEGRADING:
            recs.append("Общее качество снижается — проверить данные обучения")
        elif q_trend == Trend.IMPROVING:
            recs.append("Общее качество растёт — продолжать текущую стратегию")

        # Resource allocation
        high_importance = sorted(
            self.profiles.values(),
            key=lambda p: p.importance,
            reverse=True,
        )
        if high_importance:
            top = high_importance[0]
            if top.train_prob < 0.5:
                recs.append(
                    f"{top.name}: высокая важность, но низкая train_prob — "
                    f"увеличить ресурсы"
                )

        return recs

    # ═══════════════════════════════════════════════════════════════
    #           STATISTICS
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        component_stats = {
            name: profile.to_dict()
            for name, profile in self.profiles.items()
        }

        return {
            "total_meta_steps": self._total_meta_steps,
            "components": component_stats,
            "performance": self.performance.get_stats(),
            "recommendations": self.get_recommendations(),
        }

    def close(self):
        self._save_state()
        self._conn.close()
